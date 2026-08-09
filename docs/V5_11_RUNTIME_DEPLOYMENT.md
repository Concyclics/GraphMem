# GraphMem V5.11 查询平面部署与参数说明

本文档对应已经测量的 V5.11 查询性能路径。构建参数与在线查询参数被刻意拆分：调整 Worker、缓存或排队策略不会改变图构建的 `config_hash`，也不会让冻结实验图失效。

## 1. 运行入口

推荐的平衡配置：

```bash
python scripts/serve_v5_11.py \
  --db /path/to/graphmem.sqlite \
  --runtime-config configs/v5/runtime_v5_11_balanced.json \
  --compiled-cache-dir /trusted/local/compiled_views \
  --cpu-ids 0-7
```

仅检查配置展开结果，不启动服务：

```bash
python scripts/serve_v5_11.py \
  --db /path/to/graphmem.sqlite \
  --runtime-config configs/v5/runtime_v5_11_balanced.json \
  --validate-only
```

查询接口：

```bash
curl -s http://127.0.0.1:8090/v1/retrieve \
  -H 'content-type: application/json' \
  -d '{
    "memory_id": "user-42:memory-7",
    "tenant_id": "user-42",
    "query": "上周 Alice 最后确定了什么计划？",
    "timeout_seconds": 5
  }'
```

健康和运行指标分别位于 `GET /healthz`、`GET /v1/stats`。HTTP 层使用有界 admission；超出全局或租户配额时返回 429，超过 deadline 返回 504，不会把请求无限堆积在进程池中。

## 2. 冻结的推荐 profile

| Profile | Worker | Affinity replica | 每 Worker cache | Queue | 适用场景 |
|---|---:|---:|---:|---:|---|
| `runtime_v5_11_balanced.json` | 8 | 2 | 8 memories / 256 MiB | 64 | 默认吞吐、tail、内存平衡点 |
| `runtime_v5_11_low_latency.json` | 8 | 3 | 8 memories / 256 MiB | 32 | 热点副本更多、tail 优先 |
| `runtime_v5_11_low_memory.json` | 4 | 1 | 4 memories / 128 MiB | 16 | 小机器和低内存部署 |
| `runtime_v5_11_report_8w.json` | 8 | 2 | 8 memories / 256 MiB | 248 | 论文 8-core、最高 256 closed-loop users 的精确复现 |

前三个 profile 不预设 CPU ID，部署时应按机器 topology 显式传入 `--cpu-ids`。论文 profile 固定 CPU 0--7，只应在这八个逻辑 CPU 对应八个独立物理核时使用。

## 3. 参数分层

### 3.1 精度、延迟与 Token 预算

| 参数 | 推荐值 | 作用与代价 |
|---|---:|---|
| `query_budget.max_hops` | 2 | 图多跳深度；增加可改善长链覆盖，但扩大遍历工作量 |
| `max_visited_nodes/edges` | 96/192 | 查询硬工作预算，直接约束最坏延迟 |
| `max_frontier` | 32 | 每轮 frontier 上限 |
| `max_evidence_turns` | 32 | 最终送入证据打包的 turn 上限 |
| `max_evidence_tokens` | 5000 | 证据 Token 硬预算 |
| `graph_hop_decay` | 0.3 | 抑制远距离 graph-only 候选，避免其挤占词法命中 |
| `expansion_beam` | 2 | 每节点扩展 beam；此前测得是主要图遍历降延迟参数 |
| `hierarchy_root/child_beam` | 2/4 | 分层路由宽度；增大提高召回上界并增加 CPU |
| `obligation_aware_packing` | true | 按 QueryIR obligation/token 效率选择证据 |
| `native_seed_fusion` | true | 使用内存内只读 TurnSearchIndex，避免每 view 的 SQLite FTS 往返 |

### 3.2 缓存和内存

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| `snapshot_cache_memories` | 8 | 单 Worker 热 Memory 数量上限 |
| `metadata_cache_memories` | 8 | turn/evidence/principal 协调缓存上限，应与 snapshot 保持一致 |
| `snapshot_cache_bytes` | 256 MiB | 按离线 deep retained-size 核算的 view 上限 |
| `compiled_cache_dir` | trusted local path | 版本化 `.gmc` 只读索引目录；不能由不可信用户写入 |
| `compiled_cache_admission` | true | 频率感知准入；一次性租户只旁路服务，不污染热缓存 |

当前 mmap 共享 serialized input 页，但 pickle 还会在每个 Worker 重建 Python postings；因此它不是完全 zero-copy。论文实验中的内存应同时报告聚合 RSS 与 PSS。

### 3.3 并发、隔离和高可用

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| `workers` | 物理核数以内 | 每 Worker 一个进程、一次执行一个 CPU-bound query |
| `affinity_replicas` | 2 | Rendezvous hashing 候选数；热点可并行，代价是缓存复制 |
| `warm_replicas` | 1 | 预热只放一份，按真实负载再扩展副本 |
| `max_queued` | 32--64 | 全局 outstanding 为 `workers + max_queued` |
| `per_tenant_outstanding` | 4--8 | 防止 noisy tenant 独占队列 |
| `request_timeout_seconds` | 5 | 请求 deadline 上限 |
| `retry_broken_worker` | 1 | BrokenProcessPool 后重建 shard 并在 deadline 内重试一次 |
| `worker_cpu_ids` | 显式 topology | 一 Worker 对应一个独立物理核；不要把 SMT siblings 当作两核 |

## 4. Compiled sidecar 生命周期

服务启动时，`precompile_on_start=true` 会并行补齐缺失或过期 sidecar。每次发布包含：

1. SQLite authority 的 `(memory_id, graph_version, graph_checksum)`；
2. 临时 `.gmc` 写入、`fsync`、原子 `os.replace`；
3. 小型 `.gmc.meta.json` publish record，用于后台低成本版本轮询；
4. Worker 载入时仍完整校验 artifact schema、memory、version 和 checksum。

`sidecar_refresh_seconds>0` 时，父进程后台 maintainer 会发现新增或更新后的 Memory，异步生成新版本。生成失败不会影响 SQLite authority；查询 Worker 会安全回退到 snapshot compile，并在 `/v1/stats` 暴露失败数。

也可以离线执行：

```bash
python scripts/precompile_v5_11_read_views.py \
  --db /path/to/graphmem.sqlite \
  --output /trusted/local/compiled_views \
  --workers 4
```

第二次执行只读取小型 publish record，当前版本会被标为 `current`，不会重新反序列化整个语料索引。

## 5. 复现实验

V5.11 GraphMem runner 已从同一 runtime schema 读取检索参数：

```bash
python scripts/benchmark_v5_11_graphmem_pareto.py \
  --workload /path/to/workload.json \
  --db /path/to/graphmem.sqlite \
  --compiled-cache-dir /trusted/local/compiled_views \
  --runtime-config configs/v5/runtime_v5_11_balanced.json \
  --workers 8 --cpu-ids 0-7 \
  --clients 1,4,16,64,128,256 \
  --output /path/to/results
```

结果 manifest 会记录 runtime config 路径和 SHA-256 hash，避免“报告参数只存在于脚本”的漂移。
