# V5.11 Compiled View 与多租户缓存准入优化

## 结论

本轮优化不改变 GraphMem 的检索算法，而是把冷请求前的只读索引构造从 query worker 中移出，并为小内存缓存增加频率感知准入。对 110 个活跃 memory、8 workers、16 clients、8 tenants、Zipf α=1.1 的同一 30 秒负载：

- 推荐吞吐档（2 affinity replicas、8 memories/worker）：72.08 QPS，端到端 p95 603.8 ms，总 worker RSS 3.46 GiB。
- 严格 tail 档（3 affinity replicas、8 memories/worker）：62.80 QPS，端到端 p95 598.0 ms，总 worker RSS 3.51 GiB。
- 相对无 sidecar 的 V5.10 对照，推荐档 QPS +16.4%，p95 -32.6%，RSS -34.6%，burst 服务 p95 -41.5%。
- 200 题逐字段等价门比较 32 个非 timing/telemetry 字段，0 mismatch；检索 all-hit 均为 0.495。

## 瓶颈归因

一次冷请求在 QueryIR 编译前依次构造 `GraphReadView`、raw-turn projection、`TurnSearchIndex`、evidence 双向索引和 principal registry。最大样本的单进程测量为：图视图约 193 ms，turn lexical index 约 65 ms，其余 metadata 约 11 ms；完整 sidecar mmap 反序列化约 121 ms。

旧缓存使用节点/边行数的粗略估算。510 个 memory 的离线深度核算显示：

| 指标 | 结果 |
|---|---:|
| 编译 memory | 510 |
| 失败 | 0 |
| 4 compiler workers 墙钟 | 85.2 s |
| sidecar 文件总量 | 3,185,774,651 B |
| Python retained footprint | 14,463,170,803 B |
| retained / serialized | 4.54× |

因此原 512 MiB “估算限额”没有反映 Python dict、frozenset、tuple、Counter 与 adjacency 对象的放大。仅把缓存从 16 降到 8 又会造成 LRU pollution：一次性冷租户淘汰热 memory，换入因子达到 4.65×，QPS 下降到 55.00。

## 实现

### 1. Versioned compiled-memory sidecar

离线 cold compiler 为每个 immutable graph snapshot 物化一个 `.gmc`：

```text
SQLite authority
  └─ (memory_id, graph_version, graph_checksum)
      └─ cold compiler
          └─ atomic .gmc publish
              ├─ GraphReadView / typed adjacency / hierarchy postings
              ├─ SourceTurn projection / TurnSearchIndex
              ├─ evidence group 双向索引
              └─ PrincipalRegistry
```

worker 只接受 `(compiled schema, memory_id, graph_version, graph_checksum)` 全部匹配的 sidecar。文件以临时文件 + `fsync` + `os.replace` 原子发布；读取时使用只读 mmap，避免每个 worker 再分配一份 serialized input buffer，并利用 OS page cache。SQLite 始终是 authority；sidecar 缺失、损坏或过期时安全回退到 snapshot compile。

注意：mmap 共享的是 sidecar 输入页；pickle 重建后的 Python postings 仍是每个 worker 私有对象，本实现不是完整的 zero-copy index。sidecar 目录必须是可信本地目录，不能由不可信用户写入。

### 2. 真实 retained-byte accounting

cold compiler 对对象图做 identity-deduplicated deep traversal，覆盖 strings、postings、adjacency、counters 和 slots dataclasses，并把 `view_retained_bytes` 与 `total_retained_bytes` 写进 artifact。worker 用该数值执行 byte-bounded eviction，不在查询热路径重复深度扫描。

同时消除了两类常驻重复：terminal/routing provenance map 复用全局 provenance frozenset；turn bundle 与 `TurnSearchIndex` 复用同一个 `turn_by_id` 哈希表。benchmark 额外记录 RSS、PSS、Private Dirty、Shared Clean 和每 worker 的实际 accounted bytes。

### 3. TinyLFU-style admission 与一次性旁路

每个 worker 维护带周期衰减的 memory frequency sketch。当缓存达到 count 或 byte 上限：

\[
\operatorname{admit}(m)=
\begin{cases}
1, & f(m)>f(v_{LRU}) \\
0, & f(m)\le f(v_{LRU})
\end{cases}
\]

不满足准入的 artifact 仅服务当前请求，不进入 view、turn、evidence、principal 四套协调缓存；请求结束即释放。显式 `warm_memory`/`warm_affinity` 可强制准入。频率每 4096 次 observation 衰减一半，避免历史热点永久占据缓存。

### 4. serving telemetry

新增 per-worker：view hits/misses/builds/build waits/evictions/invalidations/build ms，sidecar hits/misses/invalid/errors/load ms/mapped bytes，admissions/bypasses/hydrations，以及 metadata entry 数。benchmark 同时输出 compiled replica load factor，区分“高并发需要的热副本”和“LRU 抖动造成的重复换入”。

## A/B 结果

共同配置：110 memories，8 workers，16 clients，8 tenants，Zipf α=1.1，5 s deadline，global outstanding=40，30 s measured interval。burst 为 128 个请求，其中 40 个被准入、88 个按设计立即拒绝。

| 配置 | QPS | p95 ms | p99 ms | RSS MiB | PSS MiB | burst service p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| V5.10，无 sidecar，cache16，R2 | 61.90 | 896.2 | 1209.6 | 5297.0 | 5136.7 | 449.0 |
| sidecar，cache16，R2 | **87.14** | 650.2 | **823.1** | 5407.6 | 5247.0 | 274.5 |
| sidecar，cache8，无 admission，R2 | 55.00 | 791.8 | 1068.6 | 3499.8 | 3339.1 | 291.9 |
| **sidecar + admission，cache8，R2** | 72.08 | 603.8 | 847.8 | **3461.9** | **3301.2** | **262.7** |
| sidecar + admission，cache8，R3 | 62.80 | **598.0** | 863.4 | 3514.3 | 3353.7 | 266.0 |

sidecar/cache16 证明冷编译确实是吞吐瓶颈，但不解决内存。cache8/no-admission 证明仅缩缓存会引发抖动。最终 R2 是吞吐、tail、内存三者的 Pareto 推荐点；R3 只适合 p95 必须低于 600 ms 的部署。

## 正确性与可用性

- 43 个 serving/retrieval/index targeted tests 通过。
- 200 题比较 `NavigationResult` 的 packed turns、candidate scores、proof、certificate、graph path、algebra 等 32 个字段，0 mismatch。
- sidecar 路径发生 0 SQLite graph rebuild；过期 checksum 会被拒绝。
- worker crash/restart、bounded admission、affinity snapshot consistency 的既有测试继续通过。

## 运行方式

正式查询平面现在从统一 runtime profile 读取这些参数，推荐直接启动：

```bash
python scripts/serve_v5_11.py \
  --db /path/to/report_graph.sqlite \
  --runtime-config configs/v5/runtime_v5_11_balanced.json \
  --compiled-cache-dir /trusted/local/compiled_views \
  --cpu-ids 0-7
```

该入口会在启动时补齐 sidecar，并可轮询 SQLite graph identity，在图发布后异步生成新版本；完整参数见 `docs/V5_11_RUNTIME_DEPLOYMENT.md`。

先由隔离的 cold compiler 生成 sidecar：

```bash
python scripts/precompile_v5_11_read_views.py \
  --db /path/to/report_graph.sqlite \
  --output /trusted/local/compiled_views \
  --workers 4 --force
```

吞吐优先推荐参数：

```python
navigator_options = {
    "compiled_cache_dir": "/trusted/local/compiled_views",
    "compiled_cache_admission": True,
    "snapshot_cache_memories": 8,
    "metadata_cache_memories": 8,
    "snapshot_cache_bytes": 256 * 1024 * 1024,
}
pool = ProcessShardedNavigator(
    db_path, workers=8, affinity_replicas=2,
    navigator_options=navigator_options,
)
```

严格 p95 档只把 `affinity_replicas` 调为 3。显式预热应使用 `warm_affinity(..., replicas=1)`，不要再把每个 memory all-to-all 复制到全部 workers。

## 后续工作

当前剩余成本是 bypass 请求仍需反序列化完整 Python object graph，且 shared page 只覆盖输入 sidecar。下一阶段若继续压 p99/RSS，应把高频 postings、node ids 与 adjacency 迁移为 offset-based compact arrays（Arrow/自定义二进制格式），让 worker 在 mmap 上直接二分/扫描；同时由发布流水线在 graph commit 后异步编译新版本，使更新后的首个请求也稳定命中 sidecar。
