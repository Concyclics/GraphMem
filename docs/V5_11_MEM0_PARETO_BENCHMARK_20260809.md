# GraphMem V5.11 vs. Mem0 OSS 并发 Pareto 基准

日期：2026-08-09
结论属性：同机、同数据、同查询分布的 retrieval data-plane 系统筛查；不是准确率对比，也不是 Mem0 Server/Cloud 的性能结论。

## 1. 结论

本轮完成了 `2 systems × 3 worker counts × 6 concurrency levels = 36` 个配置点。36 个点全部满足：

- 0 failed；
- 0 timeout；
- 0 admission rejection；
- 0 wrong memory/user partition；
- accepted request 全部完成后才停止计时，没有通过丢弃慢请求虚增 QPS。

在 18 个同 worker、同并发配对中，GraphMem 均同时满足更高 QPS 和更低 p95，即严格支配 `18/18` 个 Mem0 点。8 workers、128 并发时：

| 系统 | QPS | p50 | p95 | p99 | 平均 service | 平均 queue | Worker RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| GraphMem V5.11 | 80.09 | 1.335 s | 2.836 s | 3.414 s | 82.5 ms | 1.288 s | 3.37 GiB |
| Mem0 OSS 2.0.17 | 7.66 | 11.813 s | 17.536 s | 18.224 s | 913.8 ms | 10.200 s | 28.23 GiB |

对应 GraphMem 的 QPS 为 `10.45×`，p95 降低 `83.8%`。256 并发时，GraphMem/Mem0 分别为 `73.17/7.21 QPS`，p95 为 `5.423/36.776 s`，p95 降低 `85.3%`。

## 2. 冻结工作负载

| 项目 | 设置 |
|---|---|
| Memory 数 | 110 |
| 原始 turn 数 | 55,323 |
| Query 数 | 200（LongMemEval 100 + LoCoMo 100） |
| Workload SHA-256 | `9aa52251d72b92b33fc0a72f1ad24d41c939e4e3ce429267894dd60b8b2272d5` |
| Top-k | 32 |
| 用户分布 | Zipf `α=1.1` |
| 并发 | 1/4/16/64/128/256 个 closed-loop 逻辑用户 |
| Worker/core | 1/4/8；每 worker 绑定一个独立 CPU |
| Outstanding | 每逻辑用户最多 1 个 |
| Affinity | rendezvous hashing，`min(2, workers)` replicas |
| 测量窗口 | 每点 10 秒发起窗口；随后完整 drain |
| Readiness | 逐 worker 两轮探针 + 每并发档短 warmup |

逻辑用户数表示同时存在的独立请求流；200-query 数据中包含 110 个不同 Memory，当并发超过 110 时，不同逻辑租户可以访问同一只读 Memory。这是多租户 serving 并发，不是合成 256 份不同内容索引。

## 3. 系统配置

### GraphMem

- V5.11 QueryIR + hierarchical routing + relation algebra；
- frozen SQLite graph authority；
- compiled immutable read-view sidecar；
- 每 worker 最多缓存 8 个 Memory、256 MiB snapshot budget；
- 在线 Query 不调用 embedding 服务；
- 每 process shard 同时执行一个请求。

### Mem0

- Mem0 OSS 2.0.17；
- `infer=False`，保存与 GraphMem 完全相同的原始 turn；
- `Memory.search(..., filters={"user_id": memory_id}, top_k=32, rerank=False)`；
- 在线 Query embedding：本地 `Qwen/Qwen3-Embedding-0.6B`；
- 每 worker 一份独立 embedded Qdrant dense index，以规避 local storage 独占锁；
- 当前环境未安装可选 `fastembed`/spaCy extras，因此 BM25 和 entity boost 未启用。

离线装载为了缩短准备时间，使用 Mem0 embedder 的 `embed_batch` 与同一 vector-store insert 契约批量写入；在线测量仍完整调用公开 `Memory.search`。索引验证结果为 55,323 points、110 partitions，所有 partition 的 point 数与源 turn 数逐一相等。

## 4. 完整结果

### 1 worker

| 并发 | GraphMem QPS / p95 | Mem0 QPS / p95 | QPS 加速 | p95 降幅 |
|---:|---:|---:|---:|---:|
| 1 | 5.72 / 427 ms | 1.84 / 884 ms | 3.10× | 51.8% |
| 4 | 5.05 / 1,215 ms | 1.69 / 2,599 ms | 2.98× | 53.2% |
| 16 | 6.41 / 3,345 ms | 1.78 / 9,169 ms | 3.60× | 63.5% |
| 64 | 5.98 / 10,770 ms | 1.75 / 37,114 ms | 3.42× | 71.0% |
| 128 | 5.81 / 22,817 ms | 1.75 / 72,319 ms | 3.31× | 68.4% |
| 256 | 5.90 / 44,071 ms | 1.76 / 145,676 ms | 3.36× | 69.7% |

### 4 workers

| 并发 | GraphMem QPS / p95 | Mem0 QPS / p95 | QPS 加速 | p95 降幅 |
|---:|---:|---:|---:|---:|
| 1 | 7.08 / 357 ms | 1.82 / 901 ms | 3.89× | 60.3% |
| 4 | 19.41 / 509 ms | 4.23 / 1,630 ms | 4.58× | 68.8% |
| 16 | 23.60 / 1,274 ms | 4.98 / 4,174 ms | 4.74× | 69.5% |
| 64 | 26.78 / 4,512 ms | 4.67 / 15,832 ms | 5.73× | 71.5% |
| 128 | 24.26 / 7,292 ms | 4.66 / 27,887 ms | 5.21× | 73.9% |
| 256 | 22.30 / 15,729 ms | 5.02 / 50,112 ms | 4.44× | 68.6% |

### 8 workers

| 并发 | GraphMem QPS / p95 | Mem0 QPS / p95 | QPS 加速 | p95 降幅 |
|---:|---:|---:|---:|---:|
| 1 | 9.37 / 298 ms | 1.68 / 924 ms | 5.59× | 67.8% |
| 4 | 36.16 / 388 ms | 4.53 / 1,195 ms | 7.98× | 67.5% |
| 16 | 67.74 / 626 ms | 6.97 / 3,599 ms | 9.71× | 82.6% |
| 64 | 78.76 / 1,712 ms | 7.51 / 10,044 ms | 10.49× | 83.0% |
| 128 | 80.09 / 2,836 ms | 7.66 / 17,536 ms | 10.45× | 83.8% |
| 256 | 73.17 / 5,423 ms | 7.21 / 36,776 ms | 10.14× | 85.3% |

## 5. 方法论分析

### 5.1 Pareto 边界

在每个 worker 数内联合计算“最大化 QPS、最小化 p95”的非支配集合。Mem0 的所有点均被同 worker 下至少一个 GraphMem 点支配；GraphMem 保留 11 个联合 frontier 点。特别地，GraphMem 的 1-client 点已经同时优于同 worker 下 Mem0 的峰值吞吐和最低尾延迟，因此优势不依赖选择高并发峰值。

### 5.2 服务时间而非纯排队差异

8 workers 下，GraphMem 平均 worker service time 为 82--98 ms，Mem0 为 596--915 ms。排队时间在两边都会随并发增长，但由 Little's Law，较长的服务时间会把相同 concurrency 放大为更长 queue：

\[
L \approx \lambda W,
\qquad
W = W_{service} + W_{queue}.
\]

GraphMem 的 QueryIR/compiled-view 路径避免了在线 query embedding，并把工作集限制在命中的 Memory read view；Mem0 需要在线 embedding，再在每个 worker 的完整 dense index 上执行带 `user_id` filter 的检索。8-worker Mem0 的 service time 比单 worker 更高，说明共享 embedding 服务和多份 embedded index 的内存/缓存竞争限制了横向扩展。

### 5.3 内存代价

256 并发时，GraphMem 的 1/4/8 workers 总 RSS 分别为 0.45/1.75/3.39 GiB；Mem0 为 3.54/14.15/28.23 GiB。这里严格按 `1 GiB = 1024 MiB` 换算。Mem0 local Qdrant 需要为每个进程复制并加载完整 55K-point index，而 GraphMem 每 worker 只缓存受预算约束的热门 read views。该结果支持“有界 working set”系统设计，但只适用于本轮 embedded-process 架构；共享 Qdrant Server 的内存口径会不同。

## 6. 边界与下一步

1. Qdrant local mode 对超过 20K points 的 collection 明确给出“不推荐”警告。本结果不能直接外推到独立 Qdrant Server、量化/分片索引或 Mem0 Cloud。
2. 当前是 retrieval data-plane 对比。Mem0 使用 raw-turn `infer=False`，没有比较两套系统的抽取策略、召回准确率或最终回答准确率。
3. 当前每点为一次 10 秒 steady-state 系统筛查。报告已明确标注；论文定稿前需在独占硬件上随机化运行顺序、至少重复三次并报告 bootstrap 置信区间。
4. 本机 embedding heartbeat 与 benchmark 共享服务，属于当前部署的真实争用，但 camera-ready 版本应单独报告独占 embedding 与共享 embedding 两种场景。
5. 下一轮应补充 Mem0 + Qdrant Server，并分别测 `dense-only` 与可选 hybrid extras，检验优势中有多少来自算法路径、有多少来自 embedded backend。

## 7. 产物

- 冻结 workload：`artifacts/report/v5_11/mem0_pareto_20260809/workload.json`
- 完整聚合：`artifacts/report/v5_11/mem0_pareto_20260809/aggregate.json`
- CSV：`artifacts/report/v5_11/mem0_pareto_20260809/results.csv`
- 报告图：`GraphMem_report/figures/eval_mem0_{pareto,scaling,latency_decomposition}.{pdf,png,svg}`
- 报告表：`GraphMem_report/generated/v5_11_mem0_pareto_table.tex`
- GraphMem runner：`scripts/benchmark_v5_11_graphmem_pareto.py`
- Mem0 runner：`scripts/benchmark_v5_11_mem0_pareto.py`
- Mem0 索引准备：`scripts/prepare_v5_11_mem0_raw_index.py`
- 汇总制图：`scripts/render_v5_11_mem0_pareto.py`
