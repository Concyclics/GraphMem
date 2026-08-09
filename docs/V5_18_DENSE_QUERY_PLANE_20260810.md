# V5.18 批量 Dense Query Plane 与 per-memory FAISS（2026-08-10）

## 结论

V5.18 将当前 V5.17 accuracy64 检索路径改为：QueryIR dense views 批量编码、跨进程持久化 query cache、版本化 per-memory FAISS 精确索引，以及 affinity-aware 预热。实现不改变候选或证据排序语义。

在相同 16-memory、7,971-vector、64-turn/top-k、双方均复用 query vector 的保守 warm data-plane 对照中：

- 8 worker / C=16：GraphMem 153.85 QPS、p95 179.42 ms；Mem0 64.24 QPS、p95 487.29 ms。
- 8 worker / C=64：GraphMem 159.87 QPS、p95 660.97 ms；Mem0 71.34 QPS、p95 1,238.93 ms。
- 8 worker / C=256：GraphMem 133.60 QPS、p95 2,382.66 ms；Mem0 71.23 QPS、p95 3,968.92 ms。
- 8 worker / C=128 总 PSS：GraphMem 1.77 GiB；Mem0 4.67 GiB，GraphMem 低 62.1%。

这里的优势来自查询 CPU 路径与内存布局，不来自让 Mem0 单方面承担 query embedding。双方都使用预计算 query vector；因此这组结果只代表 warm retrieval data plane，不代表真实冷 query 的端到端延迟。

## 实现结构

### 1. QueryIR dense view 批处理

原实现依次调用 full query 和每个 owner-predicate view；一个问题通常产生两个 dense HTTP 请求。现在 `seed_operands` 先收集全部 dense view，再调用一次 `dense_search_many`：

\[
\{q_0,q_1,\ldots,q_{m-1}\}
\xrightarrow{\text{one embedding batch}}
Q\in\mathbb{R}^{m\times d}.
\]

旧 `dense_search(memory, query, k)` 接口仍然保留。未提供 batch callable 的历史 runner 会自动退回逐 view 路径，冻结的 ablation 语义不变。

### 2. 持久化 Query Embedding Cache

`QueryEmbeddingCache` 是独立 WAL SQLite sidecar，key 包含 embedding model、instruction revision 和完整 query text hash。它不写 authority graph DB，因此只读 graph replica 可以共享缓存。

每个 worker 还有有界进程内 LRU；同进程并发 miss 由 singleflight 合并。顺序为：

1. process LRU；
2. shared persistent WAL cache；
3. 单次 batched embedding API；
4. 原子 upsert WAL cache，并唤醒同 key waiter。

统计项包括 memory/persistent hit、miss、batch、singleflight wait、API latency/token 和 entry 数。

### 3. 版本化 per-memory Dense Sidecar

每个 memory 对应一个独立精确 inner-product 索引。向量在编译时归一化，因此：

\[
\operatorname{cos}(q,x_i)=
\frac{q^Tx_i}{\lVert q\rVert_2\lVert x_i\rVert_2}
=\hat q^T\hat x_i.
\]

默认 `backend=auto`：安装 FAISS 时选择 `faiss.IndexFlatIP`，否则使用 mmap `numpy_exact`。两者都是精确检索；当前没有引入 HNSW 近似误差。

sidecar manifest 绑定：

- memory ID；
- graph version 与 logical graph checksum；
- embedding model；
- turn content-hash aggregate；
- vector count / dimension；
- data file size 与 SHA-256；
- backend 与 turn-ID position map。

版本化 data file 先写临时文件、`fsync`、rename；stable manifest 最后发布。读者只会看到旧完整版本或新完整版本。校验失败时回退到 SQLite 精确矩阵，不返回 stale index。

16 个 memory 的实测 sidecar：

| 指标 | 结果 |
|---|---:|
| source-turn vectors | 7,971 |
| dimension | 1,024 |
| backend | 16 × `faiss_flat` |
| data size | 31.14 MiB |
| 4-process compile wall time | 0.31 s |
| second sync | 16 current / 0 rebuilt |

### 4. Proof-packer 静态特征缓存

CPU profile 显示 64-turn packer 为每个已选 turn 重复执行句子切分、词法 tokenization 和 number/time/negation/status 四组 regex；这些计算与 query 无关，约占 58 ms/query。

V5.18 将每个 immutable source text 的 sentence boundaries、content terms 和 critical flags 缓存为有界静态特征。query 时只计算 lexical overlap 和 operator-specific weights。公式、tie break、span 和 token budget 均未改变。

cProfile（32 次 warm navigation）中：

| 热点 | 修改前累计 | 修改后累计 |
|---|---:|---:|
| `pack_obligation_aware` | 2.054 s | 0.280 s |
| `salient_spans` | 1.866 s | 0.127 s |
| function calls | 9.94 M | 6.97 M |

## 等价性验证

### Dense index

- 16 个真实问题：batched SQLite 与 batched FAISS 相对 serial SQLite，retrieved order 16/16 相同、retrieved set 16/16 相同、candidate Jaccard = 1.0。
- 16 个 memory × 10 个真实 turn-vector query：FAISS 与 SQLite cosine Top-96 排序 160/160 完全一致；最大 score 误差为 \(2.38\times10^{-7}\)。

### Packer 优化

修改前后共享的 96 个 benchmark rows 中：

- retrieved order：96/96 相同；
- candidate order：96/96 相同。

全量工程测试：402 passed。

## GraphMem / Mem0 同条件矩阵

条件：16 memories、7,971 raw-turn vectors、64 results、closed-loop users、1/4/8 workers、C∈{1,4,16,64,128,256}、2 s measurement + 1 s warmup、1 repetition、5 s deadline。双方 query vector 均由同一 workload cache 提供；Mem0 使用 OSS 2.0.17 + local Qdrant，关闭 reranker/BM25。GraphMem 在每个 rendezvous affinity replica 上预热全部 workload memories；没有 all-to-all worker 复制。

下表列出主要并发点（QPS / p95 ms）：

| Workers | Clients | GraphMem | Mem0 | QPS ratio | p95 ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 35.53 / 477.32 | 14.34 / 1,116.95 | 2.48× | 0.43× |
| 1 | 64 | 31.45 / 2,194.90 | 14.32 / 4,449.10 | 2.20× | 0.49× |
| 1 | 256 | 32.69 / 5,007.19 | 10.26 / 4,788.88 | 3.19× | 1.05× |
| 4 | 16 | 98.23 / 207.44 | 50.57 / 429.02 | 1.94× | 0.48× |
| 4 | 64 | 103.08 / 877.59 | 51.93 / 1,304.33 | 1.99× | 0.67× |
| 4 | 256 | 96.75 / 3,142.51 | 48.54 / 4,921.65 | 1.99× | 0.64× |
| 8 | 16 | 153.85 / 179.42 | 64.24 / 487.29 | 2.39× | 0.37× |
| 8 | 64 | 159.87 / 660.97 | 71.34 / 1,238.93 | 2.24× | 0.53× |
| 8 | 256 | 133.60 / 2,382.66 | 71.23 / 3,968.92 | 1.88× | 0.60× |

PSS @ C=128：

| Workers | GraphMem | Mem0 | GraphMem change |
|---:|---:|---:|---:|
| 1 | 0.69 GiB | 0.61 GiB | +13.3% |
| 4 | 1.48 GiB | 2.35 GiB | -37.0% |
| 8 | 1.77 GiB | 4.67 GiB | -62.1% |

1-worker / C=256 是本轮唯一 p95 未占优的高并发点：GraphMem QPS 更高，但 5 s deadline 附近的队尾使 p95 略差。报告应将其画在 Pareto 图上，而不是只报告优势格子。

## 预热与高可用语义

新增 benchmark 选项 `--warm-all-affinity` 使用 `warm_affinity`：一个 memory 只加载到会实际服务它的 rendezvous replica。两轮 readiness 的总墙钟为 7.44 s（W1）、6.46 s（W4）、3.44 s（W8）；第二轮验证 cache hit 和 graph identity 一致。

在线服务启动流程现在可以同步两类 sidecar：

1. compiled graph/turn/provenance `.gmc`；
2. per-memory dense FAISS/NumPy index。

后台 maintainer 轮询 graph/embedding publication；stale dense sidecar 在新版本完成前回退 authority SQLite。worker crash 重启后从共享 sidecar/query cache 恢复，不需要重新编码全部 vectors。

## 配置与命令

`configs/v5/runtime_v5_17_accuracy64.json` 已启用：

- `dense_search_enabled=true`；
- shared query cache；
- per-memory dense sidecar；
- `dense_backend=auto`；
- 256 MiB / 32-memory dense LRU。

关键脚本：

- `scripts/precompile_dense_indexes.py`：编译/增量校验 dense sidecar；
- `scripts/benchmark_v5_18_dense_index.py`：serial/batch/FAISS 等价性与顺序延迟；
- `scripts/prepare_v5_18_dense_workload.py`：冻结 16-memory workload 和 warm query cache；
- `scripts/benchmark_v5_11_graphmem_pareto.py --warm-all-affinity`：GraphMem 矩阵；
- `scripts/prepare_v5_11_mem0_raw_index.py --embedding-db ...`：同向量 Mem0 index；
- `scripts/benchmark_v5_11_mem0_pareto.py --query-vector-cache ...`：Mem0 warm data plane；
- `scripts/render_v5_18_dense_pareto.py`：PNG/PDF/SVG 和 machine-readable ratios。

## 尚未完成的发布级验证

当前 8001 embedding endpoint 未监听，因此没有重跑真实冷 query embedding。发布报告前还需：

1. 8001 恢复后，清空 query cache，测 serial multi-call、batched one-call 和 Mem0 one-call 的 cold latency/token；
2. 使用真实 query vectors 重跑 16-question retrieval/answer，确认 12/16 strict answer 不变；
3. 将每个 cell 扩展到至少 10 s × 3 repetitions，报告 bootstrap CI；
4. 使用更大的 110-memory/full benchmark workload 复核 cache capacity、Zipf 热度和 cold-tenant admission；
5. 分离 service time 与 queue time，并在报告中同时展示 cold-start 和 affinity-warm Pareto。

当前数据足以证明实现的精确性和 warm query-plane 优势，但不应替代以上 release gate。
