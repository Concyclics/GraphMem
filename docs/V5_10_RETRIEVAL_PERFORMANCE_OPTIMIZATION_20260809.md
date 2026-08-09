# GraphMem V5.10 检索性能瓶颈与第一轮优化结果

日期：2026-08-09

## 为什么此前单 worker 比 Mem0 慢

并不是分层图必然比 dense vector search 慢，而是当前执行路径存在几项不必要的 Python 开销。

第一，服务模型是一 worker 一请求串行执行。优化前单请求 service mean 约 68.8 ms，在 8 clients 下只能提供约 14 QPS，端到端 mean 565.5 ms 中约 496.7 ms 都是队列等待。Mem0 的 query embedding 和 Qdrant 搜索大部分在外部服务/C++ 路径执行，能够释放 Python GIL，因此单 Python process 仍可重叠多个请求。

第二，同一个 immutable fact、predicate、value 和 turn text 在每次查询中被反复正则分词。原始 cProfile 中 93 次 warm query 产生约 7,619 万次函数调用，其中 `content_terms` 被调用约 77 万次；scheduler、fact lookup 和 binding 对相同字符串反复执行 regex、`casefold` 和 set 构造。

第三，候选打分存在隐藏的高阶循环。原实现对每个 candidate turn 重新扫描所有 bindings，并重新求 evidence turns 的集合并集，复杂度近似为：

\[
O(|C|\,|B|\,|E|).
\]

第四，evidence group hydration 存在 N+1 SQL。`evidence_groups(memory)` 先读 group，再对每个 group 单独查询 members；query pack 阶段又逐 group 调用 `evidence_group(id)`。这既拉高冷用户初始化，也给 warm pack 增加大量 Python/SQLite 边界切换。

第五，多用户 full workload 的 p95 还受到另一类瓶颈控制：110 个 memory、两个 affinity replicas 和每 worker 16-memory metadata cache 会造成 GraphReadView 换入换出。这个问题与 warm execution CPU 不同，必须单独解决。

## 已实现的语义保持优化

1. 对 immutable graph/query text 的 `content_terms` 和 `normalize_key` 增加 8K 有界 LRU；
2. 在 GraphReadView 中预编译 predicate term 与 owner-predicate 元数据，避免每 query 扫描无关 owner；
3. scheduler 将 query terms 移出 traversal loop，node lexical surface 按需缓存，不再全量复制第二份 node term set；
4. 将 binding provenance 一次性反向编译为 `turn -> operand_ids`，候选打分变为 O(1) lookup；
5. `evidence_groups(memory)` 改为 groups + joined members 两次批量查询；
6. snapshot metadata cache 同时保存 group members，pack 不再逐 group 查询 SQLite。

没有改变 QueryIR、图边、beam、candidate reservoir、fusion weights、turn/token budget、packing 规则或 answer 逻辑。

## Hot-path 剖析变化

相同 31 个真实问题、重复 3 次，cProfile 会放大绝对延迟，但适合比较相对 CPU 工作量：

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| 总函数调用 | 76.19 M | 11.82 M |
| profile wall time | 19.13 s | 4.53 s |
| total mean | 207.84 ms | 50.73 ms |
| graph stage mean | 69.59 ms | 7.93 ms |
| fact reservoir mean | 38.14 ms | 6.20 ms |
| evidence pack mean | 46.06 ms | 15.36 ms |

## 四用户同样本结果

四个 LoCoMo users、2,080 turns、31 个真实问题、8 clients、20 秒、top-32：

| 配置 | QPS | mean | p50 | p95 | p99 | RSS |
|---|---:|---:|---:|---:|---:|---:|
| GraphMem 1 worker，优化前 | 13.99 | 565.5 | 555.5 | 674.6 | 814.9 | 130.5 MiB |
| GraphMem 1 worker，优化后 | 41.06 | 193.9 | 184.6 | 254.7 | 421.3 | 143.9 MiB |
| GraphMem 4 workers，优化前 | 36.35 | 218.8 | 225.0 | 317.0 | 387.5 | 394.0 MiB |
| GraphMem 4 workers，优化后 | 100.65 | 79.3 | 76.3 | 127.0 | 167.5 | 425.5 MiB |
| Mem0 OSS 2.0.17，1 process | 18.22 | 437.8 | 290.7 | 1002.0 | 1116.0 | 393.7 MiB |

优化后：

- 1 worker QPS 提升 2.94 倍，mean/p95 分别下降 65.7%/62.2%，RSS 增加 10.3%；
- 4 workers QPS 提升 2.77 倍，mean/p95 分别下降 63.8%/59.9%，RSS 增加 8.0%；
- 对 Mem0 同样本，1-worker GraphMem 已达到 2.25 倍 QPS且 p95 低 74.6%；
- 相近内存级别的 4-worker GraphMem 达到 5.53 倍 QPS，p95 低 87.3%。

这里仍然是 retrieval data-plane pilot；Mem0 的 GPU embedding service RSS/显存没有计入，不能把这些数字直接宣传成全系统成本优势。

## 110-memory 全量并发结果

8 workers、16 clients、8 tenants、110 memories、60 秒：

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| QPS | 51.28 | 66.19 | +29.1% |
| mean | 309.5 ms | 239.7 ms | -22.6% |
| p50 | 213.1 ms | 87.0 ms | -59.2% |
| p95 | 872.4 ms | 898.7 ms | +3.0% |
| p99 | 1132.4 ms | 1141.0 ms | +0.8% |
| worker RSS | 4722 MiB | 5358 MiB | +13.5% |

这说明 warm CPU 路径已经明显加速，但 large-tenant p95 仍由 cold view cache churn 控制。本轮不能宣称 full-workload p95 已改善；准确的表述是“吞吐、mean、p50 改善，p95/p99 基本未解，内存有所增加”。

曾尝试在 affinity replica 队列相同时强制选择 primary，以减少重复 view；A/B 结果 QPS 下降约 6.5%、p95 上升约 4.8%、RSS 没有改善，因此已经撤回。

## 质量门

完整 200 问题、三个 QueryIR arms 重新运行：

- H11 all-hit：48.5% -> 48.5%；
- H11 recall：61.965% -> 61.965%；
- candidate all-hit：100% -> 100%；
- evidence tokens：2404.585 -> 2404.585；
- false-complete：4% -> 4%；
- visited nodes/edges 和 per-stratum 指标一致；
- 32 个相关测试通过。

因此性能收益不是通过减少候选、缩小 budget 或绕过图取得的。

## 下一轮应解决什么

最高优先级已经从 warm CPU 转到 cold view 与内存复制：

1. 把 GraphReadView、TurnSearchIndex、predicate postings、token lengths 在 build/publish 阶段预编译成可 mmap 的紧凑 artifact；
2. worker 只映射共享只读页，不再分别构造 Python dict/frozenset；
3. cache byte accounting 必须包含所有 derived indexes 和 term cache，而不只估算 nodes/edges；
4. 分离 hot query workers 与 cold compiler workers，避免一个新用户阻塞热用户队列；
5. 增加 per-memory cache hit/miss、view compile latency、eviction bytes、replica duplication telemetry；
6. 对 110-memory workload 做 3 次重复和置信区间，再决定论文中的 p95 数字。

预期下一轮目标：保持当前约 66 QPS，将 full p95 降至 600 ms 以下，并把 8-worker RSS 从 5.36 GiB 降到 3.5--4.0 GiB。

## 证据

- `../artifacts/report/v5_10/performance_optimization/summary.json`
- `../artifacts/report/v5_10/graphmem_finalopt_raw4_w1_c8_20s_20260809/summary.json`
- `../artifacts/report/v5_10/graphmem_finalopt_raw4_w4_c8_20s_20260809/summary.json`
- `../artifacts/report/v5_10/multi_tenant_w8_c16_60s_opt3_20260809/summary.json`
- `../artifacts/report/v5_10/queryir_gate_dev200_opt1_20260809/summary.json`
- `scripts/profile_v5_10_retrieval_hotpath.py`
