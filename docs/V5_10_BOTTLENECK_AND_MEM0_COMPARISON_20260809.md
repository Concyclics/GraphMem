# GraphMem V5.10 准确率、延迟与多用户并发瓶颈分析

日期：2026-08-09
结论属性：冻结日志的只读诊断 + 新增四用户 Mem0 OSS 对比样本

## 1. 执行结论

当前准确率的首要瓶颈不是候选集合容量不足，而是完整证据从原文进入最终 prompt 的链路断裂：

1. 原文到事实索引仍有明显信息损失，且置信度没有校准；
2. 跨 session 关系图很稀疏，typed relation 尚未稳定成为实际路由主干；
3. 候选阶段几乎已经包含 gold turn，但 evidence pack 经常丢掉多跳问题所需的一个或多个操作数；
4. 证据齐全后仍有一小部分 answer/rendering 错误。

当前延迟有两个不同瓶颈：

- 冷用户首次访问：`GraphReadView + turn bundle + TurnSearchIndex` 的构造没有被单独计时，是 LongMemEval p95 接近 2 秒的主因；
- 热用户稳态：`evidence_pack` 是最大的单个可见阶段，但多用户饱和以后，端到端延迟主要来自 admission 后的排队，而不是图遍历。

多用户容量曲线表明 8 workers / 16 clients 是当前较稳妥的工作点：51.28 QPS、p95 872 ms。继续将 clients 提到 32，仅增加 12.4% QPS，却使 p95 增长 46.6% 至 1.279 秒，说明服务已经进入排队饱和区。

新增的四用户 Mem0 样本显示：

| 系统配置 | QPS | mean | p95 | p99 | RSS |
|---|---:|---:|---:|---:|---:|
| Mem0 OSS 2.0.17，1 process | 18.22 | 437.8 ms | 1002.0 ms | 1116.0 ms | 393.7 MiB |
| GraphMem V5.10，1 worker | 13.99 | 565.5 ms | 674.6 ms | 814.9 ms | 130.5 MiB |
| GraphMem V5.10，4 workers | 36.35 | 218.8 ms | 317.0 ms | 387.5 ms | 394.0 MiB |

单 worker 下 GraphMem 吞吐低于 Mem0 约 23.2%，但 p95 低 32.7%、RSS 低 66.9%。在 RSS 基本相同的 4-worker 配置下，GraphMem QPS 为 Mem0 的 2.00 倍，mean/p95/p99 分别低 50.0%/68.4%/65.3%。该结果目前只能称为“系统对比样本”，不能当成完整 benchmark 结论。

## 2. 准确率瓶颈：逐层误差链

定义问题所需 gold turn 集合为 \(G_q\)，索引可表达的 turn 为 \(I_q\)，图路由可达集合为 \(R_q\)，候选集合为 \(C_q\)，最终证据包为 \(P_q\)。完整证据链要求：

\[
G_q \subseteq I_q,\qquad
G_q \subseteq R_q,\qquad
G_q \subseteq C_q,\qquad
G_q \subseteq P_q.
\]

当前系统的主要问题发生在第一、第二和第四个包含关系，而不是第三个。

### 2.1 原文到索引：覆盖率与语义保真度不足

- 全库 252,489 个 source turns 中，只有 120,287 个 turn 挂有 canonical fact，turn-level coverage 为 47.64%。
- 对有 gold annotation 的问题，gold-turn fact recall 为 LongMemEval 78.80%、LoCoMo 82.88%；只有 67% 和 77.23% 的问题满足“所有 gold turn 均有 fact”。
- 216,070 个 canonical facts 的 confidence 全部恰好为 0.5，意味着置信度字段没有区分高质量事实、歧义事实、否定/模态事实和可能幻觉的事实。
- V5.10 atomic extractor 的开发门已经改善原子性，但完整 benchmark snapshot 没有重建 extraction，因此当前 full result 尚未验证新 extractor 的真实收益。

这会造成两类错误：一是 gold turn 根本没有可用于建边和路由的原子事实；二是复合句、否定、模态、状态变化被压成不稳定表示，后续即使有 turn fallback，也无法稳定建立正确的 typed relation。

下一步不能只提高“抽取数量”。应同时测：fact recall、fact precision、negation preservation、modality preservation、state-change preservation、raw-turn fallback recall，以及 calibrated confidence 的 ECE/Brier score。建议先让 gold-turn fact recall 达到 95% 以上，同时确保原子事实精度不下降。

### 2.2 索引结构：分层图存在，但跨 session 关系表达不足

- `coarse_related` 共 653,424 条，但跨 session 仅 1,407 条，占 0.215%，只覆盖 510 个 memory 中的 72 个。
- 135,372 个 collection manifests 中，82.57% 只有一个 member，说明大量“粗化节点”没有形成真正的聚合层级。
- full snapshot 的 embeddings 表为 0；recoarsen 只有 110/510 个 memory 使用 Qwen session vector，另外 400 个使用 deterministic fallback。
- LongMemEval 的 graph-all-reachable 只有 36%，LoCoMo 为 64.91%。其中最困难的 LoCoMo Cat1 只有 30.14%。
- 开发门中 typed edge 的 precision 样本不错，但实际 typed edge walk 曾为 0；这表明“关系代数/typed graph 已实现”和“检索真的依赖它”是两件不同的事。

更关键的信号是：candidate-all-hit 已达到 LongMemEval 99%、LoCoMo 99.80%。这说明大的 fallback reservoir 最终能够兜底候选，但图本身没有承担足够的召回责任。当前准确率部分依赖高成本候选兜底，削弱了“分层图同时省 token、保多跳”的论证。

改进重点应是：统一 510 个 memory 的 embedding contract；建立实体、事件、时间、状态变化四类跨 session bridge；在 executor 中把 relation type 作为操作符约束，而非仅作为打分特征；新增 `typed_edge_walk_rate`、`cross_session_gold_path_recall`、`fallback_reservoir_dependency` 三个门。

### 2.3 候选到证据包：当前最大的端到端准确率瓶颈

| Benchmark | candidate all-hit | packed all-hit | 错题数 | 因 pack 缺 gold 的错题 |
|---|---:|---:|---:|---:|
| LongMemEval annotated 100 | 99.0% | 64.0% | 48 | 28 |
| LoCoMo annotated 1,533 | 99.80% | 60.53% | 252 | 201 |

LoCoMo 的 252 个错题中，201 个缺少 packed gold，占 79.8%。当 packed all-hit 为真时准确率为 94.50%，为假时为 66.78%，差 27.73 pp，bootstrap 95% CI 为 [23.70, 31.74] pp。这个关联不能直接解释为因果收益，但足以说明 pack 是最高优先级的可控环节。

Cat1 尤其明显：session all-hit 57.80%，graph reachable 30.14%，packed all-hit 18.79%，准确率 79.43%。这类问题要求多个 session 的证据共同存在，当前 relevance 排序和固定 turn/span budget 容易把一个高分局部片段重复装入，却丢掉另一个必要操作数。

需要把 pack 从“按 turn 打分装箱”升级为“按 QueryIR obligation 做集合覆盖”：

\[
\max_{P\subseteq C}\;
\sum_{o\in O_q} w_o\min\left(1,\sum_{t\in P}a_{ot}\right)
+\lambda D(P)-\mu R(P),
\quad
\text{s.t. } \sum_{t\in P}\tau_t\le B_q.
\]

其中 \(a_{ot}\) 表示 turn \(t\) 是否满足 obligation \(o\)，\(D(P)\) 奖励跨 session/跨 relation 的多样性，\(R(P)\) 惩罚近重复，\(B_q\) 是动态 token budget。每个 obligation、session 和 temporal operand 应有最低配额，只有覆盖证书完成后才允许把剩余 budget 给局部上下文。

### 2.4 证据齐全后的 answer 错误

- LongMemEval annotated 错题中有 20 个在 gold 已全部 packed 后仍答错。
- LoCoMo 有 51 个同类错误。

这些错误需要独立划分为：答案渲染遗漏、实体指代错误、时间归一化错误、比较/计数执行错误、judge 表述不匹配。不要再通过扩大检索预算解决。对 closed-form 的 temporal、comparison、count、set-union 问题，应优先使用 deterministic executor；自然语言生成只负责最后 verbalization。

## 3. 单请求延迟瓶颈

端到端检索延迟可分解为：

\[
T_{e2e}=T_{view/init}+T_{compile}+T_{seed}+T_{route}
+T_{graph}+T_{fact}+T_{pack}+T_{queue}.
\]

### 3.1 LongMemEval：冷用户初始化主导尾延迟

LongMemEval 有 500 个问题和 500 个不同 memory。默认 metadata cache 只保留 16 个 memory，因此几乎每个请求都是冷访问。

- total mean/p95/p99：615.93 / 1968.17 / 2144.74 ms；
- 已计时 evidence pack mean/p95：219.38 / 265.81 ms；
- 未计时 residual mean/p95/p99：348.70 / 1679.68 / 1843.22 ms；
- residual 与 total 的相关系数为 0.993。

`navigator` 在 total timer 之后先执行 `runtime.view(memory_id)`、`_turn_bundle` 和 `_turn_search_index`，但各 stage timer 在这些操作之后才开始。因此 residual 主要是 immutable GraphReadView 编译、全量 turn 装载、TurnSearchIndex/token metadata 构造，以及少量未单列的 binding/algebra。LongMemEval 的 p95 不是 HNSW/图遍历慢，而是冷用户 Python 对象编译慢。

### 3.2 LoCoMo：热缓存稳态由 pack、graph、seed/fact 共同构成

LoCoMo 只有 10 个 memory，可完全进入 16-memory cache：

- total mean/p95/p99：99.48 / 130.32 / 200.66 ms；
- pack mean/p95：34.87 / 56.34 ms；
- graph read mean/p95：22.22 / 26.70 ms；
- seed mean/p95：15.77 / 25.96 ms；
- fact reservoir mean/p95：14.85 / 24.13 ms；
- residual mean/p95：9.32 / 20.55 ms。

因此延迟优化必须分两个 SLO：cold-user first retrieval 与 warm steady-state。把两者混成一个平均值会掩盖真正的系统瓶颈。

### 3.3 优化顺序

1. 为 `runtime.view`、turn load、TurnSearchIndex、binding/algebra 增加独立 telemetry，消灭 residual 黑盒；
2. 把 GraphReadView/postings/token lengths 预编译成紧凑二进制快照，采用 mmap/Arrow/array，而不是每个 worker 重建 Python dict/object；
3. 按 tenant affinity 预热，基于活跃用户数调整 cache；冷用户走独立 compiler pool，避免拖慢热用户；
4. 将 pack 的 obligation cover、去重、token accounting 下沉到 native/vectorized 路径；
5. 共享只读 mmap snapshot，避免每个 worker 复制同一份图和索引。

## 4. 多用户并发瓶颈

现有 60 秒闭环并发结果：

| workers / clients / tenants | QPS | mean | p95 | queue mean | RSS | errors |
|---|---:|---:|---:|---:|---:|---:|
| 4 / 16 / 8 | 20.14 | 786.8 ms | 1432.9 ms | 595.3 ms | 2433 MiB | 0 |
| 8 / 16 / 8 | 51.28 | 309.5 ms | 872.4 ms | 187.7 ms | 4722 MiB | 0 |
| 8 / 32 / 16 | 57.64 | 551.9 ms | 1278.7 ms | 433.0 ms | 4745 MiB | 0 |

由 `service mean ≈ e2e mean - queue mean` 估算，三档服务时间分别为 191.4、121.7、118.8 ms。8 workers 的理论粗略上限约为 \(8/0.119\approx67\) QPS；32 clients 已接近这个上限，平均延迟的 78.5% 来自排队。增加 clients 已不能线性增加吞吐。

当前并发瓶颈是：

- 单 worker 内 pack/binding/turn scoring 仍主要受 CPU 和 Python object traversal 限制；
- process scaling 能增加吞吐，但 full workload 下每 worker 约复制 590--610 MiB resident state；
- 小样本 hash primary shard 可能不均衡，affinity replicas 能缓解，但也会复制 hot view；
- embedding/冷视图构造如果与热查询共用 worker，会造成 head-of-line blocking；
- admission/backpressure 能保证 0 error 和 0 wrong-memory，但超过容量时只能排队或拒绝，并不能创造容量。

推荐短期部署点是 8 workers / 16 clients，设置 p95 < 1 s 的 admission SLO。长期应改为 shared read-only index + cache-aware dispatch + per-tenant fair queue + deadline-aware shedding，并以 `QPS/GiB`、p95/p99、fairness Jain index、cold/warm hit rate 共同评价，而非只报峰值 QPS。

## 5. 写入与高可用延迟

当前 incremental/HA gate 中，raw durable p95 为 9.97 ms，route publish p95 为 17.94 ms；但 fact/relation 数值只是事务提交，不包含远程 extraction 和 embedding。因此它们不是完整 memory update latency。

更明显的 HA 瓶颈是 snapshot 粒度：约 706.7 MB 的 snapshot copy 需要 1.87--2.35 秒，promotion 需要 1.244 秒。也就是说，前台 delta commit 很快，但副本复制仍是整快照级，写放大和恢复延迟都偏高。

下一步应补完整的 \(T_{update}\)：

\[
T_{update}=T_{durable}+T_{extract}+T_{embed}+T_{delta-edge}
+T_{publish}+T_{replicate}+T_{visible}.
\]

并测 read-your-writes、stale-read window、update QPS、读写混合 95:5/80:20、replica lag 和 crash RTO。实现上改为 WAL/delta snapshot replication、后台 compaction、epoch/RCU pointer publish，而不是每次复制大文件。

## 6. Mem0 对比样本的方法与边界

### 6.1 公平性设置

- Mem0 OSS 2.0.17，公开 `Memory.add` / `Memory.search` API；
- 四个相同 LoCoMo memory，共 2,080 条完整 source turns、31 条真实问题；
- 8 closed-loop clients、Zipf \(\alpha=1.1\)、20 秒、每次最多返回 32 turns；
- 每个 conversation 映射为一个 `user_id`，所有搜索都用 user filter，0 cross-user contamination；
- Mem0 使用 `infer=False` 保存原文，Qdrant embedded dense search，相同本地 Qwen3-Embedding-0.6B；未开启 reranker、BM25 和 graph；
- GraphMem 使用当前 H11/native-seed/obligation-pack 路径，分别测 1 和 4 workers。

Mem0 官方说明 `infer=False` 会跳过结构化抽取并原样存储消息，搜索应通过 `user_id` filter 隔离用户。因此该设置适合隔离检索 data plane；它不代表 Mem0 默认 `infer=True` 的 extraction、冲突消解和 graph 能力。

### 6.2 结果解释

GraphMem 单 worker 的内部 service mean 约 68.8 ms，但 8 clients 下平均 queue 为 496.7 ms，所以 13.99 QPS 的瓶颈是单 worker 串行服务能力。扩到 4 workers 后 service mean 仍约 70.9 ms，QPS 升至 36.35，说明当前路径在小规模 hot-tenant 下有良好的进程级扩展性。

Mem0 的 warmup 单查询约 33--46 ms，但 8 clients 下 mean/p95 增至 438/1002 ms，说明共享 query embedding + embedded Qdrant 在并发时出现明显 contention/tail amplification。其单进程 RSS 393.7 MiB；本地 GPU embedding 服务的内存没有计入。

Mem0 原文写入 2,080 turns 合计约 75.78 秒，即顺序约 36.43 ms/turn；这只是 `infer=False` embedding + insert，不可与 GraphMem 的完整 extraction/build/update 直接比较。

### 6.3 发表前必须扩大的实验

1. 扩为全部 10 个 LoCoMo users，并至少运行 3 次、每次 120 秒，报告均值和 95% CI；
2. clients 取 1/4/8/16/32/64，GraphMem workers 取 1/2/4/8；同时锁定 CPU affinity；
3. 分别报告 cold/warm，进程 RSS + GPU/embedding service 显存 + 外部 vector DB RSS；
4. 增加 `infer=True` 的 end-to-end write/read 样本，并固定 LLM、embedding、reranker、graph 开关；
5. 在相同 32-turn 证据上补 retrieval recall、packed all-hit、answer accuracy 和 prompt tokens；
6. 运行 95:5 与 80:20 读写混合、hot-key skew、用户隔离和故障恢复；
7. 把 Mem0 官方 benchmark 数字只作为外部参考，不能与本仓库不同数据切分/模型配置的结果直接相减。

## 7. 优先级计划

### P0：先解除准确率上限

- 用 atomic extractor + raw fallback 重建完整 snapshot；
- 校准 confidence，补否定、模态、状态变化和指代测试；
- obligation/session/temporal-operand 保底 pack，先攻 Cat1 和 LME multi-session；
- 对证据齐全仍答错的 71 个 annotated case 做 answer-only taxonomy。

### P1：让创新点真正进入执行路径

- 统一全库 embedding contract，取消 400/510 memory 的 deterministic fallback；
- 让 typed cross-session relation 被 executor 实际 walk，并把 fallback dependency 降下来；
- 用 cross-session gold-path recall 与 typed-edge utilization 作为发布门，而不是只测边 precision。

### P2：消除 cold latency 和内存复制

- persisted/mmap read view 与 native pack；
- shared snapshot、cache-aware routing、cold compiler isolation；
- 在 8 workers / 16 clients 基线上把 p95 从 872 ms 降至 500 ms 以下，同时使 RSS 至少下降 40%。

### P3：完整系统论文实验

- 全量 Mem0 matched benchmark；
- accuracy--token Pareto、QPS--p95--RSS 三维 frontier；
- 读写混合、扩展性、故障恢复与 ablation；
- 所有主表仅使用 full run，当前四用户样本作为 protocol pilot 或 appendix。

## 8. 证据文件

- `../artifacts/report/v5_10/error_chain/error_chain.json`
- `../artifacts/v5_10/full_benchmark_20260809/answers/merged/retrieval.jsonl`
- `../artifacts/report/v5_10/multi_tenant_replica2_60s/summary.json`
- `../artifacts/report/v5_10/multi_tenant_w8_c16_60s/summary.json`
- `../artifacts/report/v5_10/multi_tenant_w8_c32_60s/summary.json`
- `../artifacts/report/v5_10/incremental_ha_gate/summary.json`
- `../artifacts/report/v5_10/mem0_comparison_sample/summary.json`
- `scripts/benchmark_v5_10_mem0_sample.py`
- `scripts/benchmark_v5_10_multi_tenant.py`
