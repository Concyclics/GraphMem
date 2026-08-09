# GraphMem V5.10：完整实验、误差链与下一步强化路线

日期：2026-08-09

## 1. 结论先行

V5.10 已完成三个层面的实现与验证：低成本分层关系图、QueryIR 编译检索路径，以及面向多租户的增量/高可用数据面。当前证据支持以下结论：

1. **低成本建图成立于有界候选复杂度。** 真实 HNSW assignment 加 parent-gated relation frontier 在 1K--20K 受控图上的候选工作量指数为 1.100；20K 时只检查 all-pairs 的 1.57%，同时保留 97.32% gold edge recall 和 97.60% 两跳可达率。
2. **检索路径的系统收益明确。** Native seed fusion 将 dev200 的平均检索延迟从 575.7 ms 降至 307.2 ms；全量 V5.10 相对 V5.9 的 retrieval p95 在 LongMemEval/LoCoMo 分别下降约 45.3%/93.4%。
3. **Token 与证据完备率可同时改善，但尚未解决 Multi-hop。** obligation-aware packer 在 dev200 将 all-hit 从 43.0% 提高至 45.5%，平均 evidence Token 降低 25.8%，无 all-hit 回退；然而全量 LoCoMo Category 1 的 packed all-hit 仍只有 18.79%。
4. **端到端准确率尚不能写成统计显著提升。** LongMemEval 为 72.00%，相对 V5.9 为 -0.60 pp（McNemar exact `p=0.828`）；LoCoMo 为 83.57%，相对 V5.9 为 +1.62 pp（`p=0.055`）。后者接近、但未越过 0.05。
5. **当前全量实验不是 V5.10 extraction 的端到端验证。** 它复用了冻结的 P8 fact projection，只重建层级关系并升级 H11 QueryIR、native seed fusion 与 packing。因此 atomic extractor 的收益只能引用独立 Gate，不能归因到全量 QA。

## 2. 实验范围与可复现契约

全量运行包含 LongMemEval 500 题与 LoCoMo Category 1--4 的 1,540 题，共 2,040 题。回答与 judge 均使用本地 `Qwen3-30B-A3B-Instruct-2507-FP8`，temperature 0、reasoning 关闭。精确 tokenizer 为：

```text
../artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json
sha256 prefix: aeb13307a71acd8f
```

全量图包含 510 个 Memory、886,775 个节点和 1,978,940 条边。重建耗时 419.95 s；图文件约 3.22 GB。原始答案、retrieval trace、judge 调用和合并 manifest 均按 question ID checkpoint，并验证恰好包含 2,040 个唯一题目。

主要机器可读入口：

- 全量结果：`../artifacts/report/v5_10/full_benchmark/summary.json`
- 全链路误差：`../artifacts/report/v5_10/error_chain/error_chain.json`
- 报告数据包：`../GraphMem_report/generated/v5_10_tables.json`
- 来源哈希：`../GraphMem_report/generated/v5_10_experiment_manifest.json`

## 3. 三个 Contribution 的实现与证据

### C1：Token-efficient Hierarchical Relation Index

核心不是“首次用分层图”，而是把关系建图改写成：

\[
G_0 \xrightarrow{\mathrm{HNSW\ assignment}} G_1,\ldots,G_L
\xrightarrow{\mathrm{parent\ gate}}
\mathcal C_{rel}
\xrightarrow{\mathrm{bounded\ refine}} E_{rel}.
\]

当 HNSW 每点候选数、parent fanout 和 refine degree 均为常数时，候选工作量为：

\[
T(N)=O(N\log N)+O(kN)+O(rN),
\]

其中 (k) 是 ANN/局部候选上限，(r) 是每节点 refine 上限。与 all-pairs 的 (O(N^2)) 不同，实际拟合指数应接近 1。

实现包括：

- 真实 HNSW balanced coarse assignment，移除 node-id 补位；
- parent-gated coarse-to-fine relation restoration；
- 跨 session neighbour quota；
- `uncertainty × bridge value × query value / token cost` 优先级；
- per-node 与 per-1K-node 双重 refine admission cap；
- typed relation schema 和 bounded selective refiner。

| 指标 | V5.10 结果 | 结论 |
|---|---:|---|
| 1K--20K candidate-work exponent | 1.100 | 通过 ≤1.15 Gate |
| 20K work / all-pairs | 1.57% | 候选规模显著降低 |
| 20K gold edge recall | 97.32% | 合成结构质量保持 |
| 20K gold ≤2-hop reachability | 97.60% | 支持多跳路径保持 |
| dev200 两跳 gold-session path | 52.11%→93.46% | 图结构可达性明显提升 |
| dev200 all-hit | 45.5%→46.0% | QA evidence 增益仍小 |

重要边界：20K 时生成 176,537 个 refine candidate，只接纳 11,029 个；该预算化是接近线性复杂度的必要组成，而非免费收益。typed relation judge 的小样本为 15/15，但 dev200 traversal 实际走过的 typed edge 为 0，因此当前不能宣称 typed restoration 带来准确率提升。

### C2：Low-latency Compiled Graph Retrieval

QueryIR 将查询编译为单一 AST，并由物理规划器生成 route、operand binding、relation algebra、evidence obligation 与 certificate：

\[
q\rightarrow \mathrm{AST}\rightarrow
\{O_i\}\rightarrow \pi_{route}\rightarrow
\mathcal R_{candidate}\rightarrow
\operatorname{Pack}_{B}(\{O_i\})\rightarrow Cert.
\]

证据选择被写成预算约束的覆盖问题：

\[
\max_{S}\sum_i w_i\mathbf 1[O_i\text{ covered by }S]
-\lambda\sum_{a,b\in S}\operatorname{Redundancy}(a,b),
\quad
\sum_{a\in S}\operatorname{Token}(a)\le B.
\]

实现包括：

- H11 单一 AST、稳定 operand ID 和编译置信度；
- operator-aware route 与 ancestor corridor；
- native BM25/posting seed fusion；
- cited span、小窗口、operand/session/time-stage floor；
- post-pack certificate 与 false-complete 防护；
- Temporal/Set executor shadow path，真正 bypass 继续关闭。

| 消融 | 质量变化 | 成本变化 |
|---|---:|---:|
| baseline pack→obligation pack | all-hit 43.0%→45.5%；recall 56.63%→59.93% | mean evidence Token 3289.6→2441.7（-25.8%） |
| H10 split→H11 unified | all-hit 46.0%→48.5%；false-complete 14.5%→4.0% | -36.8 mean evidence Token；延迟差 CI 跨 0 |
| SQLite FTS→native seed | all-hit 48.5%→49.5%；recall +0.42 pp | mean retrieval 575.7→307.2 ms；seed 259.7→13.8 ms |

H11 在 LoCoMo Multi-hop 子集出现 recall 下降，dev200 总体也有 3 个 all-hit 回退样本；因此 H11 是当前推荐实验主路径，但还不是无需 shadow 的最终稳定编译器。确定性 executor 尚未达到独立 holdout precision ≥99.5%，不得开启真正回答 bypass。

### C3：Scalable, Incremental and Available Execution Plane

查询面采用 immutable read view、memory affinity、bounded admission、per-tenant quota、rendezvous candidate shard 和 power-of-two choice。写入面采用：

```text
RECEIVED → RAW_DURABLE → FACT_INDEXED → RELATION_INDEXED → ROUTE_PUBLISHED
```

状态与 authority graph version 使用同一 SQLite 事务 CAS；每个阶段具备 job ID、source offset、payload hash、事件日志和幂等重放。只读 worker 使用 snapshot checksum，主指针通过原子 `LATEST` 更新；stale promotion 和 corrupt snapshot 会被拒绝。

| 多租户配置 | QPS | p95 / p99 | Worker RSS | 错误/超时/错读 |
|---|---:|---:|---:|---:|
| 4 workers, 16 clients, replica=1 | 17.03 | 2616.7 / 3189.0 ms | 2315 MiB | 0 / 0 / 0 |
| 4 workers, 16 clients, replica=2 | 20.14 | 1432.9 / 1800.1 ms | 2433 MiB | 0 / 0 / 0 |
| **8 workers, 16 clients, replica=2** | **51.28** | **872.4 / 1132.4 ms** | **4722 MiB** | **0 / 0 / 0** |
| 8 workers, 32 clients, replica=2 | 57.64 | 1278.7 / 1768.0 ms | 4745 MiB | 0 / 0 / 0 |

8w/16c 是当前 `p95≤1s` 的推荐容量点；把客户端加到 32 只增加 12.4% QPS，却使 p95 增加 46.6%，已进入排队饱和区。burst probe 明确拒绝过载请求，未形成无界队列。

增量与 HA 结果：

| 指标 | 结果 |
|---|---:|
| Raw durable p95 | 9.97 ms |
| Fact authority commit-only p95 | 0.69 ms |
| Relation authority commit-only p95 | 0.70 ms |
| Route plan / publish p95 | 4.36 / 17.94 ms |
| 每次触及行数 | 5 |
| 平均重算比例 | 0.164% |
| 4 并发 reader 观测 | 158 次，0 error，0 torn read |
| Worker SIGKILL recovery | 481.5 ms，1 restart，1 in-flight retry，0 failure |
| 706.7 MB snapshot copy | 1.87--2.35 s |
| Follower promotion | 1.24 s，checksum 通过 |

这里的 fact/relation 数字只测 authority commit，不包含远端 extraction/embedding；32 个新 session 中有 26 个请求 background rebalance，说明在线插入已经正确可见，但 partition split/merge 尚未真正实现。

## 4. 全量 Benchmark

| Benchmark | Accuracy | vs V5.9 | McNemar exact p | Turn all-hit | Prompt Token mean/p95 | Retrieval p95 |
|---|---:|---:|---:|---:|---:|---:|
| LongMemEval (500) | 72.00% | -0.60 pp | 0.8284 | 64.00% | 5085 / 6584 | 1968.2 ms |
| LoCoMo Cat.1--4 (1540) | 83.57% | +1.62 pp | 0.0551 | 60.53% | 2578 / 2859 | 130.3 ms |

相对 V5.9：

- LongMemEval all-hit 从 56% 提升到 64%，prompt mean 从 6169 降至 5085（-17.6%），retrieval p95 从 3596.0 降至 1968.2 ms（-45.3%）。p95 prompt 从 6442 增至 6584（+2.2%），不能写成所有 Token 指标都下降。
- LoCoMo all-hit 从 58.06% 提升到 60.53%，prompt mean/p95 分别降低约 2.2%/2.0%，retrieval p95 从 1989.3 降至 130.3 ms（-93.4%）。
- LongMemEval 新对 41 题、旧对 44 题；LoCoMo 新对 91 题、旧对 66 题。当前只有方向性收益，没有相对 V5.9 的统计显著准确率提升。

LoCoMo official token-F1 为 22.43%，只覆盖本报告采用的 Category 1--4；它与 LLM judge accuracy 的目标和尺度不同，不应混合比较。

## 5. 从原文到答案的误差链

### 5.1 原文 → Fact Index

- 全图 252,489 个 source turn 中只有 120,287 个至少有一条 CanonicalFact，turn-level coverage 为 47.64%。这不是 precision 指标，但说明 Fact-only 无法替代 raw evidence。
- LongMemEval gold turn 的 Fact recall 为 78.80%，100 个标注问题中只有 67 个全部 gold turn 有 Fact。
- LoCoMo gold turn 的 Fact recall 为 82.88%，1,533 个可标注问题中 1,184 个全部覆盖。
- 冻结 P8 的 216,070 条 CanonicalFact confidence 全部精确等于 0.5，置信度没有区分能力。
- 新 atomic extractor 独立 Gate 的 unit coverage 为 96.25%，但 negation/modality 只有 86.67%/88.89%，sufficiency 75.27% 低于 80% Gate；尚不能直接全量替换。

在 LoCoMo 中，“全部 gold turn 有 Fact”的条件准确率为 85.98%，缺 Fact 时为 75.36%，差 10.62 pp，bootstrap 95% CI 为 [5.68, 15.61] pp。这是强相关证据，但不是随机消融，不能直接宣称补 Fact 必然带来同等因果增益。LongMemEval 100 题小样本上方向相反且 CI 跨 0，说明还存在题型混杂。

### 5.2 Fact/Session → Relation Graph

- 全量图 `coarse_related` 653,424 条，但可识别的跨 session 边只有 1,407 条（0.215%），覆盖 72/510 个 Memory。
- 关系 confidence 均值为 0.996，与未校准的抽取 confidence 形成明显不一致，可能存在阈值饱和。
- 目标 DB 的 embeddings 表为空。recoarsen 中只有 110/510 个 Memory 使用了 5,007 个 Qwen session vector，其余 400 个使用确定性 fallback；全量结果不是统一 dense 条件。
- 135,372 个 collection manifest 中 82.57% 只有一个 member，不能把 single-member observation 当成 closed-world 集合闭包。

图关系可达率在 LongMemEval/LoCoMo 分别为 36.0%/64.91%，而最终 candidate all-hit 为 99.0%/99.80%。两者并不矛盾：seed/raw fallback 可以绕开 relation path；也说明当前准确率主要不是靠 typed multi-hop relation walk 获得。

### 5.3 Candidate → Final Pack

| Benchmark | Candidate all-hit | Packed all-hit | 错题中缺 packed gold | gold 齐但仍答错 |
|---|---:|---:|---:|---:|
| LongMemEval | 99.00% | 64.00% | 28 | 20 |
| LoCoMo | 99.80% | 60.53% | 201 | 51 |

LoCoMo packed all-hit 为真时 accuracy 94.50%，为假时 66.78%，差 27.73 pp，95% CI [23.70, 31.74] pp。候选池已经接近饱和，最直接的下一步不是继续扩大 seed reservoir，而是提高 obligation coverage/token、降低冗余并确保多 operand/multi-session floor。

LoCoMo Category 1 是最主要缺口：accuracy 79.43%、session all-hit 57.80%、graph reachable 30.14%、packed all-hit 18.79%。Category 2/4 packed all-hit 为 73.83%/72.77%，说明通用单跳和时间问题明显好于跨会话多跳。

### 5.4 Final Pack → Answer

证据齐全仍答错的样本在 LongMemEval/LoCoMo 为 20/51；这部分需要按算术/时间归一化、集合闭包、更新冲突、实体绑定和 answer verbosity 分类。certificate complete 在 LoCoMo 中与 +6.90 pp 条件准确率相关，但真正 deterministic bypass 仍关闭；只有独立 holdout 达到 precision ≥99.5%、false-complete ≤0.5% 后才能开启。

## 6. 当前缺陷：算法与工程分开看

### 6.1 算法设计缺陷

1. **Atomic contract 还不够 lossless。** 否定、模态和 state change 仍是薄弱字段；coverage 高不等于回答充分。
2. **关系恢复未形成真实 typed traversal 增益。** 图的两跳结构更连通，但 typed edge 在 dev200 中零使用；必须补 relation recall 和 QA 因果消融。
3. **全量向量条件不一致。** 400/510 Memory 使用 fallback，使 HNSW+dense 的贡献无法单独归因。
4. **Packing 仍以全局排序为主。** Candidate 近饱和却丢掉大量 gold，尤其 Multi-hop operand/session floor 不够强。
5. **闭包假设不安全。** single-member manifest 过多，Count/Set executor 不能据此证明 closed world。
6. **编译器仍有退化样本。** H11 的总体 false-complete 更低，但 LoCoMo Multi-hop recall 有下降；owner/binding 还缺独立 98% holdout。
7. **Cap 诊断语义过粗。** 当前日志几乎每题都标记 hop/frontier/turn cap，不能用于判定哪一项真正造成 evidence loss，需要记录“仍有高优先级义务未满足时触顶”。

### 6.2 工程实现缺陷

1. **新 session 只完成局部插入。** 26/32 需要 background rebalance，但 split/merge、delta HNSW merge 与 parent summary 重编译尚未落地。
2. **Fact/relation visible SLO 尚未实测。** 0.69/0.70 ms 是 commit-only，不包含模型服务；目标的 2 s/5 s 仍是未验证 SLO。
3. **复制还是全 snapshot。** 706.7 MB 每次复制需要 1.87--2.35 s；缺少 append-only change log、增量 page/delta shipping 和 snapshot retention/GC。
4. **只验证了 worker crash。** writer crash、disk full、network partition、remote LLM/embedding timeout 与整机 primary promotion 尚未形成系统化 fault matrix。
5. **容量以高 RSS 换吞吐。** 8 workers 约 4.72 GiB；需要共享 mmap/只读结构、compact arrays 和热 Memory 分层，避免 worker 数线性复制。
6. **全量 judge 仍有模型格式波动。** runner 已支持按题 checkpoint、resume 与 JSON repair，但最终论文应做 3 次重复、异构 judge 和 discordant 人工复核。

## 7. 下一步强化顺序与实验矩阵

### P0：先重建原文 → 索引链

1. 修复 negation/modality/state-change scanner 与事实合并，所有 coverage contract 失败的 turn 保留 deterministic raw span fallback。
2. 在 untouched holdout 上验证 precision、recall、sufficiency 和 reliability calibration。
3. 全量重建 510 个 Memory 的 atomic facts 与统一 Qwen session embeddings，禁止 fallback arm 混入主结果。

实验：`frozen P8 / V5.10 fact-only / V5.10 fact+raw fallback / raw oracle`，同时报告 extraction Token、build latency、gold fact recall、QA 和 confidence ECE。

### P0：再验证真实 Relation Restoration

1. 从真实跨 session gold pair 构造 relation test set；分层报告 candidate recall、typed precision/recall、≤2-hop reachability。
2. 对比 `ANN-only / HNSW hierarchy / +parent gate / +typed / +selective refine`。
3. 固定 build Token 和 relation decision 上限；报告被 frontier cap 丢弃的 gold bridge。

只有 `+typed/+refine` 在相同预算下使 LoCoMo Category 1 或 LME Multi-session 至少 +2 pp，才能把 typed relation 写入核心主结果。

### P0：强化 Obligation-aware Packing

1. 给每个 AST operand、session cluster、time stage 和 relation witness 建立 minimum coverage floor。
2. 以 marginal obligation coverage/token 做选择，并用 MMR 去重；对数字、日期、否定和 update 保留不可丢 span。
3. 新增 pack oracle：从 candidate reservoir 中求近似最大 gold coverage，用于量化“排序不足”还是“预算不足”。

目标：同 5K Token 下 LoCoMo Category 1 all-hit 18.8%→≥35%，LME Multi-session 48%→≥55%；mean evidence Token 不增加超过 10%。

### P1：完成增量与 HA 数据面

1. background split/merge、delta HNSW、parent summary 重编译和 typed relation outbox；每阶段幂等重放。
2. 端到端测量 raw durable、fact-visible、relation-visible 和 first-query-visible，而非只测事务提交。
3. change-log replication + 定期基线 snapshot；加入 retention、GC、checksum scrubbing。
4. 30--60 分钟 Zipf mixed read/write workload，注入 writer crash、disk full、corrupt delta、network partition 和服务 timeout。

### P1：最终论文冻结

1. 用统一 V5.10 extraction + embedding 重跑全部 2,040 题。
2. 每个主 arm 运行 3 次，报告均值、方差、paired bootstrap CI 和 McNemar exact p；开发集与 final split 严格分离。
3. Accuracy 非劣/提升 Gate 与 Token、p95、RSS 联合判断；禁止单独挑选一项最好结果。
4. 更新 Overleaf 时只导入 `v5_10_experiment_macros.tex` 和带来源哈希的图表，不手抄数字。

## 8. Gate 状态表

| Gate | 目标 | 当前 | 状态 |
|---|---|---:|---|
| Atomic gold fact recall | ≥95% | 独立 unit coverage 96.25%；冻结全量 gold fact recall 78.8%/82.9% | **未通过全量端到端** |
| Negation/modality | ≥97% | 86.7%/88.9% | **未通过** |
| HNSW candidate exponent | ≤1.15 | 1.100 | **通过** |
| Real gold ≤2-hop | ≥90% | 合成 97.60%；dev session 93.46% | **结构通过，真实 relation 待补** |
| Typed relation P/R | ≥85%/≥85% | 15 个 precision 样本；recall/usage 不足 | **未通过** |
| Packer non-inferiority | no all-hit regression, lower Token | dev200 5 improve/0 regress，Token -25.8% | **通过开发集** |
| Executor bypass | precision ≥99.5% | shadow only | **保持关闭** |
| LME retrieval p95 | ≤1.0 s | 1.968 s | **未通过** |
| LoCoMo retrieval p95 | ≤0.75 s | 0.130 s | **通过** |
| 8w16c serving p95 | ≤1.0 s | 0.872 s，51.3 QPS | **通过** |
| Raw durable p95 | ≤20 ms | 9.97 ms | **通过** |
| Fact/relation visible | ≤2 s/≤5 s | 未包含远端服务 | **未验证** |
| Worker crash recovery | bounded retry, no wrong read | 481.5 ms，0 failure | **通过探针** |
| Primary HA RPO/RTO | raw RPO=0，RTO≤60 s | snapshot/promotion probe | **部分完成** |

## 9. 报告可直接使用的文件

- LaTeX macros：`../GraphMem_report/generated/v5_10_experiment_macros.tex`
- 完整表数据：`../GraphMem_report/generated/v5_10_tables.json`
- 端到端图：`../GraphMem_report/figures/v5_10_accuracy_latency.{svg,png}`
- 误差链图：`../GraphMem_report/figures/v5_10_error_chain.{svg,png}`
- 容量/HA 图：`../GraphMem_report/figures/v5_10_capacity_ha.{svg,png}`
- 数据来源与 SHA-256：`../GraphMem_report/generated/v5_10_experiment_manifest.json`

这些文件目前与原有 `experiment_macros.tex` 并存，没有自动改写报告正文，避免在结论审阅前把 V5.9 和 V5.10 数字混在同一个命名空间。
