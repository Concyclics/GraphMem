# V5.13 原文—图关系—有界召回联合强化

## 1. 当前结论

当前瓶颈不能只归因于 evidence pack。权威 200 题图的构建漏斗为：

| 构建边界 | 指标 | 当前值 |
|---|---:|---:|
| Gold turn → CanonicalFact | turn recall | 81.66% |
| 全部 gold turn 均有 Fact | question all-hit | 68.50% |
| Gold fact pair 直接有 content path | pair recall | 29.83% |
| Gold fact pair 在 2-hop 内连通 | pair recall | 29.83% |
| 全部 fact pair 在 2-hop 内连通 | question all-hit | 29.00% |
| Gold session pair 在 2-hop 内连通 | pair recall | 67.40% |
| Gold fact pair 经 Entity--HAS_FACT owner portal 两跳连通 | pair recall | 67.10% |
| 全部 fact pair 均可经 owner portal 连通 | question all-hit | 64.00% |
| typed relation 数量 | 全图 | 6 |
| 与 gold fact 相邻的 content edge 连接另一个不同 gold turn | edge yield | 2.04% |
| Atomic node 没有 typed neighbour | isolated ratio | 99.98% |
| Typed component 最大规模 | nodes | 2 |
| 每个 memory 最大 typed component 的平均覆盖 | atomic nodes | 0.19% |

更精确地分解后：63 题在 source→fact 阶段已丢失；58 题具备完整 typed/content
path；70 题只能依赖高扇出的 Entity--HAS_FACT owner portal；真正既无 content path
也无 owner portal 的是 9 题。LoCoMo multi-hop 最严重：全部 fact pair 的 typed/content
两跳连通率仅 2%，owner-portal all-hit 也只有 44%。因此旧图并非完全不可达，而是
大量题只能走较宽的 owner 旁路，这正是召回候选膨胀和精度低的构建侧原因。

## 2. 已定位的构建缺陷

1. `recoarsen_report_snapshot.py` 生成了 52,272 个 deferred relation
   candidates，但没有执行 relation refinement；最终
   `same_entity_state / temporal_continuation / causal /
   contradiction_update` 均为 0。
2. 旧宽松图把 typed vocabulary 开放给所有 hierarchy level，导致 typed edge
   主要落在 `routing_card -> routing_card`，而不是 atomic fact。
3. 旧 refiner 不输出方向，causal、temporal continuation、update 的方向由
   node-id 排序决定。
4. LLM 自报 confidence 没有在 materialization 时执行
   `typed_relation_min_confidence`。
5. parent gate 把所有 candidate 的 `cross_session` 硬编码为 true，同 session
   pair 也消耗跨会话 relation token。
6. `ambiguous_only` 同时控制 generic coarse decision 和 atomic typed restoration。
   相似度高于上阈值的跨会话 fact 会直接成为 `coarse_related`，反而不进入
   `coreference / temporal / causal / update` 判别，漏掉了一批最有价值的候选。
7. 旧 recoarsen 只给 session card 提供缓存的 Qwen embedding；关系 gate 下降到
   CanonicalFact/Event/State 后会退化为 hashed lexical vector。因此旧实验并未在
   atomic relation candidate 层真正使用语义向量。

## 3. 已实现的修复

- coarse layer 只允许 `coarse_related`；typed labels 只开放给跨 session 的
  atomic endpoints。
- refiner 输出 `LR/RL/U` 方向；只有 directional relation 使用方向。
- materialization 同时执行 model confidence 与 deterministic structural gate。
- 暂停未校准的 `same_entity_state`；collection relation 由 closed manifest
  确定性构建。
- refiner 跳过 coarse-only candidates，避免把 relation token 花在 HNSW 已负责
  的 generic edge 上。
- generic coarse gate 与 atomic relation restoration 解耦：高相似跨会话 atomic
  pair 即使不在 coarse ambiguity band，也可在 endpoint-degree 与全局线性预算内
  进入 typed refiner。
- 将 `provenance_projected` 保留为负对照并恢复 `hierarchy_only` 安全默认；新增
  `atomic_summary_hybrid`：直接嵌入 CanonicalFact 摘要，并与 hashed lexical HNSW
  各保留 `k=8` 的跨 session 候选。两个通道只进入 typed 候选旁路，不改变
  routing/scene 的 coarse score，也不能材料化为 terminal `coarse_related`。
- 将 owner/predicate/scope/value/time 结构门前推到 LLM 之前。只有至少一种在线
  relation 可能通过的 endpoint pair 才调用 refiner；方向与 confidence 在调用后
  再检查。全量方向审计未过 85% 的 temporal/update/causal 暂不进入在线图。
- source DB 只读，atomic rescue 与 relation rebuild 均生成独立 snapshot。

早期五 memory smoke 的 type-only 小样本曾达到 87.5%--97.83%，但它没有完整检查
方向，不能作为在线门。对联合图全部 24 条 typed edge 做 direction-aware 独立审计
后，整体 precision 只有 70.83%，direction precision 为 83.33%；分类型分别为：
coreference 100%（3/3）、temporal 80%（12/15）、causal 50%（2/4）、update 0%
（0/2）。因此当前默认只允许 coreference 通过一阶段 LLM materialization；其它
三类留给后续独立二阶段 verifier。已有确定性 `AT_TIME / TEMPORAL_BEFORE /
STATE_NEXT` 仍负责时间导航。

第一版 200 题 relation-only 诊断臂保留旧候选策略，只启用新的 atomic
materialization 与结构门：共判别 15,915 个候选，材料化 1,391 条 typed edge，
独立分类型采样 200 条的 typed precision 为 94.5%。但只有 53 条 typed edge 真正
跨 session，gold typed-pair recall 和两跳 path all-hit 均没有任何提升；构建调用
消耗约 946 万 input tokens。该负结果证明判别精度不是主要问题，候选预算被同
session 局部边占用才是问题。后续 arm 固定判别器，只改变跨 session admission
和 atomic semantic vector。

候选向量的 read-only flat-ANN 审计先给出一个负结果：在相同 `k=8`、
cross-session quota=2 下，hashed lexical 的 gold-pair candidate recall / question
all-hit 为 66.79% / 58.5%，provenance-projected Qwen vector 反而为 64.88% /
57.5%。整 turn 投影会让同一 turn 的多个 atomic facts 共享向量，丢失
predicate/value 区分，因此不能替换 lexical channel。两者各自救回一部分题；后续
测试直接 atomic-summary embedding 后确认问题是表示粒度而非 dense 模型本身：

| Relation candidate arm | 每节点候选上限 | Pair recall | Question all-hit |
|---|---:|---:|---:|
| Hashed lexical | 8 | 66.79% | 58.5% |
| Supporting-turn projection | 8 | 64.88% | 57.5% |
| Atomic-summary Qwen | 8 | 70.72% | 65.0% |
| Lexical 4 + atomic 4 | 8 | 68.42% | 61.5% |
| Lexical 8 + atomic 8 | 16 | **76.24%** | **71.5%** |

55,153 个 fact 的 atomic-summary sidecar 共 476 个 batch、939,153 embedding input
tokens，wall time 95.4 秒。固定总预算 4+4 会把 LoCoMo multi-hop all-hit 从纯 lexical
18% 降到 10%；说明该题型不能过早压缩每通道 quota。准确率优先使用 8+8，仍保持
`O(kN)`，然后由结构前置门、per-node/global admission 和 per-type degree cap 控制
LLM 与在线边，而不是直接把 16 个候选全部材料化。

47-session 单 memory smoke 量化了结构门的系统收益。双通道提出 3,491 对，旧的
post-LLM gate 判定 240 对、消耗 151,140 input tokens、材料化 6 条；把同一结构
contract 前推后，只剩 3 对 coreference，判定 input 降到 1,924（-98.7%），最终
3 条全部材料化，独立审计 3/3 有效。该结果只证明执行路径和成本方向正确；全量
200 题重建与 precision/accuracy gate 仍需独立报告。

全量 110-memory 重建随后证明单 memory smoke 过于乐观。双通道共提出 434,905
对，经第一版结构门留下 1,053 对、LLM 实际判断 710 对（101 calls、446,201 input
tokens），材料化后图中有 498 条 coreference。全量独立审计只有 49.18% precision，
所以该 raw graph 不能上线。对 498 条冻结 judgment 做结构重放后，采用
`summary exact OR (value exact AND predicate containment >=0.75)`，并要求 LLM
confidence >=0.88；按 memory hash 拆半的 precision 分别为 91.30%/89.29%。安全
snapshot 保留 51 条、移除 447 条，重新独立审计为 88.24%（45/51），超过 85% 门。

该过程给出两个系统结论：一是 candidate recall 与 materialized-edge precision 必须
分开优化，76.24% 的候选覆盖不能直接转成边；二是结构门必须前置，新代码会先用
高精度 contract 缩小候选再调用 LLM，而不是像 raw 重放那样先花 44.6 万 tokens
再过滤。安全图相对上一版 pruned graph 的 32/48-turn evidence all-hit/recall/
precision 逐题完全一致，但平均每题少访问 1.99 个节点/边；48-turn latency 在两次
独立运行中低 58.3 ms，paired CI [-94.6,-21.0] ms。由于不是同进程交错 A/B，该
latency 只作为方向性结果，后续仍需专门 system benchmark 复核。

把安全 coreference 临时加入默认 traversal 只走了 2 条边：32-turn recall +0.17pp
且 CI 接触 0，all-hit 不变；48-turn 所有 accuracy 指标不变。因此没有修改全局默认
scheduler，仍通过 `obligation_aware_relations` 显式启用，避免历史低精度图受影响。

联合 raw semantic graph 的 `coarse_related` 膨胀到 336,543 条，总边 642,117；
content-path all-hit 仅从 33.0% 提到 35.5%，content-or-owner coverage 仍为 82.5%，
gold-edge yield 反而从 2.83% 降到 2.27%。删除 terminal/atomic generic coarse
edge 后，总边降到 341,426（-46.8%），combined coverage 不变，gold-edge yield
升至 3.26%。因此 generic coarse edge 只允许落在 routing/scene region，terminal
层必须 typed-or-abstain。

在相同 dense sidecar、beam=2 和 pack budget 下，pruned joint graph 相对仅追加
atomic facts 的基线取得稳定 paired 增益：

| Budget | All-hit | Recall | Precision | F1 | Evidence tokens |
|---:|---:|---:|---:|---:|---:|
| 32 | +5.0pp，CI [+2.0,+8.5] | +5.53pp | +0.469pp | +0.853pp | -108 |
| 48 | +6.5pp，CI [+3.5,+10.0] | +3.69pp | +0.229pp | +0.428pp | -199 |

48-turn arm 救回 13 题且无 all-hit regression；最终 all-hit/recall 为 61.0% /
72.79%。因此有效增益来自“atomic rescue 后重新进入 coarsening + sparse routing
graph + terminal coarse pruning”的联合构建，而不是 typed edge 数量。

`shared_referent` scene arm 新增 24,331 条边，将三跳 session all-hit 从 63% 提到
71%，但在 beam=2 的最终 evidence pack 中，48-turn 的 all-hit/recall/precision
全部零增益；32-turn 仅 +0.17pp recall 且 CI 接触 0，同时每题多访问 1.42 个
node/edge。因此它保持实验开关，不进入默认图。结构可达性提升若不能转化成 packed
precision，就不能算 accuracy gain。

## 4. Atomic extraction 独立消融

V5.10 v3 对 109 个已知 missing-fact turns 的 information-unit coverage 为
96.25%，evidence sufficiency 为 75.27%。将结果写入只读源图的副本后：

- 109/109 turns 均成功映射；
- 新增 387 个 CanonicalFact 与 1,067 条 provenance/collection/state edge；
- gold-turn fact recall 从 81.66% 提到 95.21%；
- question-level fact all-hit 从 68.50% 提到 89.50%。

在相同 frozen dense graph 上做 paired retrieval：

| Budget | All-hit 变化 | Gold recall 变化 | 结论 |
|---:|---:|---:|---|
| 32 turns | +1.0pp，CI [0,+2.5pp] | +0.96pp，CI 跨 0 | 尚不稳定 |
| 48 turns | +2.5pp，CI [+0.5,+5.0pp] | +2.10pp，CI [+0.83,+3.68pp] | 稳定正增益 |

Fact coverage 的大幅提升只转化成中等 packed gain，说明新 fact 仍需进入新的
coarsening/typed relation snapshot，不能只追加节点。

## 5. 构建侧目标形态

新的关系构建不是把每对事实都交给模型，而是一个四段式有界漏斗：

1. **多视图原子化。** 原文先生成可追溯的 CanonicalFact/Event/State 单元；
   predicate、owner、scope、polarity、event/observed time 是建边字段，不只放在
   summary 文本里。
2. **Coarsening candidate gate。** HNSW 先在 routing/scene 每层建立固定度数近邻，
   再只在通过 parent gate 的 child scope 内下降。atomic relation 另走 lexical 8 +
   atomic-summary 8 的 bounded cross-session 旁路；两者不共享 generic coarse edge。
   候选比较仍受 `O(kN)` 上界约束。
3. **Typed relation restoration。** 先用 owner/predicate/scope/value/time contract
   删除任何 label 都不可能通过的 pair，再把剩余 atomic endpoints 交给
   `coreference / temporal_continuation / causal / contradiction_update`；模型必须
   输出方向和置信度。当前仅 coreference 过在线门，其余进入二阶段 verifier；普通
   routing card 只保留 `coarse_related`。
4. **校准与稀疏化。** 模型判定还需通过 owner/predicate/scope 等结构一致性门；
   最终以 per-type degree cap 和全局预算做 degree-constrained selection。低置信边
   留在 deferred ledger，不写入在线导航图。

这一区分了三种容易混淆的数量：candidate recall 衡量候选生成是否漏边；typed
edge precision 衡量材料化边是否可靠；gold path coverage 衡量这些边能否真正形成
证据路径。三者必须同时报告，不能只看最终召回。

后续构建优化按以下优先级进行：

| 优先级 | 调整 | 要解决的问题 | 验证指标 |
|---|---|---|---|
| P0 | atomic rescue 与 relation rebuild 合并成同一新 snapshot | 新事实未进入层次和关系图 | fact all-hit、path all-hit |
| P0 | 高相似跨 session atomic pair 绕过 coarse ambiguity 限制 | coarse 高置信导致 typed 漏边 | candidate recall、typed path recall |
| P0 | relation-specific structural admission 与方向恢复 | 错边和反向边污染导航 | per-type edge precision ≥85% |
| P1（已实现候选旁路） | lexical + atomic-summary 双通道 ANN，8+8 quota 后去重 | 单一 embedding 漏掉 predicate 不同但可组合的 multi-hop 边 | gold pair candidate recall 66.79%→76.24% |
| P1 | 每种 relation 独立 degree cap，而非共享一个总 top-k | coreference hub 淹没 temporal/causal 边 | type coverage、hubness p95 |
| P1 | temporal/update/causal 二阶段 verifier，只复核结构可行且一阶段通过的 edge | 在不放宽在线图的前提下提高 recall | directional precision ≥85%、precision-recall Pareto |
| P2 | incremental local recoarsening + versioned edge swap | 新 memory 写入时重建延迟过高 | write p95、read availability |

构建图的停止条件不是“边越多越好”。目标是将 typed isolated ratio 大幅降低，
同时把 atomic typed degree 的 p95 控制在小常数（建议 4--8），并避免出现跨主题
supernode。若扩大 degree 只增加候选量而不提高 gold-path coverage，应判定为噪声，
不能依靠召回侧继续扩展来补救。

## 6. 有界剪枝原则

有限剪枝是必要的，但必须位于高精度关系构建之后。当前可接受的策略是：

1. 关系生成：每个 parent gate 只保留 top-k child pairs，使比较与 relation
   decision 均为 `O(kN)`。
2. 图遍历：每个 expansion 使用小 beam，并由 QueryIR obligation 选择 relation
   type；typed edge 不与所有 generic edge共享无差别队列。
3. Proof safety：已经构成一个 operand/proof unit 的全部 witness 必须原子保留，
   不能逐 turn 剪断。
4. Safety channel：保留 bounded lexical/dense beam；typed path 低置信或 obligation
   未闭合时触发 fallback。
5. Evidence pack：同时报告 precision、recall、F1、all-hit 和 answer accuracy，禁止
   用全量候选提高 coverage 冒充有效召回。

直接把 candidate pool 截到固定 top-k 不是安全方案。历史 `session_router_k=8`
在 48-turn budget 下将 all-hit 从 52% 降到 48.5%，LoCoMo multi-hop recall 从
约 38.3% 降到 33.9%；剪枝必须由关系质量、proof completeness 和 fallback
共同控制。

在当前 pruned joint graph 上补做 post-score candidate reservoir 截断：平均候选
536.7 条，截到 256 或 128 后，32/48-turn 的最终 all-hit/recall/precision 均逐题
完全不变；但 candidate all-hit 分别从 100% 降到 91.5% 和 79.5%，LoCoMo
multi-hop 在 128 时只剩 40%。两档 latency 的 paired CI 均跨 0，因为该参数位于
候选打分之后，不会减少前面的图遍历和融合开支。因此它只能作为内存/packer
reservoir 上限：256 是当前相对安全的实验值，128 不应作为默认；真正降低 latency
要在构建端减少无效边和在 traversal 前做 typed/proof-aware admission。

## 7. 正交实验矩阵

| Arm | Atomic facts | Typed relations | Dense safety | 目的 |
|---|---|---|---|---|
| A0 | frozen | frozen | off | 权威稀疏基线 |
| A1 | V5.10 rescue | frozen | off | source→fact 因果增量 |
| R1 | frozen | high-precision | off | relation-only 增量 |
| J1 | V5.10 rescue | high-precision | off | 联合增量 |
| J2 | V5.10 rescue | high-precision | on | dense safety 的剩余价值 |

每个 arm 固定 QueryIR、ranking 与 answer prompt，扫描 32/48 turns，以及 traversal
beam 1/2/4。进入 answer judge 的最低门槛：paired all-hit/recall 不回退、precision
不下降、LoCoMo multi-hop 改善，且 typed-edge endpoint precision 至少 85%。

## 8. 目标与风险

80% LongMemEval / 90% LoCoMo 是 end-to-end answer 目标，不可由 retrieval proxy
宣称达成。当前最可能的提升链为：atomic coverage 消除 source loss，typed
relations 提高 multi-hop path retention，bounded proof-aware pack 将完整路径压入
48-turn context，最后由 uncapped answer 生成。剩余风险是 LoCoMo gold relations
需要跨不同 predicate/value 的隐式语义桥；过强的 deterministic gate 可能提高
precision 但限制 recall，因此后续应增加仅作用于少量候选的二阶段 relation
verifier，而不是重新放宽第一阶段阈值。
