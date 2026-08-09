# V5.14 关系属性粗化与查询联动实验

## 1. 目标与结论

本轮验证如下假设：上层图不只传播一个通用 semantic cosine，而是携带
`scene / entity / state / temporal / collection` 关系信号；父分区连边后，子层只沿
仍满足相同关系信号的 pair 下钻；查询侧再根据 QueryIR 的 operand 与 proof
obligation 选择对应属性的边，并在跨分区后下钻到 Fact。

结论分三层：

1. **机制成立。** 合成测试中，两个 session 的文本语义不相似，但下层 fact 具有
   同一 entity+predicate 时，旧 semantic gate 不产生边，relation-mask gate 可以
   建立上层边并将 atomic pair 送入 coreference 模糊判别；候选比较与每种关系度数
   仍受常数上限约束。
2. **朴素多视图扩边无效。** 在 200 题上，lexical 8 + atomic-summary 8 的 pair
   recall/all-hit 已为 76.2355%/71.5%；再加入 entity 4 + state 4 + temporal 2 +
   collection 2 后仅变为 76.2885%/71.5%。增加 48,807 个 pair、约 90 万次比较，
   只使两道 LoCoMo multi-hop 的局部 pair coverage 轻微增加。因此 atomic 多视图
   被拆为独立、默认关闭的研究开关。
3. **当前收益受构建属性质量限制。** 最终修正版图相对 V5.13 safe 只增加 185 条边；
   32-turn 多找回一个 gold turn，总体 recall +0.167pp、precision +0.0156pp，all-hit
   不变；48-turn 所有 accuracy 指标不变。entity/state mask 对 LoCoMo multi-hop 的
   gold session-pair recall 只有 14.67%，question all-hit 只有 10%。正确关系边尚未
   建出时，增加查询 beam 或复杂调度不能恢复缺失证据。

V5.14 因而是一个**机制与诊断版本**，尚不能作为准确率主结果。最终保留的是安全、
可消融的 relation-mask 构建和 query-aware descent 实现；默认配置仍关闭实验开关。

## 2. 实现

### 2.1 Relation-aware Coarsening

每个节点或区域聚合以下有界特征：

- low-document-frequency entity；
- predicate token 与完整 predicate phrase；
- scope phrase 与 collection key；
- value；
- observation/event time point。

对关系类型 \(r\) 计算独立分数：

\[
S_r(u,v)=w_s\cos(u,v)+w_eE(u,v)+w_pP(u,v)+w_cC(u,v)+w_t e^{-\Delta t/T}.
\]

父层 gate 携带

\[
M(u,v)=\{r\mid S_r(u,v)\ge \tau_r\},
\]

子层只保留 \(M_{child}=M_{parent}\cap M(u',v')\) 非空的 pair。每层仍执行 endpoint
degree cap，因此工作量保持 \(O(kN)\)，而不是恢复全量 child cross-product。

为避免语义膨胀，time-near 与 collection-compatible 只作为 descent hint，不能单独
越过 0.78 在线材料化阈值；上层 state 必须共享低频 entity。常见 speaker 只有在
atomic fact 层同时满足精确 predicate 时才可参与 state candidate。

### 2.2 多视图候选

候选采用各视图独立配额后取并集，而不是共享总 top-k：lexical、atomic-summary、
entity posting、state composite posting、temporal neighbour、collection posting。
每个稀疏视图还执行两端 b-matching cap，避免 source-local top-k 汇聚成 incoming hub。

完整审计证明 atomic 稀疏视图成本收益不合格，因此 `atomic_relation_multiview` 默认
关闭；parent `relation_mask_propagation` 与它解耦。

### 2.3 QueryIR 联动

在线 `coarse_related` 通过 edge source 保留越过高阈值的稀疏 mask。QueryIR 只将
mask 用于 routing priority，不将其视为事实关系或 proof：

- `state_history / ordering` 优先 state-compatible；
- collection obligation 优先 collection-related；
- 多 operand 优先 shared-entity；
- temporal obligation 优先 temporal-near。

需要注意，调度器已经支持五类 mask，但最终保守图实际只材料化了 scene、entity 和
state 标签。time/collection 当前只能参与父子 gate 的候选过滤，不能单独越过在线
材料化阈值，因此这两个查询分支在本轮图上没有可消费的边。后续应先构建“同实体或
同 activity 下的近时间”和“同实体+精确 collection key”复合边，再开放对应查询配额；
不能直接降低纯时间/collection 阈值，否则会恢复饱和图。

若查询通过含 entity/state/time/collection 的 mask 跨到另一区域，下一次 expansion
允许 `SCENE_CONTAINS / REFINES_TO` 进入正常 beam，使两跳执行“跨区定位 → 结构下钻”，
而不是只有全局队列为空时才下钻。纯 scene-similar 边仍使用原保守策略。

### 2.4 工程正确性修复

- deterministic typed edge 的 `directed` 由 relation vocabulary 决定；coreference
  不再错误写成有向边；
- 无向边的 degree cap 同时约束 src 和 dst，阻止隐式 incoming hub；
- 构建 manifest 增加 mask pair、各 signal、各 candidate source 统计；
- 新路径为配置开关，旧 snapshot 与默认配置不受影响。

最终完整构图还验证了开关接线：parent relation-mask propagation 开启，atomic
entity/state/time/collection 多视图关闭；atomic candidate source 只有 lexical 和
semantic 两类。此前调试 arm 中发现并修复了这两个开关条件写反的问题，该 arm 不作为
最终结果。

## 3. 实验结果

### 3.1 Atomic candidate audit

| Arm | Pair recall | Question all-hit | 结论 |
|---|---:|---:|---|
| Hashed lexical k=8 | 66.7897% | 58.5% | 单稀疏通道 |
| Atomic-summary k=8 | 70.7179% | 65.0% | 单 dense 通道 |
| Lexical 8 + atomic 8 | 76.2355% | 71.5% | V5.13 基线 |
| 上述 + entity/state/time/collection | 76.2885% | 71.5% | +0.053pp，无 all-hit 增益 |

### 3.2 构图 arm

| Arm | Edges | Session pair 2-hop recall | Session all-hit | 主要问题 |
|---|---:|---:|---:|---|
| V5.13 safe | 333,704 | 67.40% | 63.0% | 基线 |
| 朴素 mask | 338,640 | 70.00% | 65.5% | path 增益不转化为 evidence |
| 饱和 typed metadata | 338,578 | — | — | 26,027 条边同时拥有四种 mask，失去区分度 |
| 最终 relation mask | 333,889 | 67.78% | 63.0% | 安全稀疏，但正确 entity/state 边不足 |

最终 relation-mask 图的在线 coarse 边为：

- `scene_similar`: 27,843；
- `scene_similar + shared_entity + state_compatible`: 313；
- `shared_entity`: 162；
- `scene_similar + shared_entity`: 13；
- `shared_entity + state_compatible`: 4。

其中 231 条 entity/state 边为 Scene→Scene，261 条为 RoutingCard→RoutingCard。
完整 110-memory 构图执行 10,837,091 次有界 relation comparison，接受 28,338 条
relation-mask 路由边，未调用 LLM relation refiner。

相对 V5.13 safe，直接 session-pair recall 从 55.68% 提升到 56.52%，两跳 recall 从
67.40% 提升到 67.78%，但 question all-hit 仍为 63%。失败分解也没有移动：66 题有
content path，99 题只能依赖 owner portal，14 题缺关系路径，21 题在原文到 Fact 阶段
已经缺失。这说明新增边主要补充了已有可达题目的局部 pair，而没有跨过题级门槛。

### 3.3 查询结果

相对 V5.13 safe 的逐题配对结果：

| Budget | All-hit | Recall | Precision | Gold hits | Visited nodes/edges |
|---:|---:|---:|---:|---:|---:|
| 32 turns | +0pp | +0.167pp，CI [0,+0.5pp] | +0.0156pp | +0.005/题 | -0.190/题 |
| 48 turns | +0pp | +0pp | +0pp | +0 | -0.190/题 |

唯一 accuracy 变化为 `locomo09_0046`：32-turn 从 0/3 gold 提升到 1/3。LoCoMo
multi-hop 平均 recall 从 36.23% 到 36.90%（+0.667pp），但 all-hit 仍为 16%。
typed-region descent 全集只实际走了 4 条 `scene_contains`，表明可触发的高质量
Scene→Scene entity/state 边仍过少。

查询联动是必要的：如果只建边但不在命中 region 后执行 structural hydration，新边
停留在粗图，不会变成可打包的原文证据。不过本轮 `graph_only_gold_hits` 没有提升，
说明当前收益主要来自极少量候选顺序变化，而不是稳定的新 gold path。下一轮不应先扩
beam，而应提高 typed edge coverage，并把 `typed edge -> descended fact -> packed turn`
做成逐题 trace 漏斗。

latency 来自不同运行，cache/OS 状态不能配对归因；本轮不把 latency 差值作为结论。

### 3.4 Gold-independent 构建与 gold-only oracle 诊断

构建、候选生成和查询均不使用 gold。gold 只在离线 audit 中判断关系图是否存在
理论可用路径：

| Edge subset | Overall session-pair recall/all-hit | LoCoMo multi-hop |
|---|---:|---:|
| entity/state mask | 41.67% / 40.5% | 14.67% / 10.0% |
| all relation-mask（含 scene） | 98.47% / 98.0% | 100% / 100% |

纯 scene 可达性几乎饱和，但没有足够判别力；稀疏 typed edge 更精确，却漏掉绝大多数
LoCoMo multi-hop 正确 session pair。这正是“路径很多但 evidence 不提升”的构建瓶颈。

## 4. 下一步强化顺序

### P0：先提高构建字段质量

1. **跨 session entity canonicalization。** 将 scene entities 显式投影到 fact/scene；
   使用 alias cluster + session-IDF；speaker、family、friend 等高频泛实体不得作为单独
   join key。
2. **Predicate family 与 state key。** 完整 phrase、canonical predicate、scope、
   polarity、value type 分开存储；state 边要求同 subject+predicate family，不能依赖
   通用 token overlap。
3. **时间双轴。** event time 与 observation time 独立；region 保存 interval min/max
   和 precision，time-near 只在同 entity/activity 下提议，不能按全局相邻时间连边。
4. **关系专属 parent summary。** 不再把所有 descendant 字段做普通 union；保存
   entity DF、predicate family histogram、time interval、collection manifest，避免
   mask 随层级升高而饱和。

### P1：关系通过离线门后再强化查询

1. 每种 relation 独立 beam quota，防止 scene-similar 挤掉 entity/temporal；只有具备
   entity/activity 条件的 temporal/collection 复合边可以占用 typed quota；
2. 将 structural descent 视为 region hydration phase，而不是与 cross-region hop
   竞争同一个 beam；
3. trace 记录 relation source、跨区命中和下钻转化，直接测
   `edge -> reached gold fact -> packed gold turn` 漏斗；
4. QueryIR 按 operand 分配关系预算，多 operand 不能全部沿同一 entity hub 扩张。

### P2：LLM 只处理真正模糊的候选

当前不应扩大 LLM relation refine。先达到以下 gate：

- LoCoMo multi-hop entity/state gold session-pair recall ≥ 50%；
- 每类 materialized edge direction-aware precision ≥ 85%；
- relation mask arm 的 32/48-turn all-hit 至少稳定 +2pp，precision 不回退。

通过后，再对 hard-reject 与 deterministic-accept 之间的窄区间做二阶段 LLM verifier。
这能避免在候选/属性缺失阶段用模型 token 判别大量本来就不包含正确关系的 pair。

## 5. 复现实物

- 最终图：`../artifacts/report/v5_14/relation_mask_final_dev200/report_graph.sqlite`
- 构图审计：`../artifacts/report/v5_14/relation_mask_final_dev200/build_audit/summary.json`
- 候选审计：`../artifacts/report/v5_14/relation_candidate_multiview_dev200/summary.json`
- 最终检索：`../artifacts/report/v5_14/accuracy_budget_relation_mask_final_dev200/summary.json`
- 配对比较：`../artifacts/report/v5_14/relation_mask_final_paired_vs_v5_13_safe.json`
