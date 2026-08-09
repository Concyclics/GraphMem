# GraphMem V5.8 消融实验：方法、代价与下一步

> 数据口径：Phase A / Phase B 固定 761 题；其中 LongMemEval multi-session 只有 1 题，当前结论主要反映 LoCoMo。检索 all-hit 不等于最终 answer/judge accuracy。

## 核心结论

1. **构建侧应保留原文证据引用。** 核心事实抽取加入 quote/span provenance 后，turn all-hit 从 51.2% 提升到 57.2%，增益 6.0pp。
2. **自由场景摘要和场景实体目前没有价值。** 单独加入分别回退 1.9pp 和 3.0pp；三项全开仍不如只开证据引用。
3. **总体召回暂时仍以 Legacy N5 集合覆盖为主。** 32 turns 下达到 70.0% all-hit，优于完整 AST Graph Harness@64 的 62.7%。
4. **Graph Harness 是专用多跳路径，不是当前通用路径。** 它在 LoCoMo Cat1 比 Legacy N5 高 9.2pp，但在 Cat4 低 13.4pp。
5. **事实蓄水池是当前最大退化点。** 引入宽 fact reservoir 后 all-hit 从约 70% 降到 41.7%，应优先定位候选准入、active-fact 裁剪、binding 和 packing 中的首个损失阶段。

## 双阶段实现流程

```text
图构建（T2）
Raw turns
  → Scene segmentation
  → Qwen semantic fact extraction
  → CanonicalFact / Entity / Time / State
  → Hierarchy + typed relations
  → Authority SQLite

确定性投影（T1）
Authority SQLite
  → CollectionManifest / value lattice
  → temporal closure / dialogue pairs / fact spans
  → QueryReadView

在线召回与作答（T0）
Question
  → exact / BM25 / dense seeds
  → raw-turn ranking 或 FactBinding + graph algebra
  → evidence certificate / set cover
  → evidence packing
  → answer rendering
```

`GraphMemV5Config` 控制构建阶段，任何字段变化都会改变 config hash 并触发重建。`ProjectionConfig` 和 `AnswerConfig` 可以复用同一抽取图。纯召回参数应该迁入独立 `RetrievalConfig`；当前 `query_budget` 放在构建配置中会错误地作废构建缓存。

## 方法实现与差别

| 方法 | 阶段 | 大致实现流程 | 相对差别 | 当前结果 |
|---|---|---|---|---:|
| 核心事实抽取（B0） | 构建 | Scene → LLM 抽 owner/predicate/value/scope/time → CanonicalFact | 最小语义结构，不要求 quote | 51.2% |
| 带原文证据的事实抽取（B1） | 构建 | 核心事实 + quote/span → terminal provenance | 每条事实能直接回溯 SourceTurn | **57.2%** |
| 自由场景摘要扩展（B2） | 构建 | 核心事实 + LLM scene summary | 增加非槽位化的第二套表达 | 49.3% |
| 场景实体扩展（B3） | 构建 | 核心事实 + scene entity set | 与 fact owner/participants 重复且可能冲突 | 48.2% |
| 引用、摘要、实体全开（B4） | 构建 | 同时生成 quote、summary、entities | 输出更胖，多个语义视图互相干扰 | 54.9% |
| 全开并保留自由谓词（B5） | 构建 | B4 + 不截断 predicate | 减少谓词压缩损失 | 56.0% |
| 原始 turn 三路融合 | 召回 | exact + BM25 + dense → session fanout → ranked turns | 不依赖结构化事实闭包 | 69.0% |
| Legacy N5 集合覆盖 | 召回 | 融合候选 → gain/token set-cover → evidence pack | 选择互补原始 turns | **70.0%** |
| 事实绑定 + 关系代数 | 召回 | Operand seeds → owner/predicate/scope binding → union/count/time algebra | 用查询义务和 witness 替代 turn 排序 | 50.2% |
| 宽事实蓄水池 | 召回 | 多来源 facts → reservoir → per-operand active facts | 加宽 ID 候选后再结构化裁剪 | 41.7% |
| 结构化事实激活 | 召回 | reservoir → structured fact activation → proof packing | 更依赖 fact schema 和 binding | 41.7% |
| 查询 AST Graph Harness | 召回 | QueryIR/AST → operands → binding → algebra → proof units | 显式支持多跳、集合和时间运算 | 57.2–62.7% |
| P8/P9 确定性投影 | 投影 | predicate head-stem / relaxed scope → manifest / value lattice | 不调用 LLM，只改变关系投影 | 待组合验证 |

## 构建消融结果

| 实现方式 | All-hit | Recall | 相对核心基线 |
|---|---:|---:|---:|
| 核心事实抽取 | 51.2% | 53.2% | — |
| **核心事实 + 原文证据引用** | **57.2%** | **57.2%** | **+6.0pp** |
| 核心事实 + 自由场景摘要 | 49.3% | 50.6% | -1.9pp |
| 核心事实 + 场景实体列表 | 48.2% | 50.4% | -3.0pp |
| 引用 + 摘要 + 实体全开 | 54.9% | 55.6% | +3.7pp |
| 全开 + 自由谓词 | 56.0% | 56.2% | +4.8pp |

```text
Turn all-hit
核心事实 + evidence quote  ████████████████████████████ 57.2%
全开 + 自由谓词            ███████████████████████████  56.0%
引用 + 摘要 + 实体全开      ██████████████████████████   54.9%
核心事实抽取               ████████████████████████     51.2%
核心事实 + scene summary   ███████████████████████      49.3%
核心事实 + scene entities  ██████████████████████       48.2%
```

结论不是“抽取得越多越好”。quote/span 增强事实与原文的可追溯性；summary 和 entities 会增加输出 token，并产生与 fact slots 并行的语义表达，反而干扰索引与召回。

## 构建 Token 开支

Token 口径为 Phase A 每个 memory 的构建 backbone input+output token；embedding token 单独记录，不计入生成式 LLM 开支。召回消融复用固定 SQLite，**在线生成式 LLM token 为 0**。

| 构建方法 | 平均 token / memory | 最大 token / memory | 相对核心基线 | All-hit | 每新增 10k token 的增益 |
|---|---:|---:|---:|---:|---:|
| 核心事实抽取（B0） | 126.5k | 177.0k | — | 51.2% | — |
| **核心事实 + 原文证据引用（B1）** | **167.2k** | 296.1k | +40.7k / +32% | **57.2%** | **+1.47pp** |
| 核心事实 + 自由场景摘要（B2） | 143.7k | 250.2k | +17.2k / +14% | 49.3% | 负收益 |
| 核心事实 + 场景实体（B3） | 142.1k | 246.9k | +15.6k / +12% | 48.2% | 负收益 |
| Quote + summary + entities（B4） | 270.2k | 418.7k | +143.7k / +114% | 54.9% | +0.26pp |
| B4 + 自由谓词（B5） | 234.1k | 310.1k | +107.7k / +85% | 56.0% | +0.45pp |

```text
平均构建 token / memory
B4 全特征             ███████████████████████████ 270.2k
B5 全特征 + 自由谓词   ███████████████████████     234.1k
B1 evidence quote     █████████████████           167.2k
B2 scene summary      ██████████████              143.7k
B3 scene entities     ██████████████              142.1k
B0 core facts         █████████████               126.5k
```

Token 主要花在语义抽取的重复输入和扩展输出字段：quote 会复制必要证据文本；summary 和 entities 会在已有 fact slots 之外再次表达同一 scene；多特征同时开启还会增加输出长度、截断与重试概率。B4 的平均成本超过 B0 两倍，最大单 memory 达 418.7k，也超过当前 `semantic_max_tokens_per_memory=300k` 预算。

因此推荐 B1 不是因为它最便宜，而是它的 **质量/Token 边际收益最高**。相比 B0，平均多花 40.7k token 换取 6.0pp all-hit；summary、entities 和全特征组合应删除。进一步压缩应优先减少重复 scene header、使用 turn alias、只输出 evidence offsets 而非长 quote，并复用同一事实抽取结果，不能牺牲 provenance。

## 召回质量与延迟

| 召回流程 | Evidence turns | All-hit | Recall | p50 | 实现重点 |
|---|---:|---:|---:|---:|---|
| 原始 turn 三路融合 | 32 | 69.0% | 64.9% | **40.8ms** | exact/BM25/dense 直接排 turn |
| **Legacy N5 集合覆盖** | 32 | **70.0%** | **65.8%** | 57.6ms | turn-level set-cover |
| 显式 N5 集合覆盖 | 32 | 70.0% | 65.8% | 58.4ms | 与 H0 完全一致，parity 通过 |
| 事实绑定 + 关系代数 | 32 | 50.2% | 51.7% | 60.9ms | binding + algebra + certificate |
| 宽事实蓄水池 | 32 | 41.7% | 42.6% | 128.1ms | 多来源 fact reservoir |
| 结构化事实激活 | 32 | 41.7% | 43.0% | 142.8ms | per-operand active facts |
| AST Graph Harness | 32 | 57.2% | 57.2% | 156.7ms | QueryIR + proof units |
| AST Graph Harness | 48 | 60.3% | 61.6% | 156.9ms | 增大 pack budget |
| AST Graph Harness | 64 | 62.7% | 64.7% | 158.7ms | 增大 pack budget |

```text
Overall all-hit
Legacy N5 set-cover       ███████████████████████████████████ 70.0%
Raw turn fusion           ██████████████████████████████████  69.0%
AST Harness @64           ███████████████████████████████     62.7%
AST Harness @48           ██████████████████████████████      60.3%
AST Harness @32           ████████████████████████████        57.2%
Fact binding + algebra    █████████████████████████           50.2%
Wide fact reservoir       ████████████████████                41.7%
Structured fact activation████████████████████                41.7%
```

Harness 的额外阶段同时增加了延迟：AST p50 约为 Legacy N5 的 2.7 倍。该延迟来自九路并发共享 embedding 服务，适合做相对比较，不能直接用于容量规划。

## Evidence budget 是免费且单调的杠杆

| AST Harness budget | All-hit | Recall | p50 |
|---:|---:|---:|---:|
| 32 turns | 57.2% | 57.2% | 156.7ms |
| 48 turns | 60.3% | 61.6% | 156.9ms |
| 64 turns | 62.7% | 64.7% | 158.7ms |

32→64 turns 带来 5.5pp all-hit，检索延迟几乎不变。但新增成本落在下游 answer context，而且 64 turns 仍比 Legacy N5@32 低 7.4pp。下一步必须在胜出的 Legacy N5 上重复 32/48/64 扫描。

## Graph Harness 的题型优势集中在 Cat1

| LoCoMo 类型 | N | Legacy N5 · 32 | AST Harness · 64 | AST 相对差异 |
|---|---:|---:|---:|---:|
| Cat1 多跳 | 142 | 21.8% | **31.0%** | **+9.2pp** |
| Cat2 时序 | 156 | **85.3%** | 76.3% | -9.0pp |
| Cat3 开放域 | 44 | 31.8% | **34.1%** | +2.3pp |
| Cat4 单跳 | 418 | **84.7%** | 71.3% | -13.4pp |

AST/algebra 的集合闭包和独立 witness 机制确实适合 Cat1；Cat4 只需要直接事实命中，额外 binding 和候选裁剪反而有害。由于 Cat4 题量约为 Cat1 的三倍，总体指标会掩盖 Cat1 增益。

## 实验代价分层

| 层级 | 包含内容 | 单臂代价 | 是否复用图 | 正确用途 |
|---|---|---|---|---|
| T0 在线免费层 | QueryBudget、召回 profile、融合/绑定权重、AnswerConfig | 分钟级，零构建 LLM | 完全复用 | 先定位 seed、binding、packing 和 budget 问题 |
| T1 确定性投影层 | Manifest、value lattice、temporal closure、predicate normalization | 投影重放 | 复用抽取 facts | 验证集合、时间和 span 索引结构 |
| T2 LLM 重建层 | Scene、semantic extraction、hierarchy、entity、edges | 约 2 小时/110 memories | 不复用 | 只验证已有 T0/T1 诊断支持的构建假设 |

## 下一轮实验矩阵

| 优先级 | 实验 | 具体实现 | 要回答的问题 | 通过门槛 |
|---|---|---|---|---|
| P0 | 最终答案价值验证 | Legacy N5@64 与 AST Harness@64 跑 answer + judge | AST closed-form 能否抵消较低 turn all-hit？ | 以 answer/judge 判定 |
| P0 | Legacy budget 扫描 | N5 固定，evidence turns 32/48/64 | 免费增益能否迁移到总体最佳路径？ | 单调提升且 context 不超限 |
| P1 | Fact reservoir 退化诊断 | 记录 admission→active facts→binding→candidate→packed | 28pp 首先丢在哪个阶段？ | 找到首个可复现损失点 |
| P1 | 独立 RetrievalConfig | 迁出 seed depth、binding、fusion、QueryBudget | 是否能复用同一图系统扫描召回？ | 默认值逐题 parity |
| P1 | W/S 参数扫描 | exact 1.2→1.0；graph 0.8→1.2；binding 0.30→0.20；view depth 48→96 | Harness 是权重失配还是结构有损？ | 保留 Cat1，缩小 Cat2/Cat4 回退 |
| P2 | 无标签题型路由 | 仅由问题文本编译 operator/complexity | 能否组合 N5 与 AST 优势？ | held-out 提升且不读 category |
| P2 | P8/P9 投影 | head-stem、relaxed scope、value lattice | 更好的 manifest 能否减少宽 reservoir 依赖？ | 集合提升、单跳不退化 |
| P3 | 有限 T2 重建 | C1 只开 quote；C2 scene 4→6 turns | 是否必须协同改变图结构？ | 先有 T0/T1 证据支持 |

## 执行前必须修复

1. 将硬编码 seed depth、binding threshold、fusion weights 和 `QueryBudget` 迁入独立 `RetrievalConfig`。
2. `A5_free_predicate.json` 与 `A6_out3072.json` 引用了已删除字段，应修复或明确标记作废。
3. 不再测试 `max_llm_reranks` 和 `max_iterations`：当前代码中它们不控制实际行为。
4. 补齐 50 道 LongMemEval multi-session 和 50 道 temporal 的构建/召回测量，当前 LME 样本量不足。

## 推荐决策

- 默认构建：采用 **B1——核心事实 + quote/span provenance**，关闭自由 scene summary 与 scene entities。
- 默认召回：在最终 judge 完成前保留 **Legacy N5 set-cover**。
- 实验路径：保留 **AST Graph Harness** 作为 Cat1/multi-hop 专用候选，不继续用总体 all-hit 单独裁决它的价值。
- 下一步代码：先做 `RetrievalConfig` 与 reservoir 分阶段 telemetry，不立即启动新的全量 T2 重建。
