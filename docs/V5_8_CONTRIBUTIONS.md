# GraphMem V5.8 · 系统组成、实现细节与创新点清单

日期：2026-08-07
用途：正式技术报告的素材底稿。每个步骤链接到实现代码，每个数据点链接到原始证据。

**基线口径**：B1_quote 图 / 10 memory（5 LongMemEval + 5 LoCoMo）/ 761 道 gold 轮级标注题 /
证据预算 32 轮 / `all_hit` 为检索指标（该题全部 gold 轮次都进入证据包）。
所有 pp 差异均为**同一批题的逐题配对** + McNemar 精确检验。

**证据包**：`artifacts/v5_8/report_package/`，索引见其 `EVIDENCE_INDEX.md`。

---

# 第一部分 · 系统组成

系统分为**构建期**（离线，每个 memory 一次）与**查询期**（在线，每题一次）两条流水线，
之间通过一个 SQLite 图快照连接。**查询期不产生任何 LLM token。**

## 1. 构建期流水线

入口：[`GraphBuildPipeline.build()`](../src/graphmem/build/pipeline.py#L83)

### 1.1 场景切分

[`_segment()`](../src/graphmem/build/pipeline.py#L321) 把一个 session 的连续轮次切成 scene。
切分条件是三者的合取判断（[pipeline.py:340](../src/graphmem/build/pipeline.py#L340)）：
主题相似度低于 `topic_similarity_threshold`、且无实体重叠、且不是 QA 配对。
边界由 `min_turns` / `max_turns` 约束，尾部过短的 chunk 会回并到前一个
（[pipeline.py:347](../src/graphmem/build/pipeline.py#L347)）。

| 参数 | 位置 | v5_8_final |
|---|---|---|
| `min_turns` / `max_turns` | [config.py:100](../src/graphmem/config.py#L100) | 2 / **4** |
| `topic_similarity_threshold` | [config.py:102](../src/graphmem/config.py#L102) | 0.55 |
| `max_events_per_scene` | [config.py:103](../src/graphmem/config.py#L103) | 3 |

**作用**：scene 是抽取调用的输入单元，因此 `max_turns` 直接决定单次调用的输入规模，
是构建 token 的一级控制阀。实测本图 1,700 个 scene、平均 2.75 facts/scene。

### 1.2 LLM 语义抽取

[`QwenSemanticDistiller.extract()`](../src/graphmem/build/semantic.py#L182)，
严格模式走 [`_extract_strict_batch()`](../src/graphmem/build/semantic.py#L330)。

输出 schema 由 [`strict_scene_schema()`](../src/graphmem/build/semantic.py#L51) 生成，
用**引导解码**约束字段与长度。每条 fact 的字段：
`o` owner、`p` predicate、`v` value、`g` scope、`n` polarity、
`r` 引用的轮次序号、`q` 精确引文（`quote_evidence`）。
提示词为 [`STRICT_PROMPT`](../src/graphmem/build/semantic.py#L29)。

三个可选增强字段是**贡献 A1 的消融对象**：

| 字段 | 配置 | 说明 |
|---|---|---|
| `quote_evidence` | [config.py:64](../src/graphmem/config.py#L64) | 精确引文，占抽取输出约 26% |
| `semantic_scene_summary_chars` | [config.py:91](../src/graphmem/config.py#L91) | 场景摘要句，引导解码限长 |
| `semantic_scene_entities` | [config.py:95](../src/graphmem/config.py#L95) | 场景实体列表 |

⚠️ 实现注意：输入协议用 `s1` / `s1t0` 标注 scene 与 turn，模型会把这些标签复制进
任何自由文本字段。`scope` 自 V5.4 起过滤，V5.7 的类别字段泄漏率 2.9%，
V5.8 的场景摘要与实体列表泄漏率高达 68.5%
（[semantic.py:31 注释](../src/graphmem/build/semantic.py#L31)）。
守卫函数 [`strip_aliases()`](../src/graphmem/build/semantic.py#L39)。
**任何新增自由文本字段都需要这道守卫。**

### 1.3 构建预算账本

[`BuildTokenLedger`](../src/graphmem/build/budget.py#L30)。
每次调用前 [`reserve()`](../src/graphmem/build/budget.py#L60) 预留，
调用后 [`settle()`](../src/graphmem/build/budget.py#L88) 结算实际用量。

| 参数 | 位置 | 作用 |
|---|---|---|
| `semantic_max_tokens_per_memory` | [config.py:58](../src/graphmem/config.py#L58) | 每 memory 硬上限，0 = 不限 |
| `semantic_budget_degrade_at` | [config.py:60](../src/graphmem/config.py#L60) | 用到该比例时下调每场景事实上限 |
| `semantic_expected_output_tokens` | [config.py:82](../src/graphmem/config.py#L82) | 把"输出上限"与"预算记账"解耦 |

⚠️ 最后一项的设计缘由值得写进报告：早期版本用 `semantic_batch_output_tokens` 同时充当
输出上限与预算预留，导致为避免截断而把上限提到 32768 时，
**七次调用就耗尽 220K 预算**并使抽取退化到 fallback（0.28 facts/scene）。
解耦后上限成为失控保护、预算按实际开销计（实测约 600 token/次）。

### 1.4 规范化与分层聚合

- 谓词规范化：[`PredicateCanonicalizer`](../src/graphmem/build/canonicalize.py#L18)，
  按 `predicate_embedding_threshold`（[config.py:149](../src/graphmem/config.py#L149)）做向量聚类。
- 时间归一：[`normalize_time()`](../src/graphmem/build/temporal.py#L74)、
  [`observed_interval()`](../src/graphmem/build/temporal.py#L166)。
- 分层聚合：[pipeline.py:160](../src/graphmem/build/pipeline.py#L160) 按
  `coarsen.fanout`（[config.py:112](../src/graphmem/config.py#L112)）逐级聚合 routing_card。

**实测层级结构**：scene(level 0) 1,700 → routing_card level1 **367**（= session 数）
→ level2 **50** → level3 **10**（= memory 数）。

### 1.5 建边

[`_semantic_graph()`](../src/graphmem/build/pipeline.py#L495) /
[`_lean_semantic_graph()`](../src/graphmem/build/pipeline.py#L606)（g5 变体）。
度数上限由 `relation_degree_caps`（[config.py:161](../src/graphmem/config.py#L161)）逐关系约束。

**实测边分布**（21,117 条）：`scene_contains` 10,207、`has_fact` 3,997、
`participates_in` 3,377、`refines_to` 2,117、`at_time` 625、
`collection_co_member` 607、`state_next` 179、`temporal_before` 8。

可选的 LLM 精修：[`Qwen30BRefiner`](../src/graphmem/build/refine.py#L47)，
候选由 [`_ambiguous_candidates()`](../src/graphmem/build/pipeline.py#L1029) 按
跨会话共享实体名生成，交模型判 `SAME_EVENT / PORTAL / NONE`。
当前 `refine_mode="none"` 关闭。

---

## 2. 查询期流水线

入口：[`GraphNavigator.navigate()`](../src/graphmem/retrieval/navigator.py#L262)，
分派到 [`_navigate_legacy()`](../src/graphmem/retrieval/navigator.py#L280)（h0/h1）
或 [`_navigate_harness()`](../src/graphmem/retrieval/navigator.py#L324)（h2–h10）。

### 2.1 Query IR 编译

[`compile_query()`](../src/graphmem/retrieval/query_ir.py#L70 区域)，产出
[`QueryIR`](../src/graphmem/retrieval/query_ir.py#L35)：

- **算子**：[`_operator()`](../src/graphmem/retrieval/query_ir.py#L70) 判定
  `lookup / count_distinct / union_distinct / intersection_distinct / group_by_owner / exists_all` 等。
- **操作数**：[`_ast_operands()`](../src/graphmem/retrieval/query_ir.py#L173) 产出
  `OperandSpec`（owner 别名、谓词候选、scope 候选）。
- **AST**：[`compose_operator()`](../src/graphmem/retrieval/query_ir.py#L118) 组合算子树。
- **证明义务**：[`_ast_obligations()`](../src/graphmem/retrieval/query_ir.py#L194)。

全部为**确定性本地计算，零 LLM 调用**。
算子定义见 [operators.py](../src/graphmem/retrieval/operators.py#L18)（`FactSet` 到 `DateDifference` 共 12 种）。

### 2.2 多路种子

[`seed_operands()`](../src/graphmem/retrieval/seeding.py)，通道由
[`_ChannelRunner`](../src/graphmem/retrieval/seeding.py#L148) 执行：
[`exact()`](../src/graphmem/retrieval/seeding.py#L156)、
[`bm25()`](../src/graphmem/retrieval/seeding.py#L173)、
[`dense()`](../src/graphmem/retrieval/seeding.py#L177)。
多视角由 [`build_views()`](../src/graphmem/retrieval/seeding.py#L95) 按操作数展开，
融合用 [`_rrf()`](../src/graphmem/retrieval/seeding.py#L83)。

深度常量（[seeding.py:28–37](../src/graphmem/retrieval/seeding.py#L28)）：
`LEGACY_BM25_DEPTH=96`、`LEGACY_DENSE_DEPTH=96`、`LEGACY_SESSION_FANOUT=8`、
`VIEW_BM25_DEPTH=48`、`VIEW_DENSE_DEPTH=48`。

⚠️ **`CHANNELS` 全部是轮级通道**（[seeding.py:40](../src/graphmem/retrieval/seeding.py#L40)），
注释写明「Typed postings retrieve graph nodes, not turns」——
**不存在自顶向下的路由卡入口**，367 张会话路由卡从未被用作检索起点。

### 2.3 图遍历

[`scheduler.execute()`](../src/graphmem/retrieval/scheduler.py#L49)。
BFS + 目的地优先级排序（`destination_priority`），
关系白名单 [`DEFAULT_PREFERRED`](../src/graphmem/retrieval/scheduler.py#L27)。

本轮新增两项（**贡献 B2**）：
- `expansion_beam`：入队**前**按优先级只保留每次展开的前 N 个邻居。
- `node_hops` 返回跳数，供打分侧做距离衰减。

### 2.4 事实蓄水池与绑定

- 蓄水池：[`build_fact_reservoir()`](../src/graphmem/retrieval/facts.py#L154)，
  通道 `source_projection / structured / lexical / dense`
  （[facts.py:46](../src/graphmem/retrieval/facts.py#L46)），
  容量常量 [facts.py:25–36](../src/graphmem/retrieval/facts.py#L25)。
  收敛：[`select_active_facts()`](../src/graphmem/retrieval/facts.py#L319)。
- 绑定：[`evaluate_binding()`](../src/graphmem/retrieval/bindings.py#L117) +
  [`accepts()`](../src/graphmem/retrieval/bindings.py#L173)，
  权重 [`WEIGHTS`](../src/graphmem/retrieval/bindings.py#L36)（owner 0.40 / predicate 0.28 / …），
  接受阈值 `ACCEPT_THRESHOLD=0.30`。

### 2.5 算子求值与证书

- AST 求值：[`evaluate_ast()`](../src/graphmem/retrieval/ast_algebra.py#L72)。
- 证据证书：[`evaluate_certificate()`](../src/graphmem/retrieval/certificate.py#L8)，
  检查问题槽位是否被证据覆盖。
- 闭式作答：[`compose()`](../src/graphmem/answer/composer.py#L38)。

### 2.6 证据打包

两种打包器，**贡献 C5 的对比对象**：

| 打包器 | 位置 | 策略 |
|---|---|---|
| `_rank_pack` | [navigator.py:1073 区域](../src/graphmem/retrieval/navigator.py#L1073) | 一次排序取前 N |
| **`_set_cover`** | [navigator.py:1038 区域](../src/graphmem/retrieval/navigator.py#L1038) | 边际效用贪心 |
| `pack_proof_units` | [packer.py:23](../src/graphmem/retrieval/packer.py#L23) | 按证明单元打包（h6/h8/h9） |

`_set_cover` 每步选 `fused + 1.25×槽位覆盖增益 + 会话多样性 − 0.0005×token` 最大者，
多样性项**仅在该轮次的会话尚未进入证据包时给出**（count/list 类题 0.9，其余 0.2）。

融合打分在 [navigator.py:601 区域](../src/graphmem/retrieval/navigator.py#L601)，
权重现已参数化为 [`FUSION_DEFAULTS`](../src/graphmem/retrieval/navigator.py#L88 区域)。

### 2.7 作答

[`AnswerStage.answer()`](../src/graphmem/answer/stage.py#L131)，
证据渲染 [`render()`](../src/graphmem/answer/stage.py#L108)，
配置 [`AnswerConfig`](../src/graphmem/answer/rendering.py#L27)。
**这是查询期唯一的 LLM 调用。**

---

# 第二部分 · 创新点（按三个系统目标归类）

## 目标 A · 降低单 memory 构建 token

### A1. 抽取字段的 Pareto 前沿：只有 quote 该保留

**作用位置**：§1.2 语义抽取的三个增强字段。

**实验**：单因子消融，761 题。
数据：[`PhaseA_build_arms.json`](../../artifacts/v5_8/report_package/evidence/raw/PhaseA_build_arms.json)（token）、
[`PhaseA_arm_recall.json`](../../artifacts/v5_8/report_package/evidence/raw/PhaseA_arm_recall.json) +
[`PhaseA_arm_recall_2.json`](../../artifacts/v5_8/report_package/evidence/raw/PhaseA_arm_recall_2.json)（all_hit）。

| 臂 | 构建 token/memory | all_hit | 相对 B0 |
|---|---:|---:|---:|
| B0_core（全关） | 126,482 | 0.512 | — |
| **B1_quote** | **167,175** | **0.572** | **+6.0pp** |
| B2_summary | 143,656 | 0.493 | −1.9pp |
| B3_entities | 142,074 | 0.482 | −3.0pp |
| B4_all（全开） | 270,187 | 0.549 | +3.7pp |
| B5_all_free_p | 234,147 | 0.560 | +4.8pp |

**两个反直觉结论**：
1. 三个字段里**只有 `quote_evidence` 有正收益**；摘要与实体列表**单独都是负收益**。
2. **三者叠加（0.549）低于只开 quote（0.572）**——它们互相干扰。

**B1 相对 B4：token 少 38%，精度高 2.3pp——严格 Pareto 改进。**

**可写的论点**：抽取阶段"输出更多结构"不单调改善检索；
必须逐字段消融而非整体开关。这直接指导"该让 LLM 抽取什么"。

### A2. 每 memory 的硬预算账本与上限/记账解耦

**作用位置**：§1.3。

把构建成本从事后统计量变成**可控上界**；并通过
`semantic_expected_output_tokens` 把"单次输出上限"与"预算预留"解耦，
避免提高上限以防截断时反而耗尽预算（见 §1.3 的失效案例）。

⚠️ 机制已实现并在用，但**缺少开/关对照实验**。
报告中应作为**设计描述**而非已验证贡献。

---

## 目标 B · 降低召回延迟

### B1. 查询路径完全确定性、零 LLM 调用

**作用位置**：§2.1–§2.6 全部。

Query IR 编译、多路种子、图遍历、事实绑定、算子求值、证据打包**全部为本地确定性计算**。
实测 p50 **35–184ms**（随预算与 profile 变化），数据见各
[`*_full.json`](../../artifacts/v5_8/report_package/evidence/raw/) 的 `latency_ms` 字段。

**派生优势**：检索结果完全确定，**一遍即全量测量、无采样方差**——
这也是本轮所有配对检验能用精确 McNemar 而非重复实验的原因。

### B2. 遍历饱和点的量化方法

**作用位置**：§2.3 图遍历的预算设定。

数据：[`hops.log`](../../artifacts/v5_8/report_package/evidence/logs/hops.log)、
[`beam.log`](../../artifacts/v5_8/report_package/evidence/logs/beam.log)。

| hops / nodes | 候选池 gold 覆盖 | 走边/题 | 访问节点/题 |
|---|---:|---:|---:|
| **2 / 96（当前）** | **94.95%** | 32.6 | 95.8 |
| 3 / 256 | 94.95%（**零增益**） | 192.2 | 255.3 |
| 4 / 512 | 95.67% | 358.6 | 421.7 |
| 6 / 2048 | 95.76%（**结构上界**） | 360.7 | 423.8 |

| beam | all_hit | gold 仅图可达 | 走边/题 |
|---|---:|---:|---:|
| 2 | 0.5834 | 94 | **36.8** |
| 4 | 0.5795 | 95 | 42.4 |
| 8 | 0.5808 | 94 | 43.8 |
| **16 / 32 / 无限制** | 0.5795 | 94 | 44.0 |

**三条结论**：
1. **hop=2 已达结构可达上界的 99.2%**；hop=3 零增益却多走 6 倍的边。
2. **beam=16 / 32 / 无限制逐位完全相同**——遍历是**种子受限**而非预算受限。
3. 走边仅用掉边预算的 **23%**（44/192）；剪枝 `beam=2` 走边 −16% 而图独有 gold 一个不丢（94→94）。

**可写的论点**：图遍历预算应由**测出的饱和点**决定，而不是调参。
这为"为什么可以做到几十毫秒"提供了因果依据。

⚠️ **更正**：早先一次串行测量报出剪枝带来 mean −53% / p95 −76%，
**经反序复测证伪**——进程内第一个跑的臂要付页缓存与首次 SQLite 访问的预热
（[`rev.log`](../../artifacts/v5_8/report_package/evidence/logs/rev.log)：
同一臂在位置1 为 154.2ms、位置3 为 115.3ms，而未剪枝基线在位置2 为 116.1ms）。
**剪枝可主张"省边 16%"，不可主张"省时间"。**

---

## 目标 C · 提高检索精度

### C1. 跨会话指称先验及其适用边界

**作用位置**：候选会话集的构造（当前作为独立先验，尚未并入 §2.2 种子）。

**发现**：gold 会话对共享的低文档频率词远多于随机会话对（LongMemEval，324 题）。
定义：`df ≤ max(2, 5%·会话数)` 的词记为稀有词。

| 类别 | gold 对 | 随机对 | 分离度 | gold 落在前 5% |
|---|---:|---:|---:|---:|
| multi-session | 15.1 | 1.1 | **13.9x** | 85.0% |
| temporal-reasoning | 22.6 | 1.1 | **20.7x** | 85.8% |
| knowledge-update | 20.4 | 1.1 | **19.0x** | 96.2% |

**作为候选过滤器的横向对照**（同一目标：找出 gold 需要的会话对）：

| 机制 | 召回 | 候选空间 | LLM 成本 |
|---|---:|---:|---|
| 精确实体名 | 0.6% | 29 个候选 | 需要 |
| 场景摘要 embedding | 98.8% | **35%** 的全部会话对 | 不需要 |
| **稀有词 K≥3** | **92.3%** | **12%**（130 边/memory） | **零** |

实现参考：[`build_shared_referent_edges.py`](../scripts/build_shared_referent_edges.py)。

⚠️ **适用边界（本身即发现）**：该信号在 LoCoMo 上**不成立**
（[`rare2.log`](../../artifacts/v5_8/report_package/evidence/logs/rare2.log)：
分离度 1.3x，gold 进前 5% 仅 19.9%）。
成因是持续性双人对话的词汇跨会话高度复用，稀有词库缩小约 5.7 倍
（LoCoMo 约 1,000 vs LME 约 5,700）。
**这给出了"何时可用词法跨会话先验"的判据**，是有解释力的边界条件而非缺陷。

### C2. 图层提供词法/稠密通道不可达的证据

**作用位置**：§2.3 遍历 + §2.4 蓄水池对候选池的贡献。

数据：[`graphval.log`](../../artifacts/v5_8/report_package/evidence/logs/graphval.log)、
[`reach.log`](../../artifacts/v5_8/report_package/evidence/logs/reach.log)。

| 证据预算 | 纯词法/dense | +图层 | 净贡献 |
|---:|---:|---:|---:|
| 16 | 0.5861 | 0.5926 | **+0.66pp** |
| 32 | 0.6623 | 0.6715 | **+0.92pp** |
| 64 | 0.7346 | 0.7451 | **+1.05pp** |

**1,108 条 gold 轮次中有 95 条（8.57%）仅靠图可达**——
任何词法或稠密通道都到不了。净贡献**随预算单调上升**，
说明图捞回的内容在预算宽裕时更能兑现。

### C3. 集合覆盖式证据打包（本轮最大单项收益）

**作用位置**：§2.6 打包器。

数据：[`F_setcover_isolation.json`](../../artifacts/v5_8/report_package/evidence/raw/F_setcover_isolation.json)。

| 臂 | all_hit | vs 基线 | 配对 |
|---|---:|---:|---|
| h10 + `_rank_pack` | 0.5848 | — | |
| **h10 + `_set_cover`** | **0.6570** | **+7.22pp** | 90胜35负 **p=9.2e-07** |
| h10 + `_set_cover` + 会话路由 top-8 | **0.6978** | +11.30pp | 111胜25负 **p=3.9e-14** |
| h0（legacy 参照） | 0.7004 | +11.56pp | |
| h10 + `_set_cover` @预算64 | 0.7411 | | |

**可写的论点**：在固定预算下，**如何花预算**比**捞回多少**更重要。
`_set_cover` 的会话多样性项使证据包不会被单一会话占满，
这正是多会话题所需的结构。

**同时澄清一个此前的误判**：h0 相对 h10 的 13pp 曾被归因为"harness/关系代数有害"。
实测 h0 的候选池只有 **297** 轮、**小于** h10 的 **420** 轮
（[`fair.log`](../../artifacts/v5_8/report_package/evidence/logs/fair.log)），
而 h10 装上 `_set_cover` + 会话路由后达到 0.6978 ≈ h0 的 0.7004。
**13pp 全部来自打包器与会话路由，与 Query IR、关系代数无关。**

### C4. Query IR 的净贡献（隔离测量）

**作用位置**：§2.1 编译 → §2.2 操作数种子 → §2.4 绑定 → §2.5 算子。

**实验设计**：打包器（`_set_cover`）与候选池构造（会话路由 top-8）
在**所有臂上完全相同**，唯一变化的是编译后的问题有多少送进流水线。
这是第一次对 Query IR 的干净消融。
数据：[`Q_queryir_isolation.json`](../../artifacts/v5_8/report_package/evidence/raw/Q_queryir_isolation.json)。

| 臂 | 移除的通路 | all_hit | vs 完整 IR | 配对 |
|---|---|---:|---:|---|
| Q0 | — | 0.6978 | — | |
| Q1 | operand 计数项 | 0.6938 | −0.39pp | 7胜10负 p=0.63 |
| **Q2** | **binding 打分项** | 0.6794 | **−1.84pp** | 7胜21负 **p=0.0125** |
| Q3 | `structured` fact 通道 | 0.6978 | **±0.00pp** | **0胜0负** |
| **Q4** | **以上三者** | 0.6636 | **−3.42pp** | 13胜39负 **p=0.00041** |
| Q5 | 降到 h5（无蓄水池/AST） | 0.5900 | **−10.78pp** | 19胜101负 p=1.1e-14 |
| Q6 | 降到 h4（仅调度器） | 0.5940 | −10.38pp | p=5.1e-14 |
| Q7 | 降到 h2（仅倒排） | 0.6176 | −8.02pp | p=5.5e-11 |

**四条结论**：
1. **Query IR 的信号净值 +3.42pp**，其中**事实绑定单项 +1.84pp**。
2. **fact 蓄水池 + AST 两级在良好打包器下值 +10.78pp**。
3. **两个可直接删除的零贡献组件**：`structured` fact 通道**恰好为零**
   （761 题 0胜0负）、`operand` 计数项不显著。删除可省算力而不损精度。
4. **如实说明**：h10 全栈（0.6978）**仅追平** h0（0.7004）；
   IR 机制赚回的约 10pp，补上的是 harness 架构在别处丢失的部分。

---

# 第三部分 · 方法论发现

## M1. Oracle 分解：一种误差归因方法

**方法**：把候选池限制到 gold 所在单元，用**同一个打分器**重新打包，
即可把误差干净地拆成「够不着 / 选不中 / 排不上」三段。
数据：[`oracle.log`](../../artifacts/v5_8/report_package/evidence/logs/oracle.log)、
[`split.log`](../../artifacts/v5_8/report_package/evidence/logs/split.log)。

```
候选池 gold 覆盖 94.95%（结构上界 95.76%）→ 打包后保留 62.9%
all_hit 0.5848 → oracle 会话路由 0.9474
  单会话题 (78.7%, n=599)  实际 0.6895   oracle 1.0000
  多会话题 (21.3%, n=162)  实际 0.1975   oracle 0.7531
```

**核心论断**：
**完美选择用 32 轮预算（0.9474）优于暴力打包用 256 轮预算（0.9040）**
（[`tok.log`](../../artifacts/v5_8/report_package/evidence/logs/tok.log)）。
八分之一预算、更高命中——**"检索值得做"的直接量化证明**，
也说明系统目标应是逼近 oracle，而非扩张预算。

**附带诊断**：多会话题即使 oracle 路由也只有 0.7531，
因为 2 个 gold 会话约 62 轮超过 32 的预算——这是**证据密度**问题而非选择问题。

## M2. 下游瓶颈会系统性低估上游模块的价值

**同一组 Query IR 信号，在两种打包器下测得的贡献相差三倍：**

| 打包器 | 去掉 IR 信号的代价 | 证据 |
|---|---:|---|
| `_rank_pack` | −1.18pp | [`W_fusion_weights.json`](../../artifacts/v5_8/report_package/evidence/raw/W_fusion_weights.json)（W8） |
| `_set_cover` | **−3.42pp** | [`Q_queryir_isolation.json`](../../artifacts/v5_8/report_package/evidence/raw/Q_queryir_isolation.json)（Q4） |

**同一批 harness 机制，换打包器后从"最差"变为"最有价值"：**

| 打包器 | h8/h9 相对 h0 | h10 相对 h5 |
|---|---:|---:|
| `_rank_pack` | **−28pp** | +6.96pp |
| `_set_cover` | 追平 | **+10.78pp** |

**推论：流水线各阶段的消融结果不可加，且必须在下游已优化的前提下测量，
否则会得出「上游模块无用」的错误结论。**

本轮的实际教训：先测得「harness/关系代数有害（−13pp）」，
经三次归因修正后确认那是**打包器与会话路由**所致。
该结论由两组独立实测交叉验证。

## M3. 结构做过滤、问题做排序

四个**问题无关**的结构信号，全部输给一个**问题条件**的词法分：

| 信号 | 端到端收益 | 证据 |
|---|---:|---|
| 实体合并 | +0.2pp | [`EntityMerge_recall.json`](../../artifacts/v5_8/report_package/evidence/raw/EntityMerge_recall.json) |
| `shared_referent` 跨会话边（13,386 条） | +0.13pp | [`AB_shared_referent.json`](../../artifacts/v5_8/report_package/evidence/raw/AB_shared_referent.json) |
| 摘要 embedding 配对 | 高阈值反而丢 gold | 见 C1 对照表 |
| 稀有词同层扩展 | 两子集三阈值全输平铺 | [`expand2.log`](../../artifacts/v5_8/report_package/evidence/logs/expand2.log) |

同一个稀有词信号：**作过滤器 92.3% 召回，作排序器失效**
（从锚点约 10 个邻居取 top-2）。

**原则：预计算的结构应当定义候选集；排序必须由问题决定。**

## M4. `all_hit` 在预算变化时不是有效目标函数

预算 32→256 时 all_hit 从 0.5848 升到 0.9040，近乎机械上升
（256 轮已覆盖一个 memory 的约一半）。
**跨预算的系统比较必须使用判分**，仅固定预算下的比较有效。
数据：[`B_budget_scaling.json`](../../artifacts/v5_8/report_package/evidence/raw/B_budget_scaling.json)。

---

# 第四部分 · 当前判定与未完成项

## 关系代数的当前判定

**Query IR 已由 C4 判定为正贡献**（+3.42pp 信号、+10.78pp 机制层级）。

关系代数的**闭式算子路径**（[`CLOSED_FORM_KINDS`](../src/graphmem/retrieval/operators.py#L120) /
[`compose()`](../src/graphmem/answer/composer.py#L38)）目前在约 0 道题上触发，
因为 `collection_key` 已退化为 5 个 value_type（text 3,959 / number 398 / time 210 /
currency 58 / boolean 44）、`predicate` 92.1% 唯一——**缺的是可 join 的键，不是机制**。

注意：AST 执行层本身（h10 相对 h9）仍在 C4 的 +10.78pp 之内，
**真正未被启用的只有依赖分组键的闭式求解那一小部分**。

封闭枚举的属性类验证（[`enum.log`](../../artifacts/v5_8/report_package/evidence/logs/enum.log)）：

| 键 | 键数 | 单例 | 跨≥2会话 | 可数集合 |
|---|---:|---:|---:|---:|
| 基线 (owner, collection_key) | 1,274 | 75.6% | 5.0% | 311 |
| **枚举 (owner, class)** | 1,518 | 67.3% | **12.9%** | **497** |

跨会话复用 **5.0% → 12.9%（2.6 倍）**，对照 V5.7 模型自造类别的 6.3% 复用——
**封闭词表的机制成立**。但 54.6% 落入 `other`，说明临时拟定的 52 类不适配语料，
需从语料导出的分类法才能定论。**此结果为下界。**

## 尚未完成、影响结论强度的实验

| # | 缺口 | 影响 |
|---|---|---|
| 1 | **端到端判分** | 全部结论均为 `all_hit`；与判分的映射未验证 |
| 2 | **110-memory 构建** | `lme_multi_session` 全程 n=1，分层结论实质只在 LoCoMo 上成立 |
| 3 | **方差估计** | 单图单种子；运行间抖动约 1 题，而多个效应量为 3–10 题 |
| 4 | 语料导出的属性类词表 | 决定闭式算子能否启用 |
| 5 | 跨会话边口径核对 | 本轮测 0 条，[`V5_8_CROSS_SESSION_KEY.md`](V5_8_CROSS_SESSION_KEY.md) 报 124 条 |

## 撰写时必须交代的口径

1. 「gold 100% 在候选池」与「94.95%」是两个口径（前者含会话扇出带进的零分轮次），**以 94.95% 为准**。
2. cat4 占 55% 且从不需要第二会话，跨会话工作只在 cat1 的 142 题（18.7%）上计分。
3. **进程内首臂的延迟数不可用**（预热假象）；并行跑的延迟被 embedding 排队污染。
4. 运行间抖动约 1 题；多个效应量为 3–10 题。
5. 跨预算的 `all_hit` 比较无效，仅固定预算下有效。
