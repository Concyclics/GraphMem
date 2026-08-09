# GraphMem V5.8 双阶段可调参数与消融实验设计

日期：2026-08-07
代码基线：`GraphMem/src/graphmem`，配置基线：`configs/v5/v5_8_final.json`

本文把「图构建」和「召回」两阶段的全部可调参数枚举出来，标注**代码位置、当前取值、
取值域、单次改动的代价（是否触发重建）、以及是否已被测量过**，并据此给出一份分层的
消融矩阵。目的是让每一个消融臂在开跑之前就知道它花多少钱、能不能复用已有 graph。

---

## 0. 先看三个配置对象的边界

系统里有**三个互相独立的配置 dataclass**，这个切分不是历史包袱，而是消融设计的核心：

| 配置对象 | 位置 | 覆盖范围 | 改动代价 |
|---|---|---|---|
| `GraphMemV5Config` | [config.py:180](../src/graphmem/config.py#L180) | 分段 / 抽取 / 层级 / 建边 / 查询预算 | **改任何字段都会变 `config_hash`**，refine 缓存全部失效，必须重建 graph |
| `ProjectionConfig` | [projection/config.py:41](../src/graphmem/projection/config.py#L41) | 确定性投影层（manifest / 值格 / 时序闭包 / span） | 不触发 LLM 调用，只重放投影 |
| `AnswerConfig` | [answer/rendering.py:27](../src/graphmem/answer/rendering.py#L27) | 证据渲染与作答 | 完全在线，同一个 db 直接重跑 |

`ProjectionConfig` 和 `AnswerConfig` 被刻意排除在 `GraphMemV5Config` 之外，就是为了不污染
`config_hash`（两处 docstring 都写明了这一点）。**任何新增的召回侧旋钮都应该沿用这个约定**，
否则一个纯检索实验会连带作废整套抽取缓存。

⚠️ 但注意：`query_budget` 目前**在** `GraphMemV5Config` 里（[config.py:189](../src/graphmem/config.py#L189)）。
它是纯召回期参数却会改 `config_hash`。实践中 `run_v5_6_answer.py` 用 `--max-evidence-turns`
之类的 CLI flag 绕过去，但这是个设计上的漏洞，做召回消融时要留意别误触发重建。

---

## 1. 阶段一：图构建参数

### 1.1 场景切分 `SceneConfig`（[config.py:99](../src/graphmem/config.py#L99)）

| 参数 | 默认 | v5_8_final | 取值域 | 备注 |
|---|---|---|---|---|
| `min_turns` | 2 | 2 | ≥1，≤max | 场景最小轮数 |
| `max_turns` | 8 | **4** | ≥min | 直接决定每次抽取的输入规模 |
| `topic_similarity_threshold` | 0.55 | 0.55 | [0,1] | 低于此值切场景；与实体重叠/QA 配对联合判定（[pipeline.py:340](../src/graphmem/build/pipeline.py#L340)） |
| `max_events_per_scene` | 3 | 3 | >0 | 每场景事件节点上限（[pipeline.py:431](../src/graphmem/build/pipeline.py#L431)） |
| `coreference_margin` | 0.08 | 0.08 | [0,1] | 共指消解裁决边界（[refine.py:65](../src/graphmem/build/refine.py#L65)） |
| `refine_batch_size` | 24 | 24 | >0 | 共指批大小，只影响吞吐 |
| `llm_semantic_extraction` | False | **True** | bool | 关掉退化为纯确定性图 |
| `llm_hierarchy_compression` | False | False | bool | 层级摘要是否走 LLM |

### 1.2 语义抽取 `ModelConfig.semantic_*`（[config.py:39–95](../src/graphmem/config.py#L39)）

这是**成本最高、已测最充分**的一组。

| 参数 | 默认 | v5_8_final | 取值域 | 已测 |
|---|---|---|---|---|
| `semantic_extraction_mode` | `legacy_batch` | **`strict_pair`** | legacy_batch / strict_single / strict_pair / strict_batch | 部分 |
| `semantic_batch_scenes` | 4 | 4 | >0 | — |
| `semantic_max_facts_per_scene` | 12 | **4** | >0 | ✅ A4 臂（4→3） |
| `semantic_quote_evidence` | True | True | bool | ✅ B0 vs B1：**+6.0pp all_hit，最大单因子** |
| `semantic_scene_summary_chars` | 0 | 0 | ≥0，0=关 | ✅ B0 vs B2：**−1.9pp（负收益）** |
| `semantic_scene_entities` | False | False | bool | ✅ B0 vs B3：**−3.0pp（负收益）** |
| `semantic_predicate_max_chars` | 0 | 0 | ≥0，0=关 | ✅ B4 vs B5：自由谓词 +1.1pp |
| `semantic_compile_summary` | False | True | bool | — |
| `semantic_batch_output_tokens` | 4096 | 32768 | >0 | 溢出保护，非预算旋钮 |
| `semantic_expected_output_tokens` | 0 | 600 | ≥0 | 预算记账基准，解耦「上限」与「预算」 |
| `semantic_max_tokens_per_memory` | 0 | 300000 | ≥0，0=不限 | 硬预算天花板 |
| `semantic_budget_degrade_at` | 0.75 | 0.75 | (0,1] | 触发降级的预算比例 |
| `semantic_fallback_on_overrun` | True | True | bool | 超支后是否降级后续调用 |
| `semantic_max_retries` | 0 | 1 | {0,1} | — |
| `semantic_turn_input_chars` | 0 | 2500 | ≥0 | 单轮输入截断 |
| `semantic_constrained_json` | False | False | bool | 引导式解码 |
| `semantic_individual_repair` | False | False | bool | — |

### 1.3 层级与实体合并 `CoarsenConfig`（[config.py:110](../src/graphmem/config.py#L110)）

| 参数 | 默认 | v5_8_final | 备注 |
|---|---|---|---|
| `fanout` | 8 | 8 | 每层聚合扇入 |
| `max_levels` | 3 | 3 | 层级深度 |
| `summary_tokens` | 320 | 320 | 层级摘要长度 |
| `cross_session_merge` | True | **False** | 跨会话合并；只在 g0/g4 生效（[pipeline.py:550](../src/graphmem/build/pipeline.py#L550)） |
| `entity_merge` | False | — | ✅ 已测：**+0.2pp，噪声内** |
| `entity_merge_min_sessions` | 2 | — | ≥2 |
| `entity_merge_max_session_share` | 0.25 | — | (0,1] |
| `entity_merge_min_chars` | 4 | — | — |
| `entity_merge_embedding_threshold` | 0.0 | — | 0=只做归一化合并 |

### 1.4 建边 `EdgeConfig`（[config.py:136](../src/graphmem/config.py#L136)）

| 参数 | 默认 | v5_8_final | 取值域 |
|---|---|---|---|
| `graph_variant` | g0 | **g5** | g0..g5，语义图形态总开关 |
| `embedding_k` | 8 | 8 | >0 |
| `max_candidates_per_node` | 24 | 24 | >0 |
| `max_degree_per_relation` | 12 | 12 | >0 |
| `low_threshold` / `high_threshold` | 0.45 / 0.78 | 同 | 0≤low<high≤1 |
| `refine_mode` | `ambiguous_only` | **`none`** | none / ambiguous_only / high_value_only / all_bounded_candidates |
| `max_refine_calls_per_1000_turns` | 20 | **0** | ≥0 |
| `temporal_normalization` | False | **True** | bool |
| `cross_session_portals` | False | False | bool |
| `predicate_embedding_threshold` | 0.92 | 0.92 | [0,1] |
| `predicate_cluster_scope` | `slot` | — | slot / owner / memory |
| `predicate_cluster_mode` | `mutual_pair` | — | mutual_pair / agglomerative |
| `portal_degree_cap` | 2 | 2 | >0 |
| `relation_degree_caps` | 见默认 | 9 个关系单独设上限 | 每关系一个整数 |

`relation_degree_caps` 是**唯一一个 per-relation 的旋钮**，v5_8_final 已经把它从默认的
11 个关系换成了 9 个 g5 专用关系。做边消融时它是最高杠杆的一项，但注意它是 mapping，
不能用简单的一维扫描。

### 1.5 构建 profile

`profile: b0..b5`（[pipeline.py:60](../src/graphmem/build/pipeline.py#L60)）是一个**累加式层级开关**，
level = profile 序号，逐级打开更多构建层；`b6` 是 legacy 参照，不是有效构建 profile。
`GateBAblationRunner`（[ablation/runner.py:128](../src/graphmem/ablation/runner.py#L128)）已经内置了 b0–b5 的漏斗。

---

## 2. 阶段二：召回参数

### 2.1 `QueryBudget`（[domain.py:555](../src/graphmem/domain.py#L555)）

| 参数 | 默认 | v5_8_final | 说明 |
|---|---|---|---|
| `max_hops` | 2 | 2 | 遍历跳数 |
| `max_visited_nodes` / `max_visited_edges` | 96 / 192 | 96 / 192 | 总量上限 |
| `max_seed_nodes` | 64 | — | 0=继承 `max_visited_nodes` |
| `max_traversal_nodes` | 0 | — | 0=继承；与 seed 分账，避免互相饿死 |
| `max_frontier` | 32 | 32 | 前沿宽度 |
| `max_evidence_turns` | 16 | 16（**CLI 覆盖为 32**） | 见下方「最高优先级臂」 |
| `max_evidence_tokens` | 5000 | 5000 | — |
| `max_candidate_reservoir` | 576 | — | 只存 id，加宽不耗 context |
| `max_active_facts` | 96 | — | 进入绑定的 fact 数 |
| `max_active_facts_per_operand` | 32 | — | — |
| `max_query_views_per_operand` | 6 | — | — |
| `max_answer_tokens` / `_hard` | 10000 / 13000 | — | hard ≥ soft |

**⚠️ 两个死旋钮，不要浪费消融臂：**
- `max_llm_reranks`：全代码库 **0 个消费者**，改它没有任何效果。
- `max_iterations`：唯一引用在 [navigator.py:278](../src/graphmem/retrieval/navigator.py#L278)，
  只用来**上报**迭代数，不构成循环边界。

### 2.2 硬编码常量（**不在任何 config 里，需改代码**）

这批常量在功能上完全是可调参数，但目前只能通过改源码来消融。**如果要系统性做召回消融，
第一步就是把它们提到一个 `RetrievalConfig` 里**（沿用 `AnswerConfig` 的分离约定）。

| 常量组 | 位置 | 内容 |
|---|---|---|
| 种子深度 | [seeding.py:28–37](../src/graphmem/retrieval/seeding.py#L28) | `LEGACY_BM25_DEPTH=96`、`LEGACY_DENSE_DEPTH=96`、`LEGACY_SESSION_FANOUT=8`、`SESSION_FANOUT_MARGIN=6`、`VIEW_BM25_DEPTH=48`、`VIEW_DENSE_DEPTH=48` |
| Fact 蓄水池 | [facts.py:25–36](../src/graphmem/retrieval/facts.py#L25) | `SOURCE_PROJECTED_PER_OPERAND=128`、`STRUCTURED_PER_OPERAND=128`、`ROUTING_PER_OPERAND=64`、`GLOBAL_LEXICAL=64`、`RESERVOIR_SOFT_LIMIT=384`、`RESERVOIR_HARD_LIMIT=768`、`ACTIVE_PER_OPERAND=32`、`ACTIVE_SHARED_BRIDGE=16`、`ACTIVE_GLOBAL_FALLBACK=16`、`ACTIVE_TOTAL=96`、`SOURCE_RANK_ADMIT=32` |
| 绑定权重 | [bindings.py:36–42](../src/graphmem/retrieval/bindings.py#L36) | `owner=0.40, predicate=0.28, source_projection=0.14, scope=0.10, value_type=0.08, temporal=0.08, graph_path=0.06, session=0.04`；`ACCEPT_THRESHOLD=0.30`；`OWNERLESS_MIN_SIGNALS=2` |
| 融合权重 | [navigator.py:601](../src/graphmem/retrieval/navigator.py#L601), [:893](../src/graphmem/retrieval/navigator.py#L893) | 宽蓄水池路径：`1.2*exact + bm25 + dense + 0.8*graph + 0.7*binding + 0.4*|operands| + 0.25*role + 0.5*slot + 0.12*session + adjacency`；另有一套 0.55/1.2 的并行权重 |
| 遍历优先级 | [navigator.py:823](../src/graphmem/retrieval/navigator.py#L823) | 跳数惩罚 `-0.08*(hop+1)` |
| 集合覆盖 | [navigator.py:949](../src/graphmem/retrieval/navigator.py#L949) | `gain*1.25 + diversity - 0.0005*token_cost` |
| 会话打分 | [navigator.py:774](../src/graphmem/retrieval/navigator.py#L774) | `exact*1.4 + bm25 + dense` |

融合权重（尤其 `exact` 的 1.2/1.4 与 `graph` 的 0.8/0.55 不一致）是**目前最没有被系统性
测过的一块**，而它直接决定 packing 顺序。

### 2.3 导航器与 harness profile

- `NavigatorVariant`：`n0_legacy / n1_raw_fusion / n2_provenance / n3_priority / n4_certificate / n5_set_cover`
  （[navigator.py:57](../src/graphmem/retrieval/navigator.py#L57)）。runner 已内置 N1–N5 选优。
- `HarnessProfile`：`h0..h6, h8, h9, h10`（[navigator.py:66](../src/graphmem/retrieval/navigator.py#L66)），
  **累加式**。当前默认：`run_v5_6_answer.py` 用 `h9`，`measure_v5_8_arm_recall.py` 用 `h10`。
  注意 h7 缺席、h8 先于 h7 落地，是依赖顺序导致的，不是笔误。

### 2.4 `ProjectionConfig`（[projection/config.py:41](../src/graphmem/projection/config.py#L41)）

6 个特性开关 + 7 个 tunable，已有成型的 **P0–P9 / R0–R2 臂**（[projection/config.py:134](../src/graphmem/projection/config.py#L134)）。

特性：`collection_manifest`、`dialogue_pair`、`temporal_closure`、`fact_spans`、`value_lattice`、`event_frames`
Tunable：`manifest_min_members=1`、`dialogue_pair_window=1`、`temporal_edge_cap=32`、
`span_derivation=value|value_predicate`、`span_min_chars=4`、`shared_value_cap=16`、
`predicate_normalization=raw|head|head_stem`、`chain_includes_scope`、`chain_includes_predicate`

`predicate_normalization: raw→head_stem` 已测：singleton 集合 95.9%→78.4%，可数集合 ×4.4，
**零成本**（不调 LLM、不重建、不作废缓存）。这是整份文档里性价比最高的一项。

### 2.5 `AnswerConfig`（[answer/rendering.py:27](../src/graphmem/answer/rendering.py#L27)）

`span_window`（None=全轮）、`include_dates`、`include_speaker`、`closed_form_enabled`、
`max_output_tokens=256`、`max_speaker_chars=48`

CLI 暴露（[run_v5_6_answer.py:46–71](../scripts/run_v5_6_answer.py#L46)）：
`--profile`、`--max-evidence-turns`（默认 32）、`--max-answer-tokens`、`--span-window`、
`--no-closed-form`、`--no-h10-owner-rescue`、`--no-h10-traversal`、
`--no-manifest-collection-key`、`--rank-mandatory`、`--embedding`

---

## 3. 按代价分层

| 层 | 内容 | 单臂代价 | 能否复用 graph |
|---|---|---|---|
| **T0 免费** | QueryBudget、AnswerConfig、navigator variant、harness profile、§2.2 全部常量 | 分钟级，无 LLM 构建 | ✅ 同一个 sqlite |
| **T1 便宜** | ProjectionConfig（P/R 系列） | 投影重放，无 LLM | ✅ 同一个抽取结果 |
| **T2 昂贵** | 任何 `GraphMemV5Config` 字段 | 全量 LLM 重建；110 memory ≈ 2 小时（实测 25 并发） | ❌ config_hash 变化，缓存全废 |

**结论：T2 的臂必须严格限量。** 今天被中止的那次 110 题重建，50 题耗时 48 分钟且出现 10 次
`APITimeoutError`（失败率 20%），意味着一个 T2 臂的真实墙钟成本约 2 小时且不可靠。

---

## 4. 已测结果汇总

### 4.1 B 系列构建臂（761 题，`artifacts/v5_8/phase_a/arm_recall*.json`）

| 臂 | 差异 | all_hit | recall | 相对 B0 |
|---|---|---:|---:|---:|
| B0_core | 全关 | 0.512 | 0.532 | — |
| B1_quote | +quote_evidence | **0.572** | 0.572 | **+6.0pp** |
| B2_summary | +scene_summary | 0.493 | 0.506 | −1.9pp |
| B3_entities | +scene_entities | 0.482 | 0.504 | −3.0pp |
| B4_all | 三者全开 | 0.549 | 0.556 | +3.7pp |
| B5_all_free_p | B4 + 自由谓词 | 0.560 | 0.562 | +4.8pp |

**读法：只有 `quote_evidence` 是正收益，summary 和 entities 单独都是负收益，
且三者叠加（B4=0.549）反而低于 B1 单独（0.572）——它们互相干扰。**
B1 是当前最优构建臂。

### 4.2 entity merge（`artifacts/v5_8/merge/recall.json`）

control 0.572 → merge 0.574，**+0.2pp，在噪声内**。cat1 从 0.20→0.22 是唯一有方向性的分层。

### 4.3 Phase B：召回方案的准确率与延迟（2026-08-07 实测）

脚本 [measure_v5_8_phase_b_retrieval.py](../scripts/measure_v5_8_phase_b_retrieval.py)，
结果 `artifacts/v5_8/phase_b/retrieval_arms.json`。
固定图 = B1_quote（`unit_gate_B1_quote_20260806T180104Z`，10 memory / 761 题），
只变召回方案，9 臂并行各占一个进程与一份 db 副本。

| 臂 | all_hit | recall | p50 | p95 | mean | McNemar vs h0 |
|---|---:|---:|---:|---:|---:|---|
| `n1_raw_fusion@32` | 0.6899 | 0.6489 | **40.8ms** | 395.8 | 82.6 | p=0.28（无差异） |
| **`h0_n5@32`** | **0.7004** | **0.6579** | 57.6ms | 680.4 | 114.2 | — |
| `n5_set_cover@32` | 0.7004 | 0.6579 | 58.4ms | 592.6 | 113.8 | 761 全平 |
| `h5_algebra@32` | 0.5020 | 0.5171 | 60.9ms | 685.6 | 123.2 | p=4.6e-26 |
| `h8_reservoir@32` | 0.4166 | 0.4260 | 128.1ms | 838.8 | 237.8 | p=1.3e-44 |
| `h9_facts@32` | 0.4166 | 0.4296 | 142.8ms | 864.2 | 263.6 | p=1.6e-45 |
| `h10_ast@32` | 0.5716 | 0.5722 | 156.7ms | 871.9 | 278.9 | p=4.3e-13 |
| `h10_ast@48` | 0.6032 | 0.6155 | 156.9ms | 882.1 | 279.7 | p=4.1e-08 |
| `h10_ast@64` | 0.6268 | 0.6471 | 158.7ms | 850.2 | 276.1 | p=5.4e-05 |

**两个交叉验证先确认测量是对的：**
- `h0_n5` 与 `n5_set_cover` 逐题完全一致（0 胜 0 负 761 平）——`HarnessProfile.H0_N5`
  在定义上就是 n5 路径，对上了。
- `h10_ast@32` = 0.5716，与 Phase A 用 `measure_v5_8_arm_recall.py --profile h10` 在
  同一个 B1 图上测得的 0.572 一致——本脚本与既有工具口径相同。

**结论一：整条 harness 阶梯在轮级检索上不如 legacy 路径，且更慢。**
h0/n5 的 0.7004 对 h10@32 的 0.5716，差 **13pp**，p=4.3e-13；延迟还高 2.7 倍
（57.6 → 156.7ms p50）。阶梯本身非单调：h5 0.502 → h8 0.417 → h9 0.417 → h10 0.572，
**h8/h9 的 fact 蓄水池是最差的一档，比 h0 低 28pp。**

**结论二：证据预算是唯一免费且单调的杠杆。**
h10 上 32→48→64 单调涨 0.5716 → 0.6032 → 0.6268（+5.5pp），
而延迟几乎不动（156.7 → 158.7ms）——多装 turn 的成本落在作答侧的 context，不在检索侧。
这印证并放大了 §4.4 里那条 +1.68pp 的旧测量。但即使 h10@64 也没追上 h0@32。

**结论三：分层看，harness 只在它被设计的那一类上赢。**

| stratum | n | n1 | **h0/n5** | h5 | h8 | h9 | h10@32 | h10@48 | **h10@64** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| locomo_cat1（多跳） | 142 | 0.239 | 0.218 | 0.134 | 0.106 | 0.113 | 0.176 | 0.225 | **0.310** |
| locomo_cat2 | 156 | 0.865 | **0.853** | 0.667 | 0.551 | 0.551 | 0.750 | 0.756 | 0.763 |
| locomo_cat3 | 44 | 0.250 | **0.318** | 0.227 | 0.159 | 0.159 | 0.318 | 0.341 | 0.341 |
| locomo_cat4 | 418 | 0.823 | **0.847** | 0.593 | 0.498 | 0.495 | 0.665 | 0.701 | 0.713 |
| lme_multi_session | **1** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

`h10_ast@64` 是**唯一在 cat1（多跳）上超过 h0 的臂**（0.310 vs 0.218，+9.2pp），
代价是 cat4 掉 13.4pp（0.847 → 0.713）。cat4 有 418 题、cat1 只有 142 题，
所以总分上 h0 赢——但这是题型分布决定的，不是方案优劣的结论。

**三条必须记住的限制：**
1. **延迟是 9 路并发下测的**，9 个进程共享 8001 端口的 embedding 服务，排队时间计入
   `stage_latency_ms`。相对排序可信（40 vs 158ms 的差距远超噪声），绝对值偏高。
   要用于容量规划需串行复测。
2. **`lme_multi_session` 仍然只有 n=1**，整张表实质上全是 LoCoMo。
3. **这是检索 all_hit，不是判分准确率。** h9/h10 构建的 fact 蓄水池与算子 AST 是喂给
   closed-form composer 的，而 all_hit 只数打包进去的 turn。h10 在这里的落后**不能**
   直接读成作答会更差——这一层测不到它的收益。要下结论必须跑 answer + judge。

### 4.4 已知但未采纳

`max_evidence_turns: 32 → 48`，实测 **+1.68pp，CI [+0.31, +3.14]**，且 48 轮仍装得下
10K 作答预算（[V5_8_OVERNIGHT_PLAN.md](V5_8_OVERNIGHT_PLAN.md)）。属 T0，零成本。

---

## 5. 建议的消融矩阵

### 5.1 已由 Phase B 完成的部分

E1/E2（证据预算）与 H1（h9 vs h10）已测，见 §4.3。E1/E2 单调有效且零延迟成本，
应当采纳；H1 的答案是**两个都不如 legacy 路径**，所以「统一默认值」这个提法本身就问错了。

### 5.2 Phase B 之后的下一步

| 优先级 | 动作 | 理由 |
|---|---|---|
| **P0** | 在 `h0_n5@64` 与 `h10_ast@64` 两个臂上跑 answer + judge | §4.3 限制 3：检索 all_hit 测不到 h9/h10 的 closed-form 收益。这是唯一能判定 harness 阶梯该不该保留的实验，也是目前唯一值得付 judge 成本的地方 |
| **P0** | `h0_n5` × 证据预算 32/48/64 | 预算在 h10 上单调有效且免费，但从未在**胜出的** legacy 路径上测过 |
| P1 | 串行复测胜出臂的延迟 | §4.3 限制 1；每臂约 80 秒 |
| P1 | 查 h8/h9 为何是负 28pp | fact 蓄水池是阶梯里唯一的绝对倒退，且 h10 只把它救回一半。这是个 bug 信号，不是调参问题 |
| P2 | 按题型路由：cat1 走 h10，其余走 h0 | §4.3 结论三给出的直接读法，但需先确认不是过拟合 761 题 |

### 5.3 仍未测（T0，需先做一次小重构）

| 臂 | 变更 | 假设 |
|---|---|---|
| W1 | 融合权重 `exact` 1.2→1.0 | 检验 exact 加权是否过度 |
| W2 | 融合权重 `graph` 0.8→1.2 | 图信号目前权重最低，值得试探 |
| W3 | `ACCEPT_THRESHOLD` 0.30→0.20 | 提高绑定召回 |
| S1 | `VIEW_*_DEPTH` 48→96 | 与 legacy 深度对齐 |

W/S 系列需要先做一次**小重构**：把 §2.2 常量提到 `RetrievalConfig`。这次重构本身
不改行为（默认值原样搬），是所有后续召回消融的前置条件。

### 5.2 次优先（T1，投影重放）

`predicate_normalization: raw → head_stem` + `chain_includes_scope=False`（即 P8），
再叠 `value_lattice`（P9）。已有 ARMS 定义，直接跑。

### 5.3 谨慎排期（T2，需重建）

按 §4.1 的证据，构建侧只剩两个值得花 2 小时的臂：

| 臂 | 变更 | 依据 |
|---|---|---|
| C1 | 回到 B1（只开 quote，关 summary+entities） | B1 是实测最优，而 v5_8_final 是 B5 形态 |
| C2 | `scenes.max_turns` 4→6 | 从未测过；直接影响每次抽取的上下文规模 |

`relation_degree_caps` 和 `graph_variant` 虽然杠杆高，但都是多维/离散跳变，
在 T2 预算下不适合做单因子扫描，建议先在 T0 层用召回诊断定位再决定。

---

## 6. 执行前必须修的三个问题

1. **两个 phase_a 配置已经加载不了。**
   `A5_free_predicate.json` 和 `A6_out3072.json` 引用了 `semantic_aspect_field` /
   `semantic_category_field`，这两个字段在当前 `ModelConfig` 里不存在，
   `load_config` 直接抛 `TypeError`。实测：
   ```
   A0_baseline      OK
   A5_free_predicate FAIL TypeError: unexpected keyword 'semantic_category_field'
   B0_core          OK
   ```
   这两个臂的结果无法复现，要么补回字段，要么把配置标记为作废。

2. **LME 侧的检索指标基本是空白。**
   §4.1 的 761 题里 `lme_multi_session` 只有 **n=1**，其余全是 LoCoMo。所以
   「B1 最优」这个结论**只在 LoCoMo 上成立**。今天中止的 110 题重建正是为了补这个洞
   （50 multi_session + 50 temporal_reasoning + 10 LoCoMo）。在它补上之前，
   LME 的 0.489 无法拆分成「索引没召回」还是「作答没答对」。

3. **`query_budget` 位置错误。** 它在 `GraphMemV5Config` 里，改它会作废抽取缓存，
   而它其实是纯召回参数。建议随 §5.1 的重构一起挪到 `RetrievalConfig`。

---

## 7. 图构建与索引审计（2026-08-07）

对象：B1_quote 图（10 memory / 367 session / 5,221 turn / 14,328 node / 21,117 edge）。

### 7.1 看起来正确的部分

- 规模一致：LME 每 memory ≈1,600 node / 487 turn / 48 session，LoCoMo ≈1,200 / 550 / 26。
- 孤立节点率极低：scene / canonical_fact / evidence_group_ref / routing_card /
  collection_scope 均为 **0%**，canonical_entity 0.4%，time_anchor 3.0%。
- 每场景事实数 4,669/1,700 = 2.75，与 B1 臂实测的 2.95 facts/scene 吻合。
- 路由层级自洽：routing_card level1=367（= session 数）→ level2=50 → level3=10（= memory 数）。

### 7.2 七个问题，按严重度排序

**① 图里一条跨会话边都没有。**
把每条边的证据组回溯到 session，21,117 条边**全部**落在单一 session 内：

| relation | 边数 | 证据跨会话 |
|---|---:|---:|
| scene_contains | 10,207 | 0 (0.0%) |
| has_fact | 3,997 | 0 (0.0%) |
| participates_in | 3,377 | 0 (0.0%) |
| refines_to | 2,117 | 0 (0.0%) |
| at_time | 625 | 0 (0.0%) |
| collection_co_member | 607 | 0 (0.0%) |
| state_next | 179 | 0 (0.0%) |
| temporal_before | 8 | 0 (0.0%) |

唯一的跨会话通路是 routing_card 的三层聚合（session→组→memory）。三个跨会话机制
**全部关着**：`cross_session_merge=False`、`cross_session_portals=False`、
`entity_merge` 用默认 False。这直接对应 multi_session 判分 0.489（LME 最差）
和 LoCoMo cat1 全 hit 仅 0.218–0.310（Phase B 最差分层）——cat1 的证据平均散在 2.68 个 session 里，
而图里没有任何一条边能把它们连起来。

**② 实体层实质上不参与路由。**
1,399 个有边实体中 **1,345 个（96.1%）只触达 1 个 session**，跨 ≥2 session 的只有 54 个（3.9%）。
而 LME 每 memory 有约 48 个 session。度数最高的实体是：

```
John 154 / John 145 / Joanna 139 / Tim 138 / Maria 132 / Nate 131 / Caroline 93 ...
```

全是说话人名字——在一个 memory 内恒定，区分度为零。这与 [config.py:116](../src/graphmem/config.py#L116)
的注释完全一致，说明那段注释描述的是**现状而非已修复的历史**。

**③ 度数上限只在源侧生效，hub 不受约束。**

| relation | 配置 cap | max(出边) | max(入边) |
|---|---:|---:|---:|
| participates_in | 32 | 29 | **154** |
| has_fact | 128 | **128** | 1 |
| scene_contains | 32 | 8 | 1 |
| collection_co_member | 32 | 8 | 8 |

`participates_in` 入边达到 cap 的 4.8 倍，所以 `relation_degree_caps` 挡不住说话人实体
变成超级 hub。另外 `has_fact` 出边**正好顶到 128**，说明这个 cap 是紧的、正在截断事实。

**④ 状态链断裂。** 210 个 state_head 里 **125 个（59.5%）没有任何边**，state_next 全图仅 179 条。
LME 的 knowledge_update（78 题、判分 0.667）问的正是「现在的值是什么」，靠的就是这条链。

**⑤ 时序闭包几乎不存在。** 4,669 个 fact 里只有 625 个（13.4%）带时间，
全图 `temporal_before` 只有 **8 条**。而 temporal_reasoning 是 133 道 LME 题。

**⑥ 证据 span 100% 退化为整轮。** 5,221 条 evidence_member **全部**是
`span_start=0, span_end=len(turn)`。抽取阶段算出的 span 在 build 阶段被丢弃，
所以证据没有任何轮内精度。`ProjectionConfig.fact_spans` 这个臂就是为此存在的。

**⑦ 图节点零向量，dense 通道只能匹配原始 turn。**
9,329 条 embedding = 5,221 turn + 4,108 `predicate:<hash>`，
**graph_nodes 命中 0 条**。scene、routing_card、canonical_fact、canonical_entity
全都没有向量。routing_card 的职责是路由问题，却无法被语义匹配——
这解释了为什么 §4.3 里加了整套 harness 机制的 h8/h9 反而比 legacy 差 28pp：
它们在一个没有向量索引的图上做图侧检索。

### 7.3 建议调整（按性价比）

| 优先级 | 调整 | 层 | 依据 |
|---|---|---|---|
| **1** | 给 routing_card / scene 建 embedding | 索引，无需重建 | ⑦；成本是一次 index_memory，收益直击 §4.3 结论一 |
| **2** | 开 `fact_spans` 投影 | T1 | ⑥；已有 P5 臂，零 LLM 成本 |
| **3** | 开 `temporal_closure` 投影 | T1 | ⑤；已有 P4 臂 |
| **4** | `participates_in` 加入边侧 cap，或把说话人实体列入停用表 | 需重建 | ③②；hub 同时污染绑定和遍历 |
| **5** | 打开三个跨会话开关中的至少一个并实测 | T2 | ①；但注意 `entity_merge` 已测仅 +0.2pp，因为②——实体层本身就是坏的，先修②再谈① |
| **6** | 查 state_head 为何 59.5% 无边 | 需定位 | ④；这是 bug 而非调参 |

⚠️ 顺序很重要：**①的三个开关不要先开。** entity_merge 已经实测只有 +0.2pp，
原因是②——实体层 96% 单会话且 hub 全是人名，合并一堆没用的键仍然没用。
先修实体质量（②③），跨会话连接才有东西可连。

## 8. 图结构分析：一个根因，两条独立的改进线

### 8.1 根因：`predicate` 是命题全文，因此所有符号连接键都退化为单例

```
canonical_fact: distinct predicate = 3,864 / 4,669 facts = 0.83
只出现 1 次的 predicate = 3,557（92.1%）
最常见: recommended(27) shared(26) offers(22) has(22) released(19)
```

抽取把整条命题写进 `predicate`（"lost job as a banker"、"started business"），
于是**任何包含 predicate 的连接键都等价于一个 fact 唯一 ID**。这不是多个独立缺陷，
是同一个根因的四种表现：

| 连接键 | 单例/单会话率 | 中段（2–12 会话）|
|---|---:|---:|
| canonical_entity（名字） | 96.8% 单会话 | 24 / 1,373 = **1.7%** |
| value_key | 98.2% 单会话 | 74 / 4,183 = **1.8%** |
| (owner, collection_key) | 95.0% 单会话 | 43 / 1,274 = **3.4%** |
| (owner, predicate) @ state_head | 88.5% 单成员 | — |

**四个互相独立的键，撞的是同一堵墙。** 实体层的分布还是双峰的，中间是空的：
`{1: 1329, 2: 17, 3: 5, 4: 2, ←断层→ 18: 1, 19: 4, 21: 1, ..., 29: 4}`——
要么只在 1 个会话出现（连不上任何东西），要么在 18–29 个会话出现（说话人名字，区分不了任何东西）。
`entity_merge_min_sessions=2` 与 `entity_merge_max_session_share=0.25` 想切出的中间带，
在这份语料上只有 24 个实体。这才是它实测 +0.2pp 的原因。

补充两个坏掉的前提：
- `collection_key` 已退化为 `value_type`，全图只有 5 个取值
  `{text:3959, number:398, time:210, currency:58, boolean:44}`。
  R 系列「按类而非按动词分组」的前提（`semantic_category_field`）**从未建成**，
  这也是 §6① 里那两个加载不了的配置所引用的字段。
- state_head 里 **98/210 的 `predicate` 是空的**，它们永远无法成链。
  这与 88.5% 的单成员键一起，解释了 59.5% 的孤立率和仅 179 条 `state_next`。

### 8.2 唯一 100% 覆盖的键只能给序，不能给主题连接

`observed_at` 在全部 4,669 个 fact 上非空，但它在**每个会话内恒定**
（363/363 个会话的 distinct `observed_at.start` 均为 1）——它是会话时间戳。
所以它提供的是一个跨会话全序，不是主题连接。真正能区分的事件时间
（`time_interval`）覆盖率只有 13.5%。

### 8.3 由此得到的两条改进线（互相独立，不要混做）

**线 A：序层——便宜、不依赖任何连接键。**
`observed_at` 100% 覆盖已经足够支撑「现在的值是什么」和「A 是否早于 B」。
当前 state chain 用 `(owner, predicate)` 做键，而 predicate 是命题全文，所以必然断。
**把状态链改用 `(owner, collection_key/scope) + 会话序` 重建**，不需要新的抽取。
目标分层：LME knowledge_update（78 题 / 0.667）与 temporal_reasoning（133 题 / 0.714）。

**线 B：主题跨会话——必须是稠密的，不能是符号的。**
§8.1 已经用四个键证明符号路线在这份语料上没有质量。而 Phase B 给出了正面证据：
legacy 路径赢 harness 13pp，其跨会话机制是**词法 session fan-out**
（`LEGACY_SESSION_FANOUT=8`，一个相似性机制），而 harness 的是符号遍历——
在一个 100% 单会话边的图上无物可遍历。**相似性有效，符号无效。**
所以 §7.2⑦「图节点零向量」不是锦上添花，而是线 B 的全部内容：
routing_card / scene / canonical_fact 建向量，让跨会话检索走稠密通路。

### 8.4 明确不要做的

- **不要再调 entity_merge / value_lattice / R 系列的阈值。** §8.1 已经说明它们受限于
  语料而非参数；R 系列的前提字段还不存在。
- **不要先开三个 cross-session 开关。** 没有可用的连接键时，打开它们只会加边不加信息。
- **不要加更多会话内边。** 会话内层已经很密（1,700 scene / 10,207 scene_contains），
  孤立率 0%，且 `has_fact` 出边正好顶到 cap=128，说明已在截断。
- **规范化 predicate 的三条路都试过且都失败了**（embedding 聚类 0.922、
  LLM 词表 0.929、p≤24 重建 0.949 且截断命题）。线 A/B 都绕开了这个问题，
  这是它们的设计前提，不是疏漏。

## 9. 跨会话连接方案：两个候选源的实测与由此得到的设计

### 9.1 两个候选源，实测（10 memory / 1,700 scene / 339 个 gold 需要的会话对）

**源1 — 精确实体名跨会话配对。** 这个机制**已经实现**：
[`pipeline.py::_ambiguous_candidates`](../src/graphmem/build/pipeline.py#L1029)
按跨会话共享的 event 实体名分块（已排除 user/assistant/participant）、度数封顶 24、
交 LLM 判 `SAME_EVENT / PORTAL / NONE`。它被 `refine_mode: "none"` 和
`max_refine_calls_per_1000_turns: 0` 关掉了。

实测它在这份图上能产出的全部候选：

```
跨会话候选对 = 29（10 个 memory 合计）
命中 gold 需要的会话对 = 2 / 339 = 0.6%
```

**29 个候选、命中 2 个。** 这不是阈值问题，是候选池本身是空的——与 §8.1 的
实体分布一致（可用中段只有 24 个实体）。**源1 不值得打开。**

**源2 — 场景摘要 embedding。** 1,700 个 scene summary 全部向量化后：

| 阈值 | 候选会话对 | 命中 gold |
|---:|---:|---:|
| 0.60 | 2,541 | 335/339 = **98.8%** |
| 0.70 | 1,364 | 249/339 = 73.5% |
| 0.80 | 330 | 56/339 = 16.5% |
| 0.90 | 27 | 7/339 = **2.1%** |

**注意这条曲线的方向：阈值越高，gold 命中率越低。**
最相似的会话对**不是**答案需要的会话对。这不是信号弱，是信号错——
会话之间的相关性是**问题条件的**，而预计算的相似度不知道问题是什么。

还有一个语料差异：

| 语料 | 跨会话 mean | 同会话 mean | 可分性 |
|---|---:|---:|---|
| LME（5 memory） | 0.275–0.301 | 0.526–0.561 | 好 |
| LoCoMo（5 memory） | 0.475–0.586 | 0.495–0.608 | **几乎为零** |

LoCoMo 是两个人持续聊各自生活，每个会话主题都像——而 339 个 gold 会话对几乎全在 LoCoMo。

### 9.2 由此得到的设计：不要预计算跨会话边

9.1 的曲线说明预计算跨会话边是**错误的形状**：要 98.8% 召回就得放到 0.60，
那是 2,541 对（占全部会话对的 35%），再交给 LLM 判就是每 memory 254 次调用、
510 memory 共 ~13 万次——既贵又没准。

正确的形状是**把向量给节点，在查询时做会话路由**：会话相关性是问题条件的，
那就让问题去选会话，而不是预先把会话两两连起来。这同时解释了 Phase B——
legacy 赢 harness 13pp，靠的正是 `LEGACY_SESSION_FANOUT=8` 这个**问题条件的**
词法会话扇出；harness 的符号遍历不是问题条件的，所以输了。

**方案（三步，按依赖顺序）：**

| 步 | 内容 | LLM 成本 | 层 |
|---|---|---|---|
| **B1** | 打开 `semantic_scene_summary_chars=160`，让每个场景有一句**真正的句子**摘要 | **0 次新调用** | T2 |
| **B2** | 给 scene / routing_card 建 embedding | 0 | 索引 |
| **B3** | 检索侧用问题向量做会话路由，替代/补充词法扇出 | 0 | T0 |

B1 的成本是关键：`m`（summary）**本来就在抽取调用的输出 schema 里**
（[semantic.py:28](../src/graphmem/build/semantic.py#L28)），
`semantic_scene_summary_chars` 只是用引导解码约束它的长度。
所以这是每场景多 ~48 个输出 token，**不是一次新调用**。

B1 是必需的，因为当前 summary 是**拼接的事实三元组**，带重复：

```
'Jon lost job as a banker yesterday Jon started business dance studio
 Gina lost job at Door Dash this month Jon started business dance studio'
```

[config.py:87](../src/graphmem/config.py#L87) 的注释早就写明了这一点：
「a sentence is what a question embedding can match」。9.1 里 LoCoMo 可分性接近零，
测的正是这堆三元组汤——**换成句子后是否可分，是 B1 唯一要回答的问题**。

⚠️ 已知的反向证据：B2 臂（summary 打开）在 Phase A 实测 **−1.9pp**。
但那是把 summary 喂给**词法**路由卡，不是喂给向量。§4.3 已确认全图零节点向量，
所以「summary 作为向量」从未被测过。这是本方案最大的未知，也是最先该测的一步。

### 9.3 LLM 后处理该放在哪里

你的直觉（在最上层 coarsen 图上让 LLM 做合并）方向对，但要换目标：
**不是判断两个场景是否该合并**（9.1 说明候选池不可用），
**而是给上层节点写可被问题匹配的路由文本**。

理由是扇入规模：routing_card 是 level1=367 / level2=50 / level3=10。
在 level2+level3 上做 LLM 改写，10 个 memory 只有 **60 次调用**（每 memory 6 次），
510 memory 约 3,060 次——相对源1 方案的 13 万次是两个数量级的差别。
而这 60 个节点恰好是当前唯一的跨会话结构，也是当前最粗糙的一环（词袋、无向量）。

### 9.4 与线 A 的关系

线 A（序层，§8.3）**不依赖以上任何一步**，也不需要 LLM：
`observed_at` 100% 覆盖，把状态链从 `(owner, predicate)` 改键为
`(owner, collection_key/scope) + 会话序` 即可。两条线可以并行推进，
唯一的共享成本是 B1 要求的那次重建——**如果要重建，就把线 A 的改键一起带上**，
省掉一次 T2。

### 9.5 建议的验证顺序

1. **先做 B2+B3（零成本，不重建）**：在现有三元组汤 summary 上建向量、
   做查询时会话路由。若已有增益，B1 只会更好；若无增益，说明问题在检索侧而非摘要质量，
   B1 的重建就不必付。
2. B2/B3 有效后再决定是否为 B1 付一次 T2 重建，并把线 A 的改键并入同一次重建。
3. 9.3 的上层 LLM 改写放在最后——它是在 B3 已经证明「稠密会话路由有效」之后
   才有意义的增强。

## 10. 关系边重新设计：共享稀有指称边（零 LLM 调用）

### 10.1 先看四类题实际需要什么跳

抽真实题目（`longmemeval_s_cleaned.json`）后，四类题的跳结构完全不同：

| 类别 | n | 实例 | 需要的跳 | 连接键 |
|---|---:|---|---|---|
| multi-session | 133 | 「我 Facebook Live 和最热门 YouTube 视频的评论总数」→ 33（会话 17 + 22） | 两个 fact → **聚合算子** | (owner, 属性类) |
| temporal-reasoning | 133 | 「买音箱时我学吉他多久了」→ 四周（会话 16 + 29） | 两个事件 → **时间差** | 绝对时间 |
| knowledge-update | 78 | 「Ethereal Dreams 现在挂在哪」→ 卧室（会话 8 @7月 + 30 @10月） | 同一 (实体,属性) 的多次断言 → **取最新** | (指称, 属性) + 时间序 |
| single-session-preference | 30 | 「推荐个今晚看的节目」→ 需要「脱口秀专场」偏好（**1/46 会话**） | **不是多跳**，是纯路由 | 语义 |

**preference 不该进这个方案**：它是单会话题，当前 0.867，离 0.90 只差 1.0 题，
而 multi-session/temporal/knowledge-update 三类合计差 97.6 题。

前三类共享同一个形状：**一组共享某个指称的 fact，按时间排序，上面跑一个算子**
（求和/求平均、时间差、取最新）。所以它们需要的是同一种边。

### 10.2 决定性测量：稀有词是可用的连接键

对每个 memory 的会话集算词的文档频率，取 `df <= max(2, 5%·会话数)` 的词为「稀有词」，
比较 gold 会话对与随机会话对的共享稀有词数：

| 类别 | gold 对 | 随机对 | 分离度 | gold 对百分位 | gold 进前 5% |
|---|---:|---:|---:|---:|---:|
| multi-session | 15.1 | 1.1 | **13.9x** | 96.5% | 85.0% |
| temporal-reasoning | 22.6 | 1.1 | **20.7x** | 96.5% | 85.8% |
| knowledge-update | 20.4 | 1.1 | **19.0x** | 99.1% | **96.2%** |

与前两个候选源对比（同一目标：找出 gold 需要的会话对）：

| 候选源 | 召回 | 代价 | LLM |
|---|---:|---|---|
| 精确实体名（§9.1 源1） | **0.6%** | 29 个候选 | 需要 |
| 场景摘要 embedding（§9.1 源2） | 98.8% | **35%** 的全部会话对 | 不需要 |
| **共享稀有词 K>=3** | **92.3%** | **12%** 的全部会话对 | **不需要** |

**为什么 embedding 输给词频**：embedding 把场景压成一个向量，量的是**主题**相似；
而 LoCoMo/LME 每个会话主题都像（§9.1 的 LoCoMo 可分性≈0）。稀有词量的是**指称**重合
——同一件具体的事被再次提起。多跳题问的正是同一指称，不是同一主题。

### 10.3 边预算

```
K>= 3: gold 会话对召回 92.3%   每 memory 130 边  (全部对 1,105 -> 12%)
K>= 5:                82.4%              45 边
K>= 8:                71.6%              16 边
K>=12:                60.8%               7 边
```

**建议工作点 K>=3**：130 条边相对图中已有的 2,000–2,500 条边是 **+5%**，
而它是当前图里**唯一**的跨会话结构（现状 0 条）。

### 10.4 设计

**新关系 `shared_referent`（确定性，build 期，零 LLM）：**

1. 每个 memory 在会话粒度上统计词文档频率；`df <= max(2, 0.05·N)` 记为稀有词。
2. 两个 scene/session 共享稀有词数 `>= 3` 时建边，权重 = Σ IDF。
3. 走现有 `relation_degree_caps` 封顶，避免 hub。

它替换掉失败的实体层：**不再依赖抽取去命名实体，而是从原文的词稀有度直接导出指称。**
§8.1 的四个符号键之所以全部失败，是因为它们都建立在抽取写出的自由文本上；
稀有词绕开了抽取。

**配套的两处修改（同一次重建里做完）：**

| 修改 | 目标类别 | 依据 |
|---|---|---|
| 用 `observed_at` 解析相对时间（「两周前」+ 会话日期 → 绝对时间），补齐 `event_time`，再建 `temporal_before` | temporal-reasoning | `observed_at` 100% 覆盖，而 `time_interval` 仅 13.5%、`temporal_before` 仅 **8 条** |
| 状态链改键：`(owner, predicate)` → `(owner, 共享稀有指称)`，按 `observed_at` 排序 | knowledge-update | §8.1：predicate 92.1% 唯一，所以 88.5% 的状态键是单成员、59.5% 的 state_head 孤立 |

### 10.5 这同时解释了 Phase B

§4.3 里 h8/h9/h10 输给 legacy 13pp，不是因为 harness 机制差，
而是因为**它们在一个 100% 单会话边的图上做遍历——无物可遍历**。
fact 蓄水池和算子 AST 恰恰是聚合/时间差/取最新需要的东西。
给它 130 条正确的跨会话边之后，这套机制才第一次有事可做。
**所以 Phase B 的结论应读作「harness 被饿死」，而不是「harness 无用」。**

### 10.6 限制

- 10.2/10.3 测的是 **LME 原文会话粒度**，不是构建后的 scene 粒度；scene 粒度的
  df 分布会不同，需在实现后复测。
- **LoCoMo 未测**（gold 格式不同），而 cat1 是另一个弱项。
- `df<=5%` 与 `K>=3` 两个超参是在同一批 324 题上选的，存在轻度过拟合风险；
  样本量足够但应留出一份 holdout 复核。

## 11. 召回漏斗实测：瓶颈不在索引，在排序

### 11.1 §8–§10 的前提是错的

`shared_referent` 边已实现并 A/B（761 题配对，唯一变量是新边）：

| 臂 | base | +shared_referent | Δ |
|---|---:|---:|---:|
| h0_n5@32 | 0.7004 | 0.7004 / 0.6965 | +0.00 / −0.39pp |
| h10_ast@32 | 0.5716 | 0.5729 | +0.13pp |
| h10_ast@48 | 0.6032 | 0.6045 | +0.13pp |
| h10_ast@64 | 0.6255 | 0.6281 | +0.26pp |

边不是惰性的——探针显示 `shared_referent` 是**走得最多的关系**（120 题 1,850 步，
超过 `refines_to` 的 1,671）。但它不产生提升，加宽 pack 也不解锁它。

漏斗测量给出了原因：

```
蓄水池 mean=421 轮
gold 进蓄水池 = 100.0%          all_hit @蓄水池层 = 1.000
gold 进打包   =  62.9%          all_hit @打包后   = 0.612
gold 在候选中的排名 p50=10, p90=202   （打包只取 32）
```

**gold 已经 100% 在候选池里。检索阶段没有召回问题。**
从 1.000 掉到 0.612 的全部损失发生在排序/打包这一步。

所以 §8–§10 一直在优化**可达性**，而可达性从来不是瓶颈。
§10.5「harness 被饿死」的判断也随之作废——**它不是饿死，是撑死**。

### 11.2 具体机制：融合分里图特征的量纲是错的

[navigator.py:585](../src/graphmem/retrieval/navigator.py#L585)：

```python
graph = 1.0 if turn_id in hydrated_turn_ids else 0.0
```

图的贡献是**二值标志**，不分关系、不分跳数、不看边权，权重 0.8；
再加上 `0.4 * len(operand_ids)`（无上界）和 `0.7 * bscore`。
而词法通道 exact/bm25/dense 各自归一在 ~[0,1]。量纲对不上：

| profile | 打包中纯图候选 | 这些槽位产出的 gold | 纯图候选 fused | 含词法候选 fused |
|---|---:|---:|---:|---:|
| **h10** | **8.3/题（25.9%）** | 250 题共 **9** 个 | **2.41** | 1.24 |
| h0（legacy） | 0.4/题（1.1%） | 250 题共 3 个 | 0.97 | 0.89 |

**h10 把 26% 的证据预算给了完全没有词法命中的轮次，而这些槽位的 gold 产出率约 0.4%。**
h0 只给 1.1%，分数也不失衡（0.97 vs 0.89），于是高出 13pp。
这就是 §4.3 结论一的完整机制。

同时解释了为什么 `shared_referent` 无效：**它增加的是纯图候选**，
而纯图候选正在挤占 gold 的槽位——新边越多，挤占越重，两者相抵。

### 11.3 由此得到的下一步（全部 T0，零重建）

`all_hit` 在蓄水池层已是 **1.000**，打包后 0.612——
**排序完美化的理论上限是 +38.8pp，且不需要动索引一行。**
这比 §5.2/§10 里任何一项都大一个量级。

| 优先级 | 动作 | 依据 |
|---|---|---|
| **1** | 把 `graph` 从二值 0.8 改为按边权/跳数衰减，或直接降权 | 11.2：纯图候选 fused 2.41 vs 词法 1.24 |
| **1** | 给纯图候选在 pack 中设配额上限（如 ≤10%） | 11.2：当前 25.9%，gold 产出 0.4% |
| 2 | 校准 `0.4 * len(operand_ids)` 的无界项 | 同上；这是纯图候选分数偏高的主要来源 |
| 3 | 重测 §5.3 的 W1/W2/W3 融合权重臂 | 现在有了明确的失衡方向 |
| 4 | 暂缓 `shared_referent` 与所有跨会话工作 | 11.1：可达性不是瓶颈 |

### 11.4 跳数衰减与逐步剪枝：实测（761 题）

实现：`ScheduleResult` 返回 `node_hops`，导航器用 `graph_hop_decay ** hop` 取代二值标志；
`execute(expansion_beam=N)` 在入队**前**按 `destination_priority` 只保留每次展开的前 N 个邻居
（原先每个邻居都入队，唯一约束 `queue[:max_frontier]` 砍的是最旧而非最弱的条目）。

准确率（并行跑，配对 vs `h10_ast@32`）：

| 臂 | all_hit | Δ | 配对 |
|---|---:|---:|---|
| h10_ast@32（基线） | 0.5716 | — | |
| h10_decay0.5@32 | 0.5795 | +0.79pp | 12胜6负 p=0.24 |
| h10_decay0.3@32 | 0.5808 | +0.92pp | 14胜7负 p=0.19 |
| h10_beam4@32 | 0.5716 | +0.00pp | 1胜1负 |
| h10_beam2@32 | 0.5756 | +0.39pp | 3胜0负 |
| **h10_d0.3_b2@32** | **0.5848** | **+1.31pp** | 17胜7负 **p=0.064** |

**延迟：剪枝没有收益。** 一份早先的串行测量报出 mean −53% / p95 −76%，
那是**测量假象**：进程内第一个跑的臂要付页缓存与首次 SQLite 访问的预热，
而三次测量里基线恰好都排在第一位。反序复测（同一臂跑两遍，基线夹在中间）：

| 位置 | 臂 | p50 | p95 | mean |
|---|---|---:|---:|---:|
| 1 | h10_d0.3_b2@32 | 154.2ms | 781.5ms | — |
| 2 | h10_ast@32（**未剪枝**） | 116.1ms | 202.4ms | 122.1ms |
| 3 | h10_d0.3_b2@32（**同臂**） | 115.3ms | 208.5ms | 122.2ms |

同一个臂在第 1 位 154/781ms、第 3 位 115/208ms，而未剪枝基线在第 2 位是 116/202ms
——**剪枝与不剪枝的延迟无差别**。任何进程内首臂的延迟数都不可用。

所以合入依据只剩准确率的 +1.2~1.3pp：两次独立测量方向一致
（17胜7负 p=0.064；16胜7负 p=0.093），可复现但**未达 0.05 显著**。
另注意本测量有约 1 题的运行间抖动（同臂两次 0.5848 / 0.5834），
推测来自 dense 检索的批次/浮点顺序。

但 1pp 也确认了 11.1 的诊断：腾出槽位是必要不充分条件。
gold 排名 p90=202 而 pack 取 32——那批 gold 不是被图候选挤掉的，
**是词法分数本身排不上来**。h0 与 h10 共用同一候选池（gold 100% 命中），
h0 用更简单的打分拿 0.7004，h10 用更复杂的打分拿 0.5848：
**11.6pp 的差距全部在 rerank 公式里，不在图这一侧。**
下一步应查 `0.4 * len(operand_ids)` 的无界项与 §5.3 的 W 系列权重。

### 11.5 已作废的结论

- §8.3 线 B「主题跨会话必须走稠密」——前提（可达性不足）不成立。
- §10.5「Phase B 应读作 harness 被饿死」——**方向反了**，是过度加权。
- §10 的 `shared_referent` 设计本身成立（边建成、被走、代价低），
  但它解决的问题不是当前的瓶颈。保留代码，暂缓推进。

## 12. 分层图与关系边是否足够（761 题实测）

### 12.1 召回侧根本没有用到层级

- **种子全是轮级的。** `seeding.py` 的 `CHANNELS = ("exact","bm25","dense")` 全部命中原始
  turn，注释写明「Typed postings retrieve graph nodes, not turns」。
  **不存在自顶向下的卡片路由**：367 张会话路由卡从未作为检索入口被使用。
- **层级关系只是遍历里平权的一条边。** `refines_to` 排在 `DEFAULT_PREFERRED` 末位，
  且 `view.neighbors(include_inverse=True)` 不区分上行/下行——从场景爬到路由卡与
  从路由卡下钻到场景在遍历里是同一件事，没有方向语义。
- **结果**：427 张路由卡既无向量（§7.2⑦）也不产生轮次证据，走上去是死路。
  分层图是为自顶向下路由设计的，而召回是自底向上做的，三层聚合从未被调用。

### 12.2 宽度：加宽是空操作

| beam | all_hit | gold 仅图可达 | 图捞回/题 | 走边/题 |
|---|---:|---:|---:|---:|
| 2 | 0.5834 | 94 (8.48%) | 134.0 | **36.8** |
| 4 | 0.5795 | 95 | 134.4 | 42.4 |
| 8 | 0.5808 | 94 | 134.6 | 43.8 |
| 16 / 32 / off | 0.5795 | 94 | 134.8 | 44.0 |

**beam 16 / 32 / off 逐位完全相同**——遍历本来就没有超过约 16 个邻居可展开。
`beam=2` 一个图独有 gold 都没丢，走边少 16%：**剪枝安全，加宽无用**。
走边只用掉预算的 23%（44/192），**遍历是种子受限，不是预算受限**。

### 12.3 深度：结构在第 4 跳耗尽，当前预算已拿到 99.2%

| hops / nodes | 候选池 gold 覆盖 | gold 仅图可达 | 走边/题 | 访问节点/题 |
|---|---:|---:|---:|---:|
| **2 / 96（当前）** | **94.95%** | 90 (8.12%) | 32.6 | 95.8 |
| 3 / 256 | 94.95% | 90 | 192.2 | 255.3 |
| 4 / 512 | 95.67% | 98 | 358.6 | 421.7 |
| 6 / 2048 | **95.76%** | 99 | 360.7 | 423.8 |

- **hops=3 相对 hops=2 覆盖率零增益**，却多走 6 倍的边。纯浪费。
- **hops=4 ≈ hops=6**：图在第 4 跳就走干净了。
- **结构上界 95.76%**，而当前预算已达 94.95%——**已经拿到上界的 99.2%**。
  把预算放大到 hops=6/2048（11 倍走边、4.4 倍访问节点）只多买 **+0.81pp**。

### 12.4 结论：结构够用，瓶颈全在打包

| 层 | gold 覆盖 / all_hit |
|---|---:|
| 图结构可达上界 | 95.76% |
| 当前预算下候选池覆盖 | 94.95% |
| 实际打包进证据的 gold | **62.9%** |
| 只用词法/dense 的 all_hit 上界 | **0.8515** |
| h0_n5@32 实际 | 0.7004 |
| h10 最优实际（W5） | 0.5926 |

**关系边与分层结构对 95.76% 的 gold 是足够的，当前预算已实现其中 99.2%。**
候选池到打包之间丢掉约 32pp，而这一步不需要图、不需要重建、不需要新边。

至此排除法已经走完：索引/边（够用）、可达性（饱和）、遍历宽度（空操作）、
遍历深度（第 3 跳起零增益）、融合权重（单项最大 0.79pp）、图过度加权（1.3pp）。
**剩下唯一没被测过的是 `_rank_pack` 的集合覆盖逻辑与 legacy packer 的差异**，
而 h0 与 h10 共用同一候选池却差 11pp，指向的正是这里。

### 12.5 可以定的默认值

`expansion_beam=2` — 12.2 证明它不伤召回且省 16% 的边，保留。
`graph_hop_decay=0.3` — 纯 rerank 侧，不改变可达性，+0.92pp 方向稳定，保留。
`max_hops` 可从 2 保持不变；**不要**为了覆盖率放大到 3（零增益、6 倍代价）。

## 13. 决定性结果：全部打包损失都是会话稀释

### 13.1 oracle 实验

把候选池限制在 gold 会话内，**用同一个打分器**取前 32（761 题）：

```
平均 gold 会话数 = 1.35
候选池 578 轮/题  ->  限定在 gold 会话内只剩 31.1 轮/题
当前打包命中的 gold 会话 = 89.8%

实际          all_hit = 0.5848   gold 进包 = 58.0%
完美会话路由  all_hit = 0.9474   gold 进包 = 94.3%
```

**+36.3pp。** 比此前测过的任何一项大一个数量级
（证据预算 +5.5pp、decay+beam +1.3pp、W5 权重 +0.79pp、shared_referent +0.13pp）。

三个数字合起来解释了机制：

1. 答案平均只住在 **1.35 个会话**里。
2. 那些会话在候选池里只有 **31.1 轮**——**几乎正好等于 32 的打包预算**。
3. 完美路由下 gold 进包 94.3%，而候选池覆盖上界是 94.95%
   ——**打包损失归零，打分器一分没丢**。

所以打包环节本身没有缺陷：**把候选池限制对，取前 32 就等于全取。**
0.5848 → 0.9474 的差距 **100% 来自候选池被稀释到 578 轮**，
其中约 547 轮属于无关会话。

### 13.2 这不是"找不到会话"，是"没有按会话分配预算"

当前打包已经触达 **89.8%** 的 gold 会话——正确会话基本都进了包，
只是每个会话只分到少数几轮，被大量无关会话的轮次挤薄。

因此存在一个比完整自顶向下重构小得多、却能拿到大部分收益的改动：
**打包预算按会话分配，而不是全局按 fused_score 排序**。
先选 top-k 会话（k≈2–3，因为平均 gold 会话数 1.35），再在会话内填满。

### 13.3 对自顶向下方案的判定

13.1 证明了粗层选择就是瓶颈所在，而 367 张会话路由卡正是为此而建、且从未被使用（§12.1）。
方案成立，但前置条件仍然是硬的：

| 前置 | 现状 | 依据 |
|---|---|---|
| 路由卡需要向量 | 全图节点向量 = 0 | §7.2⑦ |
| 路由卡文本需可被问题匹配 | 是拼接的三元组汤、带重复 | §9.2 |
| 摘要需能跨会话区分 | LoCoMo 可分性≈0（0.475–0.586 vs 0.495–0.608） | §9.1 |
| 层级边需方向语义 | `include_inverse=True`，上下行不可分 | §12.1 |
| 关系代数需可 join 的 fact | predicate 92.1% 唯一 | §8.1 |

**推进顺序应当是「先按会话分配预算（T0，零索引改动），再做自顶向下路由」**：
前者立刻兑现 13.2 的一部分并给出真实上限，后者才需要付 §9.2 的重建。

### 13.4 scene summary 作为证据的正确理由

preference 仅 30 题、已 0.867、离 0.90 差 1.0 题——不值得为它单独设计证据类型。
但摘要在**打包 token 效率**上有独立价值：打包是 token 受限的
（`max_evidence_turns=32` / `max_evidence_tokens=5000`），而 32→48→64 单调 +5.5pp
已证明"装得下更多"本身就是收益。一条摘要 ~48 token 覆盖整个场景，
一条原始轮次 100+ token 只覆盖一句话。

因此摘要应作为**分层证据包的粗粒度层**：先用摘要铺覆盖面，
只在算子需要精确值（数字、日期、序数）时下钻到原始轮次。
这与 13.2 的按会话分配是互补的——一个解决"选哪些会话"，一个解决"每个会话花多少 token"。

## 14. 会话路由的真实上限，与最终架构

### 14.1 词法路由封顶在 ~70%

四种会话打分函数，指标为「全部 gold 会话都进入 top-k」的比例（761 题，平均 26.6 会话/memory）：

| router | top-1 | top-2 | top-3 | top-5 | top-8 |
|---|---:|---:|---:|---:|---:|
| sum_idf（无长度归一） | 52.4% | 64.0% | 67.4% | 75.0% | 80.8% |
| **bm25（带长度归一）** | **54.1%** | **65.4%** | **70.0%** | 74.8% | **82.1%** |
| rare（df≤5% + bm25） | 43.1% | 52.2% | 54.5% | 57.7% | 62.5% |
| rare_bin（df≤25% 二值） | 50.3% | 61.2% | 65.2% | 70.2% | 76.0% |

接上现有打分器取前 32：top-2 → 0.6307，top-3 → 0.6505，top-5 → **0.6859**
（实际 0.5848，oracle 0.9474）。

**BM25 只比朴素 sum_idf 好 2.6pp，词法路由整体封顶在 top-3 ≈ 70% / top-8 ≈ 82%，
对应 all_hit 上限约 0.69。**

### 14.2 一个被推翻的假设：稀有词不迁移

§10.2 测出稀有词重合在 **gold 会话对**上有 14–21x 分离度，我据此推测它也适合
query→会话路由。**实测相反：`rare` 是四个里最差的（top-3 54.5% vs bm25 70.0%）。**

原因是两个任务的输入不同：会话-会话比较的双方都是长文档，共享稀有词表示同指；
而 query 只有十几个词，其中大部分在该 memory 里并不稀有（"how many"、"what did I"
加少数内容词），把 df≤5% 之外的词全部丢弃等于丢掉 query 的大半信号。
**稀有词适合连接两个文档，不适合把一个短查询路由到文档。**

这同时说明：**「最好的方案」确实需要那次 T2 重建**——词法已经到顶，
要逼近 oracle 必须有比词袋更好的会话表示，即真句子摘要 + 节点向量（§9.2 的 B1/B2）。

### 14.3 h0 与 h10 的 13pp 缺口：结案

`LEGACY_SESSION_FANOUT = 8`，而本节 top-8 覆盖 82.1%，h0 实测 0.7004。
**h0 赢 h10 的 13pp，就是「h0 做了会话路由，h10 没做」。**
此前把这 13pp 依次归因于融合权重（W 系列最大 0.79pp）、图过度加权（1.3pp）、
`_rank_pack` 逻辑（§13 证明其无缺陷），全部证否。真正的差异是**检索单元**。

### 14.4 最终架构：算子感知的会话路由

词法封顶 70% 的根因是它对所有题型一视同仁，而四类题需要的会话集**结构上就不同**：

| 题型 | 需要的会话集 | 可用的结构信号 | 现状 |
|---|---|---|---|
| knowledge-update | 提到该实体的**最新**会话 | `observed_at` **100% 覆盖**，给出会话全序 | 未使用 |
| temporal-reasoning | 落在日期区间内的会话 | 同上 + `time_interval` | 未使用 |
| multi-session 聚合 | 提到该属性类的**全部**会话（不是 top-k） | 属性键 | 键已退化 |
| preference | 偏好陈述最密的**单个**会话 | 场景摘要 | 摘要是三元组汤 |

**A 层应当是算子感知的**：query IR 已经解析出算子与时间/主体约束，
但这些约束目前只用于绑定 fact，从未用于**筛选会话**。
`observed_at` 是 100% 覆盖且完全未被使用的结构信号——
knowledge-update（78 题）问的正是「现在的值」，而"最新"在会话序上是一个确定性查询，
不需要任何相似度。

三层架构（§13.3 已述）保持不变，但 A 层的输入从「一个词法分」变成
「词法分 + 稠密分 + 算子决定的硬约束」，其中硬约束这一路是当前完全空白、
且数据已经就绪的部分。

### 14.5 修正的优先级

| 层 | 动作 | 依赖 |
|---|---|---|
| **A-1** | 会话级 BM25 打分 + 按会话分配打包预算 | 无（T0）；已测 top-5 → 0.6859 |
| **A-2** | 算子感知的会话硬约束（先做 `observed_at` 的"最新"路径） | 无（数据 100% 就绪） |
| **A-3** | 真句子摘要 + 节点向量 → 会话稠密路由 | **T2 重建**；14.2 证明词法已到顶 |
| B | 属性键 → 关系代数可 join | 同一次 T2 |
| C | span 回推 + 分层证据包 | 投影层（T1） |

**A-1/A-2 不需要重建且共同决定 A 层能到多远；A-3 是逼近 oracle 的必要条件。**
B 与 C 在 A 层仍锁在 0.69 时做，收益会被上游截断。

## 15. 系统状态汇总（2026-08-07 收盘）

基线：B1_quote 图 / 10 memory / 761 gold 标注题 / h10 profile / 证据预算 32。
起点 `all_hit = 0.5848`，h0 legacy 参照 `0.7004`，oracle 会话路由 `0.9474`。

### 15.1 确认有收益（全部零索引改动、零重建）

| 项 | 收益 | 配对显著性 | 状态 |
|---|---:|---|---|
| 证据预算 32→48→64 | **+5.5pp** | 单调 | 未合入 |
| 会话路由 top-8 | +1.58pp | 22胜10负 p=0.050 | 已实现，未设默认 |
| 跳数衰减 `graph_hop_decay=0.3` | +0.92pp | 14胜7负 p=0.19 | **已合入默认** |
| exact 权重 1.2→1.0（W5） | +0.79pp | 6胜0负 **p=0.031** | 未合入 |
| 逐步剪枝 `expansion_beam=2` | +0.39pp，走边 −16% | 3胜0负 | **已合入默认** |

合计约 **+9pp**，叠加后预计 0.67–0.68，**仍低于 h0 的 0.7004**。

### 15.2 确认无收益（已证否，不应再投入）

`shared_referent` 跨会话边（+0.13pp，13,386 条，走得最多的关系）、
实体合并（+0.2pp）、稀有词同层扩展（两子集三阈值全输平铺）、
`operand` 无界项封顶（±0.26pp）、图权重下调（−0.26pp）、
按会话配额（+0.26pp，加了略差）、beam 加宽至 16/32（与 off 逐位相同）、
hops 2→3（覆盖率零增益、6 倍代价）。

### 15.3 属性类枚举验证（本节新增）

用 52 类固定枚举对现有 4,669 条 fact 重新分类：

| 键 | 键数 | 单例 | 跨≥2会话 | 可数集合 |
|---|---:|---:|---:|---:|
| 基线 (owner, collection_key) | 1,274 | 75.6% | 5.0% | 311 |
| **枚举 (owner, class)** | 1,518 | 67.3% | **12.9%** | **497** |
| 枚举 (class) 单独 | 290 | 25.9% | 60.7% | 215 |

跨会话复用 **5.0% → 12.9%（2.6x）**，可数集合 **+60%**。
对比 V5.7 模型自造类别的 **6.3% 复用**，封闭词表的机制成立。

**⚠️ 未分离的混淆：54.6% 的 fact 落入 `other`**，而 `(owner, other)` 是个大桶，
会人为压低单例率。真实语义分组的贡献未知。词表是临时拟定的，
需要一份**从语料导出**的分类法（聚类后命名）才能定论。**此结果是下界。**

### 15.4 已识别、未修复的缺陷

1. 证据 span **100% 退化为整轮**（5,221/5,221）。
2. 图节点**零向量**（9,329 条 embedding 全为 turn + predicate）。
3. 分层图在召回侧**完全未用**：427 张路由卡不是入口，`refines_to` 无方向语义。
4. 状态链断裂：125/210 孤立，其中 **98 个 `predicate` 为空**；`temporal_before` 全图 **8 条**。
5. **`observed_at` 100% 覆盖但从未使用**——knowledge-update 的"取最新"是确定性查询。
6. 死旋钮：`max_llm_reranks` 零消费者；`max_iterations` 仅用于上报。
7. `A5_free_predicate.json` / `A6_out3072.json` 加载即抛 `TypeError`。

### 15.5 写报告必须交代的口径

- **`lme_multi_session` 全程 n=1**：所有分层结论实质只在 LoCoMo 上成立。
- **「gold 100% 在候选池」与「94.95%」是两个口径**（前者含会话扇出带进的零分轮次），以 **94.95%** 为准。
- **cat4 占 55% 且从不需要第二会话**（`V5_8_CROSS_SESSION_KEY.md`）：
  跨会话工作只在 cat1 的 142 题（18.7%）上计分。
- **测量有约 1 题的运行间抖动**；**进程内首臂的延迟数不可用**（预热假象，
  曾据此错报剪枝 −53% 延迟，反序复测证伪）。

### 15.6 七次被证否的假设（报告的主体应是这部分）

| 假设 | 依据 | 实测 |
|---|---|---|
| 跨会话边不足 | 21,117 条边 0 条跨会话 | 加 13,386 条，+0.13pp |
| harness 被饿死 | h8/h9 低于 legacy 28pp | 给了边不动——是**撑死**非饿死 |
| 图候选挤占槽位 | 纯图占包 25.9%、gold 产出 0.4% | 衰减+剪枝 +1.3pp |
| `operand` 无界项失衡 | 纯图 fused 2.41 vs 词法 1.24 | 封顶/去掉 ±0.26pp |
| `_rank_pack` 有缺陷 | 池 94.95% → 包 62.9% | oracle 证明**打包零损失** |
| 稀有词可做同层扩展 | 会话对上 14–21x 分离 | 两子集三阈值全输平铺 |
| 会话路由可拿 +19.3pp | 单会话 oracle 1.0 vs 实际 0.6895 | **+1.58pp** |

**统一的教训有两条。**
其一：oracle 的收益来自「**候选池比预算还小**」（1.35 会话 ≈ 31 轮 ≈ 32 预算，等于全取），
不是"选对会话"。路由 top-3 是 93 轮抢 32 个，超订 3 倍，排序问题原封不动——
这解释了为什么所有召回侧改动都只值 1pp 量级。
其二：四个失败的跨会话信号（实体合并、shared_referent、摘要 embedding 配对、
稀有词扩展）**全部是问题无关的**，全部输给问题条件的词法分。
**结构应做过滤，问题应做排序，不要让结构排序。**
稀有词在「给定 gold 锚点」下是 92.3% 召回，但那是**过滤器指标**；
用作排序器（约 10 个邻居取 top-2）即失效——两个指标不可混用。

## 附：参数总数

字段数为 `dataclasses.fields()` 实测：

| 阶段 | 位置 | 字段数 | 其中真正可消融 |
|---|---|---:|---|
| 构建 | `ModelConfig` | 32 | 26（扣除 4 个 endpoint/model-id、`thinking_enabled`（被 `__post_init__` 强制 False）、`max_concurrency`） |
| 构建 | `SceneConfig` | 8 | 7（`refine_batch_size` 只影响吞吐） |
| 构建 | `CoarsenConfig` | 9 | 9 |
| 构建 | `EdgeConfig` | 16 | 16（含 1 个 per-relation mapping） |
| 召回 | `QueryBudget` | 16 | 14（`max_llm_reranks`、`max_iterations` 无效） |
| 召回 | 硬编码常量 | 20 个具名常量 + 约 20 个内联权重系数 | 全部，但需先重构 |
| 召回 | `ProjectionConfig` | 15 | 15 |
| 作答 | `AnswerConfig` | 6 | 6（另有 10 个 CLI flag） |
| 离散 profile | build b0–b5 / navigator n0–n5 / harness h0–h10 / graph g0–g5 | 4 组 | — |

构建侧合计 65 个字段、58 个可消融项；这些**全部属于 T2**，每个臂约 2 小时。
