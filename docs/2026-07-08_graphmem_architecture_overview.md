# GraphMem 项目整体思路（架构总览）

> 本文是 **当前阶段** 的统一设计文档，汇总构建、索引、图结构、检索与实验结论。  
> 细分专题见文末「相关文档」；英文原始 idea 见 [`idea.md`](./idea.md)。

---

## 0. 一句话

> **Leaf 存原文证据，Root 存无损对话导航，LLM JSON 只做结构化索引；构建时把相对时间写进绝对日期，检索以 semantic 为主干，typed / graph / fusion 做高置信扩池，作答前按时间序拼 context 并给锚点日期。**

GraphMem 要解决的不是「有没有图」，而是：**在长对话记忆场景下，如何用可接受的构建 token 成本，把足够完整的证据写进图里，并在检索时真正用到图结构与多路信号，而不是退化成纯 embedding top-k；最后在 QA 层把 timeline 和日期锚点交给模型，减少相对时间的二次推理错。**

---

## 1. 问题与诊断（我们为什么要改）

### 1.1 Baseline 的三层瓶颈

| 层面 | 旧 baseline 问题 | 后果 |
| --- | --- | --- |
| **构建** | `user_only` leaf、320 token 摘要、截断/parse 失败仍写入 | 金标证据未入库；root summary 有损 |
| **图** | 边建在有损 summary 上；检索几乎不用 leaf-leaf / 多跳 | 构建图的成本与检索收益脱节 |
| **检索** | hybrid 以 embedding 全局排序为主 | 日期/数字/专名题弱；multi-session 难跨 session 聚合 |
| **作答** | 单次 QA，无跨证据算术/去重 | 约一半错题是金标 session 已命中仍算错 |

详见 [`2026-06-30_root_leaf_diagnosis.md`](./2026-06-30_root_leaf_diagnosis.md)。

### 1.2 实验教给我们的事

| 尝试 | 结果 | 教训 |
| --- | --- | --- |
| 纯 PPR 替换 hybrid（`e2e_free`） | 准确率持平，**leaf 召回大降** | 不能扔掉 embedding 保底 |
| build2048 + raw + graph-first | subset50 **64%**，+6pp vs 旧 hybrid | 构建保真 + 图检索有效，但 session 召回可再优化 |
| + fusion 全盘重排（无保护） | **54%**，-10pp | fusion 改 context 会伤 preference / 引发 over-abstain |
| typed 边替代 keyword + 密图 PPR | multi-session 13 题检索偏航 | **噪声边**比「有没有 typed 边」更致命 |

**核心共识**：构建、图、检索、作答要分层优化；任何一层「夺权」都可能伤到其他层已获得的收益。

---

## 2. 总体架构

```text
                    ┌──────────────────────────────────────────────────┐
  Haystack          │                  构建（Build）                    │
  sessions    ───►  │  Leaf(raw) → Session Root(无损)                  │
                    │  → LLM 索引(JSON) + leaf enrichment              │
                    │     └─ compact_facts 时间归一化 (yesterday→日期)  │
                    │  → 图边(稀疏) → Embedding                        │
                    └──────────────────────┬───────────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────────┐
  Question    ───►  │                  检索（Retrieve）                 │
                    │  Query anchors                                   │
                    │  → Root 初排 (embed + typed)                     │
                    │  → Leaf 融合 (semantic + BM25 + entity, RRF)     │
                    │  → Graph-first PPR 扩池（稀疏图上游走）           │
                    │  → Top-K leaf（按相关性选，非时间序）             │
                    └──────────────────────┬───────────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────────┐
                    │              上下文组装（Context Assembly）         │
                    │  summaries + leaves 按 session/turn 升序排列        │
                    │  question_date + last session date 作 timeline 锚点│
                    └──────────────────────┬───────────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────────┐
                    │                  作答（QA）                         │
                    │  LLM 读有序 context + 锚点 → prediction             │
                    │  (可选) note extraction / compute_plan             │
                    └──────────────────────────────────────────────────┘
```

**四层分工**：Build 保真 + 预解析时间；Retrieve 按相关性找证据；Context Assembly 把证据排成 timeline；QA 做最终推理。检索层不改排序逻辑，时间序只在进 prompt 前施加。

### 2.1 设计原则

1. **证据与索引分离**：答题永远以 `leaf.raw_text` 为准；摘要、`compact_facts` 与 anchor 只服务检索与连边。
2. **无损 Root**：`SummaryNode.summary` 存完整子节点原文拼接；LLM 产出的 JSON **不得**替代 summary 正文。
3. **构建期消歧时间**：相对时间（yesterday / last week）在写 `compact_facts` 时解析为绝对日期，避免 QA 二次推算。
4. **Semantic 主干，Structured 辅助**：embedding 负责保底与广覆盖；typed / keyword / graph 只做 boost 或扩池。
5. **图要稀疏且可解释**：宁可少连，不可乱连；跨主题桥是 PPR 偏航的主因。
6. **Protected 融合**：semantic top-K 不可被 BM25/entity 完全洗掉。
7. **QA 时间序与锚点**：检索按相关性选 leaf，拼 context 时 oldest→newest，并注入 `question_date` 与 retrieved 最新 session date。

---

## 3. 写入形态（Build Schema）

### 3.1 节点层级

```text
Leaf (turn 级)
  raw_text          ← 无损对话原文（user + assistant）
  compact_facts     ← 方案2：构建时 LLM 写的短事实（检索索引）
  anchor_terms      ← 实体/时间/动作/状态/关键词（typed 检索与连边）
  embedding         ← 对 retrieval_text 编码

Root (session 级, SummaryNode)
  summary           ← 无损：该 session 下所有 leaf 原文拼接
  parsed_summary    ← LLM JSON，仅索引用
  anchor_terms      ← 从 JSON + 正文抽取的结构化锚点
  retrieval_text    ← 供 embedding / BM25 的检索文本
  embedding
```

### 3.2 关键构建策略

| 策略 | 配置 | 作用 |
| --- | --- | --- |
| Raw leaf | `build_leaf_text=raw`, `retrieval_leaf_text=raw` | 保留 assistant 侧证据（preference/assistant 题） |
| 大摘要预算 | `session_summary_max_tokens=2048` | 降低截断与 parse 失败 |
| 跳过摘要前压缩 | raw 变体自动 `skip_compression_for_raw_build` | 避免 LLMLingua 压掉数字/日期 |
| 无损 Root | `enable_lossless_root_summary=True`（默认） | 截断/parse 失败不再让 root 退化 |
| Leaf 富化（方案2） | `enable_leaf_enrichment=True` | 构建 session 摘要时顺带写 per-leaf `facts/keywords` |
| **构建期时间归一化** | leaf enrichment prompt + `_normalize_temporal_compact_facts` | 写 `compact_facts` 时将 yesterday/last week 等解析为 YYYY-MM-DD |
| 截断恢复 | `_summarize_job` 升 token / 对半拆分 merge | 尽量产出可用 JSON 索引 |

构建细节与 graph-first 首批实验：[`2026-07-06_graph_first_build_improvements.md`](./2026-07-06_graph_first_build_improvements.md)。

#### 3.2.1 构建期时间归一化（2026-07-10）

LoCoMo temporal 错题在 recall 已高时，主要卡在 QA 对相对时间的二次推理。因此在 **leaf enrichment 写 `compact_facts` 时** 做 Mem0 风格的 TEMPORAL_EXTRACTION：

```text
Session date: 2023/07/15
Child raw: "I attended the workshop yesterday."
         ↓  LLM leaf enrichment（strict prompt）
compact_facts: ["Attended workshop on 2023-07-14"]
         ↓  确定性后处理 _normalize_temporal_compact_facts
retrieval_text / QA context 均可见绝对日期
```

| 环节 | 实现 | 说明 |
| --- | --- | --- |
| Prompt | `_time_normalization_rule(..., for_leaf_facts=True)` | leaf `f[]` 必须以 YYYY-MM-DD 为主时间表述 |
| 后处理 | `_normalize_temporal_compact_facts` | 括号日期提升；yesterday/today/last week 等确定性补全 |
| 缓存 | `temporal_normalization_version=1` | 改 prompt/后处理需新 output-dir，勿 resume 旧 cache |

**不改检索栈**：归一化结果写入 `compact_facts` → `retrieval_text`，embedding/BM25/typed 自然受益。

### 3.3 图边（Root 级）

模块：`src/graphmem_demo/root_graph_edges.py`

```text
必留（信号边）                    慎用（易噪声）
─────────────────                ─────────────────
temporal_neighbor   时间邻接       entity_neighbor   需专有实体或≥2 泛词
corpus keyword      主题 overlap   update_neighbor   需≥2 共享 action + 合格实体
semantic_neighbor   嵌入相似       time/state/event  分数门槛 + 语义支撑
typed keyword       anchor 对齐
```

**当前策略（2026-07-08）**：

- Typed 边与 corpus keyword **叠加**，不再 `prefer_typed_edges` 替换 keyword。
- 建边后 `prune_noisy_root_edges`：泛 entity 过滤、embedding cosine 支撑、每 root typed 边 cap、分数底线。
- PPR 传播：keyword/semantic 权重大于 entity/update。

`enable_typed_root_edges` 在开启 `graph-first` 或 `typed_retrieval` 时自动生效（见 `_effective_typed_root_edges`）。

---

## 4. 检索栈（Retrieve Stack）

检索是 **多层正交能力叠加**，而非单一路径：

```text
Query
  │
  ├─[A] Query anchors          typed_retrieval.query_anchor_terms()
  │
  ├─[B] Root 初排               embedding × (1-α) + typed_overlap × α
  │
  ├─[C] Leaf 三路融合           semantic + BM25 + typed entity  (RRF)
  │       └─ protected fusion   semantic top-10 保底
  │
  ├─[D] Graph-first PPR         seeds → 稀疏图游走 → 扩 candidate pool
  │       └─ session_coverage   multi-session 至少覆盖 N 个 session
  │
  └─[E] Top-K leaf → context_text（oldest→newest）→ QA
```

### 4.1 Typed 检索

模块：`src/graphmem_demo/typed_retrieval.py`

- Query 解析为与写入时同 schema 的 anchor：`entities / times / quantities / actions / state_phrases / keywords`
- Root 初排与 PPR personalization：`rank_roots_hybrid()`、`personalization_scores()`
- Fusion 的 entity 路升级为完整 typed overlap（与 root/leaf 统一 scorer）

配置：`enable_typed_retrieval=True`（默认），`typed_retrieval_embedding_blend=0.55`。

### 4.2 Fusion 检索

模块：`src/graphmem_demo/fusion_retrieval.py`  
文档：[`2026-07-06_fusion_retrieval.md`](./2026-07-06_fusion_retrieval.md)

- 三路并行：semantic cosine、BM25 keyword、entity/typed overlap
- 默认 RRF 融合；query-adaptive 权重（日期题加重 keyword 等）
- **Protected fusion**（`enable_protected_fusion=True`）：先锁 semantic top-10，再 RRF 重排其余 → 避免 fusion 全盘夺权

### 4.3 Graph-First 检索

模块：`src/graphmem_demo/graph_retrieval.py` → `graph_first_retrieve()`

- Embedding 只选 **PPR seeds**（root + leaf）
- 候选池 = seeds ∪ global_leaf_top_k ∪ PPR 高分叶
- `blended_score = (1-α)·PPR + α·embedding`，默认 α=0.25
- `session_coverage≥2`：multi-session 题避免单 session 垄断
- **绝不**放弃 hybrid 的 global leaf 池（区别于失败的 `e2e_free`）

### 4.4 推荐组合（当前最佳实践）

```bash
python scripts/run_token_demo.py \
  --llm-local \
  --data data/longmemeval_s_subset50_balanced.json \
  --output-dir runs/subset50_typed_sparse_gf \
  --variants direct_session_k16_compact_graphmem \
  --enable-graph-first-retrieval \
  --enable-fusion-retrieval \
  --question-type all
```

注意：

- 换构建策略后 **使用新 output-dir**，勿 `--resume` 旧 memory cache（fingerprint 含 `keyword_edge_version`、`root_graph_edge_policy_version`、`temporal_normalization_version` 等）。
- 本地 judge：`export DEEPSEEK_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507`。

### 4.5 QA 上下文时间序 + 锚点日期（2026-07-10）

检索仍按相关性选 leaf，但 **拼进 prompt 前** 按时间升序排列，并注入 timeline 锚点：

```text
Question date: 2023/07/20
These conversations took place around 2023-06-15.   ← retrieved context 最新 session_date
Question: When did I finish the book?

Retrieved memory evidence:
  [Session s1 | 2023/05/01 | turn 0] ...   ← oldest first
  [Session s2 | 2023/06/15 | turn 3] ...
```

| 函数 | 作用 |
| --- | --- |
| `_sort_leaves_chronologically` / `_sort_summaries_chronologically` | `_context_text` 展示层排序，不影响 `_fit_context_budget` 裁剪逻辑 |
| `_latest_reference_date` / `_reference_date_from_retrieval` | 取 retrieved context 最新 session 日期作锚点 |
| `_answer_messages(..., reference_date=...)` | user prompt 加 `These conversations took place around <date>.` |

与 mem_baselines 一致：**oldest→newest** timeline + **question_date / last session date** 双锚点，减轻 LoCoMo 相对时间推理错。

---

## 5. 数据流：一道题的一生

以 LongMemEval-S multi-session 题为例：

1. **构建**：对该题完整 haystack（~40–50 sessions）逐 session 摘要 → 写 root（无损正文 + JSON 索引）→ 写 leaf（raw + facts/keywords）→ 建 root 图边 → embed。
2. **检索**：question → query anchors → root 排序 → fusion 排 leaf → graph-first PPR 扩池 → 取 top-K leaf → `_context_text` **按 session/turn 时间升序**拼 context。
3. **作答**：LLM 读 context（含 reference date 锚点）→ prediction。
4. **评测**：`auto_eval.jsonl`（strict/relaxed）；`retrieval_results.jsonl`（session/leaf 命中）；`question_stats.jsonl`（构建与检索指标）。

每题 **独立构建**（per-question memory graph），非全局共享图。

---

## 6. 失败模式与归因

| 失败类型 | 典型症状 | 主要层级 | 应对方向 |
| --- | --- | --- | --- |
| `retrieval_miss` | 金标 session 未进 context | 检索/图 | 恢复 keyword、稀疏 typed 边、提高 session_coverage |
| `answer_over_abstain` | session 全中却称「证据不足」 | QA | 抑制拒答 prompt；算术题用 compute_plan |
| `answer_reasoning` | 证据在但跨 session 算错/数错/日期错 | QA / 构建 | 时间归一化 compact_facts；context 时间序 + reference date；compute_plan |
| `build_summary_degraded` | 截断/parse 导致索引残缺 | 构建 | 无损 root、2048 token、跳过摘要前压缩 |
| fusion 副作用 | session 中但 leaf 换一批 | 检索 | protected fusion；勿让 BM25/entity 全盘重排 |

Multi-session 13 题深析表明：**session 全命中 ≠ 答对**；大量错题在 QA 聚合层，但 `6d550036` 类题说明 **噪声边 + PPR 偏航** 仍必须先修。

---

## 7. 代码地图

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| Pipeline 编排 | `src/graphmem_demo/pipeline.py` | 构建、检索、QA、配置、变体 |
| Root 图边 | `src/graphmem_demo/root_graph_edges.py` | 高置信建边 + 噪声剪枝 |
| Graph 检索 | `src/graphmem_demo/graph_retrieval.py` | PPR、graph-first、边权 |
| Typed 检索 | `src/graphmem_demo/typed_retrieval.py` | Query/root anchor、hybrid 初排 |
| Fusion 检索 | `src/graphmem_demo/fusion_retrieval.py` | 三路 RRF、protected fusion |
| 数据模型 | `src/graphmem_demo/models.py` | LeafNode、SummaryNode、GraphEdge |
| 实验入口 | `scripts/run_token_demo.py` | CLI、subset 跑批 |
| 评测 | `scripts/evaluate_answers.py` | LLM judge |
| 阶段归因 | `scripts/analyze_stage_audit.py` | 检索 vs 推理 vs 构建 |

测试：

- `test/test_token_demo.py` — 构建、变体、graph-first 邻域召回
- `test/test_typed_retrieval.py` — typed anchor 与排序
- `test/test_root_graph_edges.py` — 噪声剪枝、additive keyword

---

## 8. 实验里程碑（subset50）

| Run | Strict Acc | Session 全命中 | 备注 |
| --- | :---: | :---: | --- |
| baseline `temp0_summary4x` | 58% | 92% | 旧 hybrid |
| `build2048_raw_graphfirst` | **64%** | 82% | 当前全集最佳 |
| `raw_nocompress_fusion_gf` | 54% | 82% | fusion 无保护退步 |
| `typed_graph_gf`（13 题） | — | 76.9% | typed 密图 + keyword 清零 |
| `lossless_leaf_enrich_gf`（13 题） | — | 84.6% | 无损 root + leaf 富化 |
| `typed_sparse_gf`（待跑） | — | — | 稀疏 typed + additive keyword |

---

## 9. 后续方向（按优先级）

1. **验证稀疏图策略**：`typed_sparse` run vs `typed_graph_gf` / `build2048`，看 multi-session 检索是否回升。
2. **Protected fusion + 稀疏图联合跑 full 50**，目标 ≥64% 且 session 召回 ≥85%。
3. **QA 层**：multi-session 计数/金额题的跨 session 聚合；`enable_compute_plan` 算术门控；降低 over-abstain。
4. ~~**构建对齐**：日期 ISO 归一化~~ → **已实现**构建期 compact_facts 时间归一化 + QA reference date（2026-07-10）。
5. **Full500**：在 longmemeval_s_cleaned 500 题上验证泛化。

---

## 10. 相关文档

| 文档 | 内容 |
| --- | --- |
| [`idea.md`](./idea.md) | 英文原始研究 idea、HMG 分层图记忆 |
| [`2026-06-30_root_leaf_diagnosis.md`](./2026-06-30_root_leaf_diagnosis.md) | 基线诊断、root-leaf 边无效、推理瓶颈 |
| [`2026-07-06_graph_first_build_improvements.md`](./2026-07-06_graph_first_build_improvements.md) | raw/2048/截断修复 + graph-first 检索 |
| [`2026-07-06_fusion_retrieval.md`](./2026-07-06_fusion_retrieval.md) | 三路 fusion + protected fusion |
| [`report01.md`](./report01.md) / [`report02.md`](./report02.md) | 早期实验报告 |
| [`llm_budget_expansion_directions.md`](./llm_budget_expansion_directions.md) | Token 预算扩展方向 |

---

## 11. 一句话（给新接手的人）

> 先把**完整证据**写进 Leaf 和无损 Root，构建时把**相对时间解析成绝对日期**写进 `compact_facts`；再用**稀疏、可解释的图边**和**多路检索**把对的 session/leaf 选出来（按相关性，不按时间）；进 prompt 前把 context **按时间序排列**并给 **question_date / last session date** 锚点。embedding 永远是保底，typed 和 graph 只帮忙、不夺权。检索层保证「不把对的证据换走或游丢」，QA 层保证「timeline 清晰、少做日期心算」。
