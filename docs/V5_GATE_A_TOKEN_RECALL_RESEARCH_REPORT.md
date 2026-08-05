# GraphMem V5 Gate A：Token、图特征与精确证据召回诊断

## 技术摘要

这次冻结的 Qwen3-30B retrieval-only run 已覆盖 LongMemEval 100 题和 LoCoMo 100 题。核心结论不是“图太小”，而是**图构建极贵、图与候选极宽，但精确证据导航仍然不足**：27,623,686 个 backbone token 中，99.41% 用于构建，只有 0.59% 用于 query planner；LongMemEval 官方 session all-hit 为 0.82，但人工 gold-turn all-hit 只有 0.71；LoCoMo session all-hit 为 0.69，官方 evidence turn all-hit 只有 0.05。

LongMemEval 可以、也应该使用精确 turn-level recall。当前 100 题已有 217 条双阶段复核标注，按 `(question_id, session_id, zero-based turn_index)` 与最终 packed source turn 精确匹配。该口径显示 29 道题未取齐 gold turns：15 道在 session routing 阶段已经缺失，3 道进入了正确 session 但 gold turn 未成为候选，11 道 gold turn 曾成为候选但在最终打包时被丢弃。

LoCoMo 的问题更集中：232 条官方 evidence turns 中仅 13 条进入最终 context。图构建本身并非主要缺口——约 89.2% 的官方 evidence turns 已有 source-bound frame——而是 frame/turn 的 query-time candidate generation 与排序没有把它们暴露出来。95 道 turn all-hit 失败中，31 道属于 session routing miss，64 道属于 within-session candidate miss，未观察到“候选齐全后单纯因打包丢失”的 LoCoMo 失败。

以下“特征价值”是 trace attribution，不是因果消融。可观测上最值得保留的是 lossless SourceTurn、frame→source provenance、BM25/exact/dense 多路召回、semantic-turn、scene-window 和低成本 planner；最应优先缩减的是全量 session LLM extraction、全局 identity consolidation、超宽 inverted/structured 候选以及低精度的 source additions。最终删除决策仍须在固定 navigator 下做 paired ablation。

## Token 几乎全部消耗在构建，而不是导航

| Benchmark / stage | Calls | Input tokens | Output tokens | Total tokens | 全 run 占比 |
|---|---:|---:|---:|---:|---:|
| LongMemEval session extraction | 4,736 | 11,848,631 | 10,977,393 | 22,826,024 | 82.63% |
| LongMemEval identity consolidation | 100 | 3,074,851 | 204,280 | 3,279,131 | 11.87% |
| LoCoMo session extraction | 272 | 408,272 | 582,528 | 990,800 | 3.59% |
| LoCoMo identity consolidation | 10 | 343,005 | 20,480 | 363,485 | 1.32% |
| 两个 benchmark 的 query planner | 148 | 152,201 | 12,045 | 164,246 | 0.59% |

合并后，session extraction 占 86.22%，identity consolidation 占 13.19%，query planner 仅占 0.59%。输入共 15,826,960 tokens（57.29%），输出共 11,796,726 tokens（42.71%）；仅 session extraction 的输出就占总 token 的 41.85%。因此，先优化 query prompt 不会显著改变总成本。

每个独立 memory 的平均构建成本：

- LongMemEval：100 个 memory，平均 261,052 tokens/memory，约 5,546 tokens/session、530 tokens/source turn。
- LoCoMo：10 个共享 conversation memory，平均 135,429 tokens/memory，约 4,989 tokens/session、232 tokens/source turn。

输出上限正在制造明显浪费和质量风险。LongMemEval 4,736 次 session extraction 中 3,241 次以 `finish_reason=length` 结束，3,238 次记录了包括 `partial_json_salvaged` 在内的 parse error，分别约占 68.43% 和 68.37%。LongMemEval 的 100 次 identity consolidation 有 99 次触顶；LoCoMo 10 次 identity consolidation 全部触顶。截断率在成功与失败问题之间接近，因而它不是本 run 内 all-hit 差异的单一解释，但它说明大量 token 被用于生成无法完整闭合的 JSON。

## 图很大，但用于跨证据导航的边很少

LongMemEval 100 个图包含 145,503 个节点和 207,196 条边：49,443 个 SourceTurn、61,515 个 RoleFrame、29,809 个 EvidenceGroup、4,736 个 RoutingCard。summary/turn 比为 1.94，即每个原始 turn 额外产生接近两个 summary-like 节点。

边结构高度集中在 provenance 和局部连续性：`source` 62,850、`routing_contains` 61,515、`next_turn` 44,707、`dialogue_pair` 25,185，四类合计占 93.75%。真正提供跨事件或 typed shortcut 的边较少：`semantic_neighbor` 720、`same_event` 923、`reference` 95、`state_transition` 1,214、`temporal_endpoint` 836、`collection_member` 9,151。

实际 retrieval trace 中的 1,438 次 graph expansions 更偏 provenance closure：1,260 次 `source`、167 次 `collection_member`、8 次 `same_event`、3 次 `temporal_endpoint`。直接抵达 packed gold turn 的 51 次 expansion 全部来自 `source`。这不证明其它关系无用——它们可能先找 frame，再通过 source 回到 turn——但说明当前 navigator 主要把图当作“frame 到原文的回链”，还没有充分利用跨 session typed navigation。

LoCoMo 共享图包含 11,598 节点和 15,371 条边；其 evidence turn 约 89.2% 已被至少一个 frame 通过 source edge 绑定。因此 LoCoMo 精确 turn 召回低不能简单归因为“构图没有抽取事实”，更可能是 candidate/ranking 没有把已存在的 frame 或 source turn带到查询前沿。

## Session 指标高估了精确证据导航能力

| Stratum（每类 50 题） | Session all-hit | Packed-session all-hit | Candidate turn all-hit | Final turn all-hit | Turn mean recall | Turn mean precision |
|---|---:|---:|---:|---:|---:|---:|
| LME multi-session | 0.78 | 0.74 | 0.84 | 0.60 | 0.783 | 0.099 |
| LME temporal | 0.86 | 0.84 | 0.86 | 0.82 | 0.886 | 0.075 |
| LoCoMo Cat1 multi-hop | 0.46 | 0.44 | 0.00 | 0.00 | 0.015 | 0.0038 |
| LoCoMo Cat2 temporal | 0.92 | 0.92 | 0.10 | 0.10 | 0.100 | 0.0043 |

`Session all-hit` 使用 legacy `retrieved_session_ids`；`packed-session all-hit` 从最终 context 中的 source turn 反推 session。二者有 0–4 个百分点差异，说明 card/session trace 与最终 evidence context 不是完全同一集合。后续报告应把官方 gold-session recall、packed-session recall 和 exact turn recall 分开呈现。

LongMemEval 的 exact turn 结果是可用的：overall turn all-hit 为 0.71、turn any-hit 为 0.91、等权 mean recall 约 0.834。代价是低 precision：每题平均打包约 21.9 个 unique turns，而 gold turns 通常只有 1–数个，平均 turn precision 约 0.087。

更严格的 span 检查显示，完整人工 gold span 在最终 context 中的 all-hit 仅为 multi-session 0.06、temporal 0.22。这个数不能直接当作“证据无效率”：当前 packer 经常只保留同一 gold turn 的相关子句，而人工 span 可能覆盖更长的充分证据区间。它适合作为压缩审计的保守下界；若要作为主指标，需要把标注升级为“必要最小子 span”或做语义充分性复核。

## All-hit 低的具体原因

### LongMemEval：计数闭包、负证据和最后打包是主要薄弱点

29 道 exact turn all-hit 失败可分为：

- 15 道 session routing miss：gold session 没进入 retrieval session 集合。
- 3 道 within-session candidate miss：session 已命中，但 gold turn 没进入 fine/V4.1 candidate trace。
- 11 道 pack drop：candidate gold turns 已齐，但最终 context 未全部保留。

这些 pack drops 的平均 evidence 仍远低于配置的 10k query target（四类平均仅约 4.1–4.3k rough tokens），所以主要不是 hard token budget 耗尽，而是优先级、去重、closure 或 optional-stage policy 提前淘汰。

按 query kind，LongMemEval `count` 的 turn all-hit 只有 0.40，`temporal_order` 为 0.50，`entity` 为 0.33；`duration` 为 0.87，`aggregate` 为 1.00。按人工 support role，multi-session 的 `negative_scope` reference recall 只有 0.455，显著低于 aggregation member 和 fact（约 0.81）。这符合 all-hit 对集合闭包的要求：只找到一个成员即可 any-hit，但 count、distinct、absence、时间端点需要把所有成员和边界取齐。

### LoCoMo：主要失败发生在正确 session 内的 turn candidate generation

LoCoMo Cat1 的 session any-hit 为 0.90，但 turn any-hit 只有 0.04；Cat2 的 session all-hit 为 0.92，但 turn all-hit 只有 0.10。95 道 turn all-hit 失败中，31 道缺 session，64 道在 session 已命中的情况下仍未把官方 evidence turn 放入候选，没有观察到候选已齐后才被 packer 丢弃的 LoCoMo 案例。

这意味着优先级应从“继续扩大 session recall”转向：

1. 在选中的 session 内对 raw SourceTurn 做强制 exact/BM25/dense rerank；
2. 让命中的 RoleFrame 无条件执行 source closure，再对 source turn 排序；
3. 对 multi-hop/list/count 问题按 entity/role 维护多槽位 coverage，而不是只保留相似度最高的若干 turns；
4. 用官方 evidence turn 做离线诊断，但保持 build/query 在线代码不可访问 gold。

## 哪些特征表现出价值，哪些应优先消融

下列贡献允许重叠，不能相加，也不能单凭这张表宣称因果提升。

| Trace feature | Packed gold refs | Packed outputs | Gold precision | Exclusive packed gold refs | 建议 |
|---|---:|---:|---:|---:|---|
| graph expansion | 51 | 765 | 6.7% | 10 | 保留；source closure 是核心 |
| semantic-turn evidence | 69 | 535 | 12.9% | 8 | 保留并提高 session-conditioned 排名 |
| scene window | 85 | 997 | 8.5% | 5 | 有用但偏宽，扫描窗口大小 |
| late scene window | 73 | 959 | 7.6% | 3 | 与 scene 高重叠，优先做单独消融 |
| planner selected evidence | 36 | 97 | 37.1% | 4 | 高精度且 planner 仅占 0.59% token，保留 |
| capability supplements | 17 | 106 | 16.0% | 2 | 小而有效，保留到 paired ablation |
| reply-bound evidence | 16 | 625 | 2.6% | 2 | 缩小邻域，不宜直接全删 |
| source additions | 7 | 1,491 | 0.47% | 2 | 噪声很高，优先限额/阈值化 |
| answer-bearing source IDs | 4 | 488 | 0.82% | 0 | 与其它通道重叠，优先消融 |

候选通道也显示类似结构：planner-selected source 的 packed gold precision 为 37.1%；coarse lossless exact/dense/BM25 对 gold session card 的覆盖最强；BM25、exact、dense、lossless-semantic-turn 对 turn 有广泛重叠。`sidecar_inverted` 产生 24,000 个候选但没有直接命中 exact gold turn，`role_relation` 224 个候选未进入 packed context，`structured` 8,068 个候选只覆盖 5 个 gold refs 且无 exclusive packed gold。这些是第一批应隔离扫描的通道，但 group/frame 间接贡献仍需通过真实 ablation 判断。

### 应保留的系统不变量

- Lossless SourceTurn 与稳定 provenance：所有路径最终必须能回溯到原始 turn。
- Frame→source closure：当前 graph expansion 中全部直接 gold-turn 收益来自 source relation。
- Dense + BM25 + exact 的多路 seed：各通道高度重叠，但 coarse lossless channels 对 gold sessions 的覆盖稳定。
- Query planner：成本极低且选择精度最高，不应为了省 token 先删除。
- Typed role/time/state 表示：虽然当前 navigator 利用不足，但它们是 temporal、negative scope、collection closure 的必要结构，应该改变构建/导航方式而不是未经消融直接删除。

### 第一批成本优化

1. **取消全量 identity LLM consolidation。** 该阶段占 13.19% 总 token，几乎全部请求触顶。先用 deterministic alias/entity normalization，只有冲突簇和跨 session 高价值候选调用 Qwen refine。
2. **把 session extraction 改为 local-first + selective refine。** session extraction 占 86.22%，且 68% LongMemEval 调用触顶。实体、时间、speaker、turn adjacency、exact keyphrase 和 embedding 可本地生成；LLM 只补 EventFrame/RoleFrame 的歧义字段。
3. **限制输出而不是只压输入。** 输出占总 token 42.71%；用短 schema、字段枚举、每 turn/frame 数上限、分段 JSONL 和 deterministic fallback，避免生成长 JSON 后再 salvage。
4. **减少重复 summary-like 节点。** LongMemEval summary/turn=1.94；应把 RoutingCard、EvidenceGroup、RoleFrame 的文本投影分离，embedding/Neo4j 只存短 routing fields，不为每种索引复制完整描述。
5. **把宽候选变成分层漏斗。** 先用 session-conditioned raw-turn exact/BM25/dense 找少量 turns，再做 source/temporal/dialogue closure；对 source additions、late scene、reply-bound 设置独立预算和贡献日志。

## 推荐消融顺序

1. 固定 navigator，比较 lossless-only、deterministic typed-local、当前 full extraction，验证昂贵 LLM frames 的增量 turn recall。
2. 固定图，逐一关闭 `sidecar_inverted` direct candidates、structured、answer-bearing、late-scene、source-additions；每轮记录 gold candidate recall、post-pack recall 和 visited nodes。
3. 对 scene window、dense K、reply closure 做 `4/8/16` 或窄/中/宽扫描，而不是二元删除。
4. 单独测试 identity consolidation：off、deterministic、ambiguous-only；报告 build token 与 exact turn all-hit 的配对差异。
5. 对 count/list/negative-scope 设置 collection coverage slots，观察 LongMemEval count turn all-hit 能否从 0.40 提升。
6. 对 LoCoMo 在已命中 session 内强制 raw-turn rerank和frame→source closure；这是当前最直接的 64 题修复路径。

优选仍应使用四个 strata 等权 turn/evidence all-hit；当差距不超过一个百分点，再按 build token、visited nodes、evidence tokens 和延迟决胜。

## 指标定义与复现

- Session all-hit：官方 gold sessions 是否全部出现在 legacy `retrieved_session_ids`。
- Packed-session all-hit：官方 gold sessions 是否全部能从最终 packed source turn 反推得到。
- Candidate turn all-hit：gold turns 是否全部出现在 fine ranked IDs、fine/V4.1 channel trace 或 feature-emitted turn candidates 的并集。
- Turn all-hit：gold turns 是否全部出现在去重后的 `packed_source_turn_ids`。
- Turn precision：packed gold turns / unique packed source turns。
- LongMemEval gold turns：100 题、217 refs，人工双阶段复核。
- LoCoMo gold turns：100 题、232 refs，官方 `Dn:m` 映射到 `session_n:turn:(m-1)`；抽样文本已验证一致。

报告只做描述性与诊断性归因。Feature/channel 表是重叠 trace attribution，不是因果结果；删除特征必须经过固定数据、固定模型、固定 navigator、固定 budget 的 paired ablation。

## 验证结论与限制

整体评估：**可供研究使用，但必须携带口径 caveat**。

- 已核对 200 个唯一 question、110 个独立 memory、5,266 个 provider calls、27,623,686 tokens、0 reasoning tokens。
- LongMemEval 217 个人工 refs 和 LoCoMo 232 个官方 refs 均进入日志；LoCoMo 共享图已按 memory_id 回挂，未把 10 图复用误判为 90 次构建缺失。
- Token 总数与冻结 Gate A audit 完全一致。
- Turn ID 为确定性精确匹配；重复 packed IDs 在 precision/recall 前已去重。
- Span full-containment 是严格下界，不代表语义充分性；主指标应使用 turn recall，后续可增加 minimal-span 或 semantic sufficiency review。
- 当前只有一个 full configuration。对“可删除”的结论是消融优先级，不是已证实的无损删除清单。

## 给 GPT Pro 的进一步研究问题

1. 如何以 deterministic schema 替代 80% 以上 session-extraction 输出，同时保持 temporal endpoint、negative scope 与 collection closure？
2. 为什么 LoCoMo 的 evidence frames 已存在，却没有进入 fine candidates：是 embedding text、owner/role filtering、session quota、sidecar ranking，还是 V4.1 optional-stage ordering？
3. scene-window 与 late-scene 的 73–85 个 gold refs 有多少是互相重叠，最小充分窗口是多少？
4. count/list 问题是否应使用 set-coverage objective，而不是单节点相似度与固定 top-k？
5. 当前 720 条 semantic-neighbor 边几乎没有出现在 expansion trace；是边质量不足、阈值过高，还是 navigator 根本没有消费该关系？
6. 能否把 packed context 的 turn-level recall 设为硬约束，在其后再优化 evidence tokens，而不是先按启发式压缩再观察 recall？

## 研究包与日志字段

研究包位于 Gate A artifact 的 `research/` 子目录，核心文件如下：

| 文件 | 粒度 | 主要字段与用途 |
|---|---|---|
| `llm_call_log.jsonl` | 5,266 次 provider call | benchmark、memory/question、stage、model、cached/uncached input、output、reasoning、latency、retry、finish reason、max tokens、error；用于完整 token/延迟/截断审计 |
| `session_build_log.jsonl` | 5,008 个 session extraction | turn/frame/coverage/lossless-only 数、parse error、extraction mode、prompt/completion、finish reason 与 completion cap |
| `memory_build_log.jsonl` | 110 个独立 memory | session/node/edge/leaf/summary 数、关系分布、构建 token、每 session/turn token、identity token、parse/truncation、build latency |
| `question_research_log.jsonl` | 200 道题 | exact gold/retrieved/candidate/packed turn ID、session 与 turn 指标、failure stage、缺失 evidence、channel/feature trace、图规模、query token 与 latency |
| `turn_failure_cases.jsonl` | 124 道未 all-hit 题 | 逐题最早失败阶段及所有相关 trace；是 `question_research_log` 的失败子集 |
| `token_breakdown.csv` | benchmark × stage | calls、input/output/total、cache、retry、length finish、latency p50/p95、run share |
| `feature_trace_contribution.csv` | trace feature | active questions、emitted/packed turns、gold coverage、overlap precision、exclusive refs |
| `retrieval_channel_gold_coverage.csv` | retrieval channel | candidate/packed refs、gold refs、precision 与 exclusive contribution |
| `graph_expansion_relation_contribution.csv` | edge relation | expansion 数、gold destination、packed destination 与 packed-gold destination |
| `turn_reference_metrics.csv` | stratum × support role | gold/candidate/packed refs、candidate/packed recall、source-frame binding rate |
| `research_summary.json` | run 汇总 | 本报告所有 headline、build、graph、retrieval、failure 与指标口径 |
| `bundle_manifest.json` | 文件级 | producer SHA-256、各研究数据文件的 SHA-256 与 bytes，用于复核未被修改 |

日志不复制 benchmark 对话原文或答案；LongMemEval span 只保留 offsets、hash、role、confidence 和 packed/full-containment 布尔值。`question_research_log.jsonl` 保留问题文本以便失败分析，但 online build/query 代码仍不能读取任何 gold 或题型标签。
