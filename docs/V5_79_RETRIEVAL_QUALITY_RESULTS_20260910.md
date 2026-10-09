# V5.79 召回质量与多视图呈现验证（2026-09-10）

## 1. 目标与约束

本轮只修改从 QueryIR 到证据呈现的读取链路，不重建 Memory，也不使用
gold、参考答案或 judge 标签参与在线路由。目标是在保留 64-turn 图前缀的
前提下，提高关系、时间和跨轮对话 witness 的有效覆盖，同时阻止新增候选
挤掉旧版本已经命中的证据。

正式证据始终是不可变的原始 conversation turn。关系标签、视图名称和
编号只作为阅读导航；系统不把生成式摘要或图边标签当作答案事实。

## 2. 实现

### 2.1 单调的多 lane 召回

- 冻结原图前缀、基础 lexical focus、基础 relation lane 和基础 temporal
  lane；新增视图只能去重后追加，不能重排或替换已有 witness。
- relation 检索拆为 direct、半径为 2 的 dialogue packet、QueryIR relation
  concept，以及经过 query-trigger 门控的 expanded relation concept。
- temporal 检索拆为绝对时间归一化视图，以及独立的 duration/event-order
  视图。后者只对 `before/after`、ordinal 和 duration 义务开放。
- 所有 lane 在追加前同时排除图中已有 turn 和此前 lane 已选 turn，保持
  单边、单证据、单计数语义。

### 2.2 精度门控、排序与选择

- 无绝对日期的普通 `when` 问题不再统一触发 event-transition 扩展；它们
  继续由图、lexical、relation 和 dense 主路径处理。
- closure-risk 问题可使用更大的 relation packet 配额，但 direct relation
  保留独立名额，避免椭圆回复替换直接命中。
- expanded relation/temporal 只在规范化 query 确实产生新概念且 seed
  顺序与基础视图不同的时候开放。
- 每一层都记录候选来源、实际追加 turn、选择 lane 和分数，因而可以区分
  “边不存在”“候选未入选”“证据已给出但回答未转化”。

### 2.3 图式证据呈现

- Graph 与 Source-focus 分区展示，各自按 session 聚类并在 session 内保持
  对话顺序，使 premise、bridge 和 value turn 相邻。
- 对 lookup/list/temporal 路由保留一个 exact-source Focus Capsule，用于
  缓解长上下文位置损失；重复项明确要求只计数一次。
- 可选 Query-view navigation map 最多列出 3 个视图、6 个可见 F/G 编号，
  例如 `event-order/duration`、`relation-match` 和 `semantic`。该 map 不包含
  事实摘要。
- 可选 `{via=...}` 只展示经过 gate 的 semantic graph provenance；回答
  prompt 明确要求逐条核对邻接原文。

## 3. 全量 LoCoMo 离线覆盖门禁

下面的 all-hit/partial/missing 是 annotated gold-turn coverage，不是回答
accuracy。

| 方案 | All-hit | Partial | Missing | 相对 v13 的 all-hit 回退 |
|---|---:|---:|---:|---:|
| v13：三 relation lane + coarse temporal | 1,296 | 171 | 69 | -- |
| v17：宽 event-transition | 1,300 | 171 | 65 | 0 |
| v20：QueryIR 精度门控 | 1,300 | 171 | 65 | 0 |
| v19：精度门控 + relation/map 呈现 | 1,300 | 171 | 65 | 0 |
| v21：按 QueryIR route 选择呈现 | 1,300 | 171 | 65 | 0 |

新增的 4 道 all-hit 分别覆盖：持有时长反推首次获得时间、第三次比赛、
挫折后的定向鼓励，以及一个事件之后的后续行动。1,296 道旧 all-hit 全部
保留。

精度门控把 event-transition 激活问题从 415 降到 134（-67.7%），追加
witness 从 829 降到 268（-67.7%），同时保留全部 4 道新增 all-hit。

## 4. Token 与本地回答诊断

相对同样带 1-turn Capsule 的 v13，v19 的 packing prompt 平均增加 177
token，p95 增量 278，单题最大增量 477；全部低于 500-token 增量门限。
v21 只对 QueryIR 判为 `inference` 的 35/1,540 题启用 relation provenance
与 navigation map，其余 prompt 与 v20 逐字节相同。相对 v20，v21 全集平均
只增加 1.4 packing token，p95 增量为 0，单题最大增加 111 token。

下表是严格归一化后 reference/prediction 的包含关系代理，只适合比较同一
模型、同一数据上的相对变化，不能作为 benchmark accuracy 或正式 judge
分数。

| 方案 | N=1 命中数 | Best-of-2 | Best-of-4 | Best-of-8 |
|---|---:|---:|---:|---:|
| v13 | 613 | 644 | 656 | 670 |
| v17：宽召回 | 620 | 643 | 654 | 662 |
| v20：只做精度门控 | **624** | **647** | 655 | 667 |
| v19：门控 + 图式导航 | 622 | 637 | 648 | **671** |

解释：宽 event-transition 能改善首答，但会降低多样本可恢复性；QueryIR
门控在 281 个发生变化的 prompt 上相对宽召回净增 8、净损 4 个 N=1 代理
命中。该代理只用于快速筛选，最终取舍以下面的正式 judge 为准。

所有本地回答均为 1,540/1,540、0 retry。v19 的实际 API prompt token
mean/p95/max 为 6,164/7,156/8,142，8 个 completion 合计 token 的
mean/p95/max 为 241/627/6,626；共有 3 个 choice 达到 2K 输出上限。

## 5. 正式 Luna-medium 判分

所有方案均固定 candidate-1，并使用同一版 memory-benchmarks LoCoMo prompt、
GPT-5.6-luna medium、temperature 0 和 seed 0。外部 payload 只有 question、
reference answer 与 candidate answer，不发送 memory、evidence 或完整回答
prompt。

| 方案 | Overall | Cat-1 | Cat-2 | Cat-3 | Cat-4 |
|---|---:|---:|---:|---:|---:|
| v13 | 87.73% | 83.69% | 85.98% | 62.50% | 92.63% |
| v20：精度门控、精简呈现 | 88.05% | 84.75% | **87.54%** | 58.33% | 92.75% |
| v19：全局图式呈现 | 88.05% | 84.75% | 86.29% | **63.54%** | 92.63% |
| v21：仅 inference 图式呈现 | **88.12%** | **84.75%** | **87.54%** | 58.33% | **92.87%** |

v20 和 v19 相对 v13 都净增 5 题，但优势分布不同：v20 更适合 temporal，
v19 更适合 inference。v21 将呈现开关编译进 QueryIR route，在 v20 基础上
1 gain / 0 loss；相对 v13 为 27 gains / 21 losses，净增 6 题（+0.39 pp）。
这些差异的 exact McNemar p 值分别为 1.0 和 0.471，尚不显著，因此本轮
结果支持“路由呈现是安全的小幅改进”，不支持把全局 relation label/map
描述为已证实的大幅增益。

### 5.1 同一 prompt 的 Best-of-N 上界

只对 candidate-1 的 183 道错题判后续样本；相同 answer hash 直接复用同一
verdict，共新增 474 次 source-free judge 请求，0 request retry、0 semantic
retry、0 failure。该结果是回答模型的 oracle 上界，不是无需选择器即可
部署的单回答准确率。

| k | Best-of-k 正确数 | Accuracy | 相对上一个 k 新恢复 |
|---:|---:|---:|---:|
| 1 | 1,357 | 88.12% | -- |
| 2 | 1,372 | 89.09% | 15 |
| 3 | 1,377 | 89.42% | 5 |
| 4 | 1,378 | 89.48% | 1 |
| 5 | 1,382 | 89.74% | 4 |
| 6 | 1,384 | 89.87% | 2 |
| 7 | 1,385 | 89.94% | 1 |
| 8 | **1,387** | **90.06%** | 2 |

30 道首答错误可以由同一不可变 prompt 的其他采样恢复。其中 all-hit 20
道、partial 4 道、missing 6 道，说明模型波动是可测量因素，但并不能解释
全部误差。

## 6. 已否决或暂不启用的方向

- v14--v16 将扩展词表混入基础 rank，出现旧 all-hit 被挤出的非单调回退。
- 对全部无绝对日期 `when` 问题开放 transition，在 829 个额外 witness 中
  只有 3 个 annotated gold turn，并降低 Best-of-8 代理。
- 旧 multi-view dense paraphrase 追加 374 个 witness，但 annotated gold
  新命中为 0，不重新启用。
- focus-overlap promotion 的既有 paired judge 下降；session diversity 没有
  带来 gold coverage 增益，均不进入默认配置。
- 不使用生成式 evidence summary；它会引入 attribution、modality 和时间
  失真，且难以维持 source-only 契约。

## 7. 剩余瓶颈

v21 首答仍错 183 题，其中 119 题已 all-hit、37 题 partial、26 题 missing，
另有 1 题没有 evidence 标注。Best-of-8 后仍错 153 题：99 题属于 all-hit
后的呈现/模型缺口，54 题属于 partial/missing 的召回缺口。标注证据在最终
prompt 中的位置 p50/p95/p99 为 30/73/81，说明若只继续扩大 turn 数，新增
证据很容易落在长上下文尾部而无法稳定转化。

按生产 QueryIR route 看，首答最弱的是 aggregate（29/47）和 multi-hop
（55/70）；按 benchmark 类别看，Best-of-8 后 Cat-3 仍只有 67/96。现有
QueryIR 只把 35 题路由成 `inference`，而 benchmark Cat-3 有 96 题；v21
相对 v20 的唯一净增题甚至属于 Cat-4。这说明生产 route 与 benchmark
类别并不等价，也暴露出当前 inference obligation detector 过窄。下一步
不应统一扩大 turn 数，而应：

1. 为 `both`、共享属性和多 owner 问题建立每个 subject 的独立证据槽；
2. 为列表义务建立 value-level 去重与“未覆盖 owner/关系”停止条件；
3. 对 inference 建立可观测的 premise/bridge/conclusion 槽，只补未满足槽，
   不追加同主题泛相似 turn；
4. 将 99 道 all-hit 且 Best-of-8 仍错的问题作为 presentation set，检查
   指代、否定、模态、计数和时间换算，而不是继续加 recall；
5. 训练或验证一个不读取 reference/judge 的 online selector；在此之前，
   90.06% 只能标注为 Best-of-8 oracle，不能写成单回答结果。

## 8. 复现产物

- v17：`../artifacts/report/v5_78_retrieval_audit/unified_overlay_graph64plusadaptive_focus16_relation_three_lane_v17_strict_monotonic_capsule1_lossless`
- v19：`../artifacts/report/v5_78_retrieval_audit/unified_overlay_graph64plusadaptive_focus16_relation_three_lane_v19_precision_gated_navmap_relations_capsule1_lossless`
- v20：`../artifacts/report/v5_78_retrieval_audit/unified_overlay_graph64plusadaptive_focus16_relation_three_lane_v20_precision_gated_capsule1_lossless`
- v21：`../artifacts/report/v5_78_retrieval_audit/unified_overlay_graph64plusadaptive_focus16_relation_three_lane_v21_queryir_routed_presentation_capsule1_lossless`
- source-free judge：`../artifacts/report/v5_79_paired_judge`
- v21 正式 gap audit：v21 目录下的 `gap_audit_official/summary.json` 与
  `per_question.jsonl`
