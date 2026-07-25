# GraphMem V2 LoCoMo 全集报告（2026-07-25）

## 结论

本次使用官方 `locomo10.json` 的 10 组 conversation、1986 道题完成全集测试。
每组 conversation 只构建一次 `hierarchical_state_graph_v2`，同组问题共享图索引。
最终运行使用 `deepseek-v4-flash`、thinking disabled、本地
`Qwen3-Embedding-0.6B`（1024 维），并按 conversation 分成 4 个独立进程。

- 官方 LoCoMo token F1：**20.97%**
- 旧版同一官方 scorer：**11.55%**
- V2 绝对提升：**+9.42 个百分点**
- 构建 token：P50 103,030，P95/最大 112,715，10/10 低于 300K
- 回答 token：P50 5,351，P95 5,811，最大 6,165，1986/1986 低于 10K
- 构建和回答 reasoning token：均为 0
- Canonical 成功运行 DeepSeek token：11,687,972

## 10 组 conversation 构建开支

| Conversation | Sessions | Questions | Miss input | Hit input | Output | Total |
|---|---:|---:|---:|---:|---:|---:|
| conv-26 | 19 | 199 | 31,955 | 7,296 | 35,790 | 75,041 |
| conv-30 | 19 | 105 | 27,122 | 7,296 | 37,249 | 71,667 |
| conv-41 | 32 | 193 | 48,803 | 12,288 | 45,160 | 106,251 |
| conv-42 | 29 | 260 | 44,350 | 11,136 | 44,134 | 99,620 |
| conv-43 | 29 | 242 | 47,564 | 11,136 | 44,330 | 103,030 |
| conv-44 | 28 | 158 | 47,980 | 10,752 | 53,983 | 112,715 |
| conv-47 | 31 | 190 | 48,896 | 11,904 | 45,304 | 106,104 |
| conv-48 | 30 | 239 | 46,702 | 11,520 | 53,480 | 111,702 |
| conv-49 | 25 | 196 | 37,984 | 9,600 | 47,359 | 94,943 |
| conv-50 | 30 | 204 | 45,919 | 11,520 | 47,983 | 105,422 |
| **合计** | **272** | **1,986** | **427,275** | **104,448** | **454,772** | **986,495** |

回答阶段合计：

- Cache miss input：6,383,972
- Cache hit input：4,266,368
- Output：51,137
- Total：10,701,477
- 单题 mean / P50 / P95 / max：5,388.46 / 5,351 / 5,811 / 6,165

## 准确率

| Category | Questions | V2 F1 | 旧版 F1 |
|---:|---:|---:|---:|
| 1 | 282 | 19.74% | 21.46% |
| 2 | 321 | 18.65% | 10.70% |
| 3 | 96 | 16.41% | 7.11% |
| 4 | 841 | 33.30% | 15.18% |
| 5 | 446 | 1.12% | 0.00% |
| **Overall** | **1,986** | **20.97%** | **11.55%** |

V2 在 category 2/3/4 显著优于旧版，但 category 1 略有回归，category 5
基本未适配官方 abstention 口径。

## 为什么准确率仍低

### 1. Session recall 看起来高，但掩盖了 turn-level evidence 丢失

最终检索的 gold-session any/all/mean recall 分别为 95.92% / 87.66% /
92.14%。仅看这个指标会误以为召回已经足够。

用官方 `evidence` turn 做更严格审计：

- leaf semantic top-28 evidence recall：74.40%
- leaf BM25 top-28 evidence recall：61.73%
- leaf entity top-28 evidence recall：56.11%
- 最终 post-pack evidence leaf recall：约 57.24%
- 在最终 prompt 中逐字找到完整官方 evidence turn：micro 41.59%，
  per-question macro 47.41%
- category 1 的全部 evidence turn 同时进入 prompt：仅 9.93%

因此主要问题是相关 conversation/session 被选中后，session 内真正回答问题的 turn
没有被保护到最终 evidence pack。粗图路由和 session recall 指标对此不敏感。

### 2. 图扩展确实运行，但对最终答案的增益太弱

- typed/operand expansion 共扩展 31,478 个节点
- 最终保留 6,991 个，retention 22.21%
- 图可挽救 gold session 的问题仅 44 个，最终保留挽救证据的仅 16 个
- 新增 graph fact 成为 operator source 的问题仅 1 个（0.05%）
- operand expansion 成为 operator source：0
- `participates_in` 有 763 次从 seed 看是反向入边而被阻塞，其中 126 次位于 gold session

边数量并不少：`contains` 6,927、`same_entity` 4,124、`source` 3,966、
`participates_in` 3,852、`semantic_neighbor` 2,764。但大量扩展只是结构遍历，
没有转化为 operator operand 或最终答案证据。当前图更像候选补充器，还不是有效的
query-time reasoning graph。

### 3. 回答格式与 LoCoMo 官方 scorer 不兼容

- prediction 平均 16–30 words，gold 平均 3–7 words，输出明显过长
- 日期常输出 ISO（如 `2023-05-07`），gold 为 `7 May 2023`
- 279 道 category-5 题输出 `not enough information`，官方只接受包含
  `no information available` 或 `not mentioned`
- 只做零 token 的日期和 abstention 规范化，离线 F1 可从 20.97% 升到约 35.0%

这说明约 14 个百分点是 answer adapter / metric contract 问题，但即使修复格式，
turn-level evidence recall 和答案抽取仍然不足。

## LoCoMo V2.1 建议

1. 增加 LoCoMo 双人对话模式，两个 speaker 都作为一等主体，不使用传统
   user/assistant 权重语义。
2. 将 leaf semantic/BM25 的 direct evidence 保护提前到 packer：
   top-28 候选中命中 exact speaker/entity/date 的 leaf 不得被 routing-card 配额挤掉。
3. 路由卡选中 session 后执行 session 内二次检索，而不是只沿 card 指针取固定 leaf；
   multi-hop 至少保护每个候选 session 的 2–4 个不同 evidence turn。
4. 允许 `participates_in/source/contains` 在 query expansion 中按受控规则反向遍历，
   再沿 `same_entity/same_predicate/before/after` 扩展；按“是否进入最终 pack /
   operator source”训练和调参，而不是按扩展节点数。
5. 将 graph-rescued leaf/fact 设为 packer protected tier，避免 44 个 rescue candidate
   最终只保留 16 个。
6. 加入无 LLM token 的 LoCoMo answer adapter：短答案、ISO 日期转自然日期、
   abstention 固定短语、列表去解释性前缀。
7. category 3 需要单独允许受证据约束的 commonsense inference；当前 LongMemEval
   风格的严格 grounded prompt 会过度拒答。
8. 后续调优主指标改为 official evidence-turn post-pack recall，而不是 gold-session
   recall；目标建议先达到 evidence any ≥90%、multi-hop all ≥75%，再重跑答案。

## 运行与统计说明

Canonical 结果来自 4 个完整 conversation 分片合并后的成功运行。调度调优期间的
不完整运行另有 191 个已落盘回答、1,034,941 answer token；此外被中止时仍在途的
调用无法从本地日志恢复 provider usage，因此不计入 canonical 每题统计。上表的
10 组构建和 1986 题回答记录内部都满足 hit + miss = prompt、prompt + output = total。
