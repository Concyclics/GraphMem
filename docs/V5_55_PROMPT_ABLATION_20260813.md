# V5.55 无思考 Answer Prompt 消融

## 结论

Prompt-only 改写没有稳定修复“召回正确、回答错误”。在 turn64、相同
PreparedAnswer、相同 evidence IDs/顺序的严格配对条件下：

- all-hit 诊断集上，GPT-5.4-mini 的题型选择性改写从 42/88 提升至
  45/88，但 Qwen3-30B 从 58/88 降至 55/88；
- 扩展至 LongMemEval 500 后，该改写使 GPT-5.4-mini 从 345/500
  降至 342/500，Qwen3-30B 从 381/500 降至 377/500；
- 完整集变化均不显著，且方向为负，因此 `selective_v1` 不应合入默认
  Answer Prompt。

这说明诊断集上的局部收益来自少量 Temporal/State 样本，不能外推为
通用 readout 改进。要继续提高准确率，应将题型解析、事件绑定、时间端点、
聚合 operands 和缺失性判定编译为结构化 AnswerPlan，而不是继续向自然语言
Prompt 追加规则或整体翻转 evidence blocks。

## 固定实验口径

- 索引与召回：V5.54 Full，turn64，12K evidence，PreparedAnswer 冻结；
- 诊断集：具有 turn-level gold 且 packed all-hit 的 90 题，其中 2 题为
  deterministic bypass，不进入模型 Prompt 消融，模型请求共 88 题；
- Answer：Qwen3-30B-A3B-Instruct-2507-FP8 与 GPT-5.4-mini；
- 思考模式：Qwen 请求显式 `enable_thinking=false`，GPT 请求显式
  `reasoning_effort=none`；
- 输出上限：2K tokens。除 Qwen 的 `Baseline + Route + Primary-last`
  在 `gpt4_59c863d7` 上出现 1 次 repetition/length 截断外，其余诊断臂
  均无截断；两个 Selective-v1 完整 500 题实验均 `output_truncated=0`；
- Judge：GPT-5.6-luna，reasoning none，固定 Mem0 LongMemEval prompt；
- Prompt 变换不读取 gold。gold 仅用于确定 all-hit 诊断集成员；
- paired judge：若候选 prediction 与基线逐字相同，则继承基线 verdict，
  避免 temperature=0/seed=0 下仍存在的 Judge 抖动。

## 消融臂

| 实验臂 | 改动 |
|---|---|
| Baseline | V5.54 原 Prompt |
| Lean | 缩短 system instruction |
| Lean + Route | Lean，加正则题型路由卡 |
| Lean + Route + Primary-last | 再将图证据块整体逆序 |
| Lean + Route + Primary-last + Closure | 再加完整性验证门 |
| Baseline + Route | 原 system，加题型路由卡 |
| Baseline + Route + Closure | 再加完整性验证门 |
| Baseline + Route + Primary-last | 再将图证据块整体逆序 |
| Baseline + Route + Primary-last + Closure | 三项全部启用 |
| Selective-v1 | 仅 Temporal 加路由卡并逆序；仅 State-update 加路由卡 |

`Primary-last` 不删除证据，但会翻转连续 GRAPH/CHAIN/AUX block 的全局顺序；
它用于检验“把主证据靠近问题”是否足够。结果显示这种粗粒度顺序变换会破坏
时间链与事件局部性。

## all-hit 诊断结果

| Prompt | Qwen3-30B | Δ | GPT-5.4-mini | Δ |
|---|---:|---:|---:|---:|
| Baseline | 58/88 (65.91%) | — | 42/88 (47.73%) | — |
| Lean | 53/88 (60.23%) | -5 | 35/88 (39.77%) | -7 |
| Lean + Route | 54/88 (61.36%) | -4 | 31/88 (35.23%) | -11 |
| Lean + Route + Primary-last | 44/88 (50.00%) | -14 | 34/88 (38.64%) | -8 |
| Lean + Route + Primary-last + Closure | 40/88 (45.45%) | -18 | 27/88 (30.68%) | -15 |
| Baseline + Route | 51/88 (57.95%) | -7 | 37/88 (42.05%) | -5 |
| Baseline + Route + Closure | 40/88 (45.45%) | -18 | 36/88 (40.91%) | -6 |
| Baseline + Route + Primary-last | 49/88 (55.68%) | -9 | 39/88 (44.32%) | -3 |
| Baseline + Route + Primary-last + Closure | 33/88 (37.50%) | -25 | 37/88 (42.05%) | -5 |
| Selective-v1 | 55/88 (62.50%) | -3 | 45/88 (51.14%) | +3 |

Selective-v1 在 GPT-5.4-mini 上是 3 gain / 0 loss，但 McNemar exact
`p=0.25`；在 Qwen3-30B 上是 2 gain / 5 loss，`p=0.4531`。Closure
instruction 对两个模型均明显有害：它放大了“缺任一 endpoint 就拒答”的倾向，
将可由上下文推导的答案错误地变成 `Insufficient information`。

## LongMemEval 500 严格配对验证

| 模型 | Baseline | Selective-v1 | Gain/Loss | McNemar p |
|---|---:|---:|---:|---:|
| Qwen3-30B | 381/500 (76.20%) | 377/500 (75.40%) | 5 / 9 | 0.4240 |
| GPT-5.4-mini | 345/500 (69.00%) | 342/500 (68.40%) | 6 / 9 | 0.6072 |

完整集仅对 75 道 Temporal 和 81 道 State-update 改 Prompt，其余 344
道请求严格复用原 Prompt/答案。paired judge 又分别继承了 402 道 Qwen 和
425 道 GPT 的相同 prediction verdict。两个模型均无输出截断。

分题型看，GPT-5.4-mini 的 State-update 从 62/81 小幅升至 63/81，
Temporal 却从 45/75 降至 41/75；Qwen 的 State-update 从 63/81 降至
62/81，Temporal 从 52/75 降至 49/75。诊断集只包含 2 道 State-update，
因此其局部正向结果不能代表完整分布。

## 典型 gain/loss

- 有效案例：`a1cc6108` 中 Qwen 由 21 岁修正为 11 岁；`71017276` 中
  GPT-5.4-mini 由 0 周修正为 4 周。显式 endpoint 约束确实能修复部分算术题。
- 退化案例：`51c32626` 原本正确回答 February 1st，改写后因严格缺失门
  退化为无法确定；`73d42213` 原本正确回答 9:00 AM，逆序后被 1:00 PM
  干扰；`c18a7dc8` 原本正确回答 7 年，改写后错误拒答。

这些案例说明问题不是“是否加一句更强约束”，而是 Prompt 当前没有显式表示
`question slot -> event -> source-time -> operand` 的绑定关系。只改变长 evidence
列表的全局顺序，会同时修复一部分 recency 错误并制造另一部分错配。

## 下一步最小可行强化

1. 用 QueryIR/图结果生成确定性的 AnswerPlan，不让 answer model 自己从
   64 turns 中猜题型与 operands；
2. Temporal 证据按“同一事件局部聚簇 + 聚簇内时间顺序”组织，不整体逆序；
3. 给每个 endpoint/operand 标明 `supported / derived / missing`，只有真正
   missing 时才触发 abstention；
4. Route 必须来自 QueryIR typed operator 和高置信槽位，而不是正则分类；
5. 先在全量 500 上做 paired validation，再考虑固化，避免 all-hit 小集过拟合。

机器可读汇总位于
`artifacts/report/v5_55/prompt_ablation/summary.json`，Prompt 物化、回答、usage、
Judge calls 和逐题 verdict 均位于同目录下。
