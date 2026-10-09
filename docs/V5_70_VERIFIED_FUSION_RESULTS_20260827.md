# V5.70 Verified Fusion 全量实验结论

## 1. 结论

V5.70 的“保留 64-turn 图证据 + 最多 8 条 flat witness + Luna-max
独立复核”没有超过 V5.64 基线，因此不应升级为默认查询路径。为消除远端
temperature-zero Judge 对相同答案的少量漂移，本文以 prediction-byte paired
结果为主口径：

| Benchmark | V5.64 Luna-max | V5.70 paired | Delta | Gains / losses | McNemar exact p |
|---|---:|---:|---:|---:|---:|
| LongMemEval | 439/500 (87.80%) | 437/500 (87.40%) | -0.40 pp | 15 / 17 | 0.8601 |
| LoCoMo | 1353/1540 (87.86%) | 1347/1540 (87.47%) | -0.39 pp | 17 / 23 | 0.4296 |

独立重判的原始分数为 LongMemEval 436/500（87.20%）和 LoCoMo
1348/1540（87.53%）。其中分别有 1 和 5 道题在答案字节完全相同时发生
Judge verdict 翻转；paired 口径对这些题继承基线 verdict，只对答案发生变化的
213/500 和 400/1540 道题使用新 verdict。

V5.70 相对基线的 evaluator-only union 上限也只有 LongMemEval
454/500（90.80%）和 LoCoMo 1370/1540（88.96%）。所以即使存在一个完美
选择器，这一条新候选路径本身仍不足以达到 92%。

## 2. 固定协议

- Memory build：冻结的 Qwen3-30B 图数据库，不重建、不读取 gold 进行逐题选择。
- 基线回答：V5.64，Luna-max，64-turn。
- V5.70：完整保留原 64-turn 图证据，额外加入最多 8 条 flat witness；证据
  Prompt 相对原 Prompt 最多增加 500 个 Qwen tokenizer token。
- 回答：`gpt-5.6-luna`，reasoning effort `max`。
- Judge：`gpt-5.6-luna`，reasoning effort `medium`，固定 benchmark prompt。
- 覆盖：LongMemEval 500 + LoCoMo Category 1--4 共 1,540，合计 2,040。
- 选择器不读取逐题 gold、类别或 Judge。安全模式是在 aggregate gold audit 后
  选择，因而本实验属于迭代开发结果，不应被表述为未调参的最终测试集结果。

回答和 Judge 均达到完整覆盖，Prompt hash mismatch 为 0，输出截断为 0。
`locomo03_0078` 的 V5.70 请求连续两个 600 秒窗口没有返回，最终使用明确标记的
previous-answer fallback。该 fallback 策略不依据 verdict 选择答案，但执行决定
发生在查看旧 verdict 之后，已记录在 `fallbacks.jsonl`，不能隐去这一事实。

## 3. 运行时可见路由的配对变化

| QueryIR route | LongMemEval net | LoCoMo net | 解释 |
|---|---:|---:|---|
| aggregate | +2 | +1 | flat witness 能补齐遗漏操作数，是唯一稳定的正向路径 |
| inference | 0 | +1 | 有少量补充事实收益，但样本较少 |
| multi_hop | -1 | +1 | 收益不稳定，关系链与近邻噪声同时增加 |
| state | -1 | 0 | 新证据容易把历史状态误当当前状态 |
| lookup | -1 | -1 | 原答案已准确时，额外近邻诱发无依据改写 |
| list | -2 | -3 | excerpt 和候选扩张会漏掉原列表项或引入同主题项 |
| temporal | +1 | -5 | LongMemEval 有补证收益，但 LoCoMo 的近邻日期和事件混淆更严重 |

若事后仅在 `aggregate` 和 `inference` 路由采用 V5.70，其理论净收益仍只有
LongMemEval +2 题、LoCoMo +2 题，离 92% 分别还差 19 和 62 题。路由门控能
避免回退，但不能解决候选正确性上限。

## 4. 主要失败机制

1. **补证与加噪同时发生。** 新路径修复了植物总数、教育年限、日期和累计数值等
   缺操作数问题；但在乐器、车辆、活动、最近状态和日期问题上，同主题 flat
   witness 被错误绑定到目标实体或目标时间。
2. **复核器存在不对称误修。** Prompt 中把旧答案声明为 fallible proposal 后，
   Luna-max 会在新增近邻出现时主动推翻本来正确的答案。配对结果中，新增正确
   不足以覆盖被改坏的基线答案。
3. **`temporal` 仍缺少显式事件表。** 自然语言约束不足以稳定区分 mention time、
   event time、计划时间、最近一次事件以及相邻人物的事件。
4. **`list/lookup` 不适合统一扩展。** 这两类题更依赖原始精确 span；额外证据和
   截断 excerpt 会降低 attribution precision。
5. **Judge 有可测漂移。** 相同 prediction 的 verdict 翻转率约为 0.35%--0.44%。
   因此小于约 0.5 pp 的独立重判差异不能直接归因于系统改动，必须使用 paired
   carry-forward。
6. **Luna-max 长尾显著。** 1,206 个成功外部回答累计发生 4,193 次可恢复传输
   retry；一个 count 请求连续两个 600 秒窗口无返回。服务返回的 usage 中还存在
   completion token 高于客户端 `max_completion_tokens` 的记录，说明该代理的
   usage/上限执行需要单独审计，不能直接用于严格成本结论。

## 5. 92% 可达性分析

在同一 Qwen build 上，现有五个全量 Luna 回答臂的 evaluator-only oracle 约为
LongMemEval 469/500（93.8%）和 LoCoMo 1415/1540（91.88%）。加入此前只在
错误题+控制题上运行的 typed-readout 候选后，探索性 oracle 约为 94.2% 和
92.99%。后一个数字不可作为系统成绩：候选可用性和 oracle 选择都使用了 evaluator
信息；它只说明“候选池内存在足够多正确答案”。

简单的文本 medoid/一致性选择仅达到 LongMemEval 88.4%、LoCoMo 88.12%；这还是
在看到结果后分别选取最佳阈值的探索性上限，不是可报告的测试成绩，且仍远低于
oracle。这表明当前瓶颈已从“再生成一个答案”转为“构造可验证的 typed
operands，并用来源证据安全选择答案”。直接增加更多 Luna reasoning 或 flat
turns 不具备达到 92% 的证据。

## 6. 下一版最小有效升级

1. 只对 `aggregate` 启用 flat packet；`lookup/list/state` 默认保留 V5.64，
   `temporal` 在事件表尚未完成前同样回退。
2. 将 count/sum/comparison 从自然语言 worksheet 升级为可审计 operand table：
   `subject, predicate, object, event_time, status, source_turn_id`，先做实体与事件
   去重，再由确定性算子执行。
3. 为 temporal 构造 endpoint table，分别记录事件时间、提及时间、计划/完成状态和
   source-time normalization；禁止只凭相邻日期近邻改写答案。
4. 为 lookup/list 保留完整精确 span，不把截断 excerpt 当作唯一事实来源；只有在
   新 span 满足主体、关系和对象三元绑定时才允许推翻旧答案。
5. 在形成至少一个新的、全量且标签无关的 typed candidate 后，再运行
   evidence-grounded jury。Jury 只处理候选实质冲突的题，其余题 byte-for-byte
   保留 V5.64；最终仍使用 paired Judge 口径。

## 7. 产物

- V5.70 materialization manifest：
  `../artifacts/report/v5_70/verified_fusion64_v3/manifest.json`
- 完整回答 manifest：
  `../artifacts/report/v5_70/verified_fusion64_v3/answer_luna_max/run_manifest.json`
- 完整结果摘要：
  `../artifacts/report/v5_70/verified_fusion64_v3/full_summary.json`
- LongMemEval paired manifest：
  `../artifacts/report/v5_70/verified_fusion64_v3/paired_audit/lme/manifest.json`
- LoCoMo paired manifest：
  `../artifacts/report/v5_70/verified_fusion64_v3/paired_audit/locomo/manifest.json`
- 单题 fallback 记录：
  `../artifacts/report/v5_70/verified_fusion64_v3/answer_luna_max/fallbacks.jsonl`
