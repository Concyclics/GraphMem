# 多标签 QueryIR 升级与全量验收（V5.80--V5.81）

## 1. 目标与冻结基线

本轮针对 LoCoMo 中“一个问题同时包含多个回答义务”的情况扩展 QueryIR，并严格区分三件事：内部语义标签、物理检索路由、回答阶段可见契约。冻结基线为 v21；本地 Qwen3-30B candidate-1 为 1357/1540（88.12%），同一不可变 Prompt 的 Best-of-8 oracle 为 1387/1540（90.06%）。运行时只允许使用问题与 Memory，不读取 benchmark category、参考答案、gold evidence、judge verdict 或历史回答。

## 2. 最终实现

### 2.1 多标签语义编译，单一物理主路由

`compile_query_obligations` 在保留既有物理主路由的同时，从问题文本独立编译 aggregate、temporal、latest-state、exhaustive-set、multi-entity、comparison、causal、counterfactual、inference、multi-hop、geographic-resolution、alias-resolution 与 negative-existence 等义务。标签可供检索策略、审计与 provenance 使用，但不再默认全部写入回答 Prompt。

复数答案头可识别 `book recommendations`、`sports` 等集合型 WH 短语，同时不会把 supporting clause 中的时间数量误当作列表请求。geographic-resolution 进一步限定为明确要求 city/country/state/province/county/region/continent 等行政或地理层级的答案头；`Where ...?`、`Which places ...?` 和一般的 visit/live 关键词仍按直接地点抽取处理，避免模型将明确地点扩写成附近地点或历史地点。

### 2.2 高精度可见契约

全量配对实验表明，generic inference、exhaustive、multi-entity、alias 与 polarity 提示会重复主路由已有约束，增加 Prompt 熵，并可能诱发过度推断。最终 `presentation_tags()` 只允许两类会改变答案结构的正交义务进入 Prompt：

- `comparison`：要求分别绑定候选项后再比较；
- `geographic_resolution`：只允许把已命中的源地点转换到问题明确要求的行政层级。

其余标签仍保留在 QueryIR trace 中，不触发额外回答文本。最终全量共有 15 个 comparison 与 32 个 geographic-resolution 可见契约。

### 2.3 证据不变量与稳定排序

可见契约必须在证据选择完成后追加，不得替换、删除或重排冻结 evidence。审计比较每题 `evidence_turn_ids`、最终 payload hash 与运行策略，并在任一证据删除时失败。

实验中发现 temporal 边界候选的浮点分数会受 Python set 迭代顺序影响：数学上同分的 token 权重以不同顺序累加后产生约机器精度量级的差异，进而交换最后一条 witness。修复后 lexical、relation-concept、temporal 与 auxiliary marginal rank 均按稳定 token 顺序累加，并在 turn-id tie-break 前消除亚精度噪声。关键边界题在 `PYTHONHASHSEED=1` 与 `777` 下得到相同选中序列 hash。

### 2.4 批量向量预热与回答并发门禁

Prompt 物化前对 query view 去重并以最多 256 条批量预热，cache key、instruction revision 与归一化方式与在线检索一致。冻结全量的 1529 个唯一 query 向量全部命中持久化缓存，因而本轮物化为 0 embedding API 调用。

回答重放按 `workers × n` 限制在途解码序列。`n=8`、上限 256 choice 时，请求 worker 自动限制为 32；连接、timeout、429 与 5xx 使用稳定退避等待自动重启，非可恢复错误立即失败。候选复用必须同时匹配模型、采样参数、seed 与 `prompt_payload_hash`。

## 3. 迭代结果

| 版本 | 策略 | Prompt 变化 | Evidence 变化 | 平均 Token 增量 | candidate-1 | Best-of-8 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| v21 | 冻结基线 | -- | -- | -- | 1357 (88.12%) | 1387 (90.06%) | 基线 |
| v22 | 全标签呈现并追加 obligation evidence | 693 | +331 / -0 | +58.64 | 1351 (87.73%) | 1389 (90.19%) | 单回答回归，拒绝 |
| v23 | 不扩证据，generic route 多标签呈现 | 284 | 0 / 0 | +8.07 | 1356 (88.05%) | 未作为最终门禁 | 仍低于基线，拒绝 |
| v24 | 收紧 geography，保留多类呈现 | 242 | 0 / 0 | +6.04 | 1358 (88.18%) | 1384 (89.87%) | oracle 回归，拒绝 |
| v25 | 仅结构型契约，排序修复前 | 48 | +1 / -1 | +1.49 | 未计分 | 未计分 | 证据不变量失败，拒绝 |
| v27 | 结构型契约 + 稳定排序 | 47 | 0 / 0 | +1.49 | **1359 (88.25%)** | **1387 (90.06%)** | 通过 |

v27 相对 v21 的 candidate-1 配对结果为 2 gains、0 losses，净增 2/1540（+0.13 pp）；McNemar exact p=0.5，因此该增益只能描述为“无观测回归的小幅改善”，不能声称统计显著。Best-of-8 总分与基线持平，逐题为 1 gain、1 loss；因此只能得出“聚合能力上界未降低”，不能声称每道题的采样集合均单调改善。

宽版 v22 的结果解释了为什么不能以 oracle 单独选版本：它的 Best-of-8 多 2 题，但 candidate-1 少 6 题，新增证据和重复指令降低了单次回答稳定性。最终版本优先满足单回答不回归，并把 oracle 作为第二道门禁。

## 4. 验收与产物

最终 Prompt 与审计：

- `../artifacts/report/v5_81_narrow_queryir/v27_structural_contracts_deterministic/prepared_answers.jsonl`
- `../artifacts/report/v5_81_narrow_queryir/v27_structural_contracts_deterministic/upgrade_audit.json`
- `../artifacts/report/v5_81_narrow_queryir/v27_structural_contracts_deterministic/answer_n8/run_manifest.json`

配对判分与能力上界：

- `../artifacts/report/v5_81_narrow_queryir/paired_judge/v27/paired_delta_manifest.json`
- `../artifacts/report/v5_81_narrow_queryir/v27_structural_contracts_deterministic/gap_audit_official/summary.json`

所有外部 judge 输入只包含 `question`、`reference_answer` 与 `candidate_answer`；source Memory、prepared Prompt、evidence ID、图与数据库均不离开本机。最终代码全量测试为 626/626 通过。

## 5. 结论

本轮可固化的升级不是“向 Prompt 注入更多 QueryIR 标签”，而是把多标签编译与执行职责分离：丰富标签留在内部用于检索和诊断，只有高精度、会改变答案结构的 comparison 与行政层级 geography 契约进入回答界面。该设计以平均 1.49 packing token/题的增量获得 candidate-1 小幅无回归提升，并保持 90.06% Best-of-8 上界；更广的义务呈现和 evidence 扩展均未通过全量门禁，不应作为默认路径。
