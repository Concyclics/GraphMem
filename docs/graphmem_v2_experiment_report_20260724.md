# GraphMem V2 分层状态图实验报告

## 结论

`hierarchical_state_graph_v2` 已完整实现 L0 原文、L1 原子事实、L2 路由卡、L3 状态链，以及带方向、置信度和来源的 typed graph。结构校验、DeepSeek token 预算和关闭 reasoning 均达到要求；冻结盲测没有证明 90% 准确率。

旧版 500 题经固定 Mem0 judge 重判为 385/500（77.0%）。冻结开发集为 72/72，盲测错误集为 26/67（38.81%）。把开发集收益也计入的乐观投影是 459/500（91.8%）；只使用未见盲测错误的恢复率估计约为 85.9%。两者都不是实测 500 题结果。

## 为什么盲测表现明显下降

盲测集不是随机抽取的 67 题，而是旧版 500 题中封存的困难错题，因此 38.81% 表示困难错题恢复率，不能直接解释为完整 500 题准确率。开发集经过多轮规则与算子调优，72/72 明显包含开发集过拟合；24 道旧版正确控制题不足以估计其余 361 道旧版正确题的回归。

41 个盲测错误中，回答推理或格式错误为 25 个（61.0%），召回排序或图扩展错误为 9 个（22.0%），两者合计 82.9%。索引答案词支持率达到 92.15%，但 post-pack 支持率降至 68.10%，source-leaf expansion recall 只有 52.52%。同时 typed expansion 每题平均固定新增 44 个节点，表明扩图撞上候选上限并引入噪声，而非按查询有效补足关系。

因此，结构化图 schema 不是首要失败点。首要问题是开发规则泛化不足、扩图缺乏判别性、正确来源叶没有稳定进入 evidence pack，以及最终 LLM 改写了 ledger 中已计算出的确定性结果。

## 分层图索引诊断

### 合理的部分

- L0 原始轮次保持无损，L1/L2/L3 都能追溯到已有 source leaf。
- 盲测中 source、routing pointer、edge endpoint、state chain 完整性错误均为 0。
- `supersedes/contradicts/before/after` 为有向关系；没有恢复全局相邻日期边。
- 路由卡全部通过 180-token 粗略上限，且在 pack 中保留率为 100%。
- 索引中的 gold-term support 为 92.15%，说明大部分答案线索已经进入索引。

### 需要升级的部分

- session JSON 抽取 parse error 为 10.02%，length finish 为 7.79%；单次大 JSON 抽取仍会丢失长会话事实。
- source-leaf expansion recall 只有 52.52%，L1 命中后没有稳定展开到真正支持答案的 L0。
- typed expansion 每题平均恰好新增 44 个候选，已经成为固定上限扩散；这会引入高阶泛化节点和 semantic noise。
- 两个 temporal scope warning 表明时间关系验证仍需更严格。

因此图 schema 本身是合理的，主要索引问题是抽取鲁棒性和候选边的判别性，而不是缺少更多边。

## 召回与证据打包诊断

- 盲测 gold-session recall 为 85.97%，对多会话和时间题不够。
- pre-pack index support 为 92.15%，post-pack support 降到 68.10%，说明排序、来源展开和 operand 选择造成了主要信息损失。
- packer 对 card/fact/leaf 的保留率分别为 100%/98.93%/97.89%；高保留率与低答案支持并存，说明问题不是简单的 token 裁剪，而是 pack 前选错了证据。
- 相邻 leaf 平均增加 26.45 个，数量偏大；相邻文本挤占了真正 source leaf 和状态链 operand 的优先级。

下一版应先在较大的候选池上执行确定性 operator，再只打包 operator operands、来源叶和少量路由卡，而不是先选 10–14 个事实再计算。

## 回答流程诊断

41 个盲测错误中，25 个被归因为 answer reasoning/format，占 60.98%。这说明即使 ledger 中已经得到较接近的状态、计数或时间结果，最终一次 LLM 调用仍可能覆盖确定性结果。

下一版应把 count/list/latest-state/date-difference/set-update 等算子的结果设为不可改写的 answer constraint，并增加本地输出校验。自由文本事实和偏好解释仍使用一次 DeepSeek；可确定答案直接使用模板渲染或要求模型逐字段复制，不增加第二次修复调用。

## Token 与服务验收

盲测每题构建 token P50/P95/max 为 253,192/273,507/283,685；回答为 7,471/8,836/9,293。构建和回答超限均为 0，reasoning token 为 0。

每次 DeepSeek 调用记录 cache miss input、cache hit input、output、reasoning、stage 和预算归属。judge 单独记录并排除；embedding 完全不进入 DeepSeek 预算。8001 服务仅健康检查，长 L0 的 embedding request view 截断到 8,000 token，存储的 L0 内容不截断。

## V3 升级顺序

1. 用本地确定性抽取补齐数字、日期、否定、动作状态和稀有实体；LLM 仍负责语义原子化。对失败或 length finish 的 session 做预算受控的分块重试。
2. 路由卡增加 rare anchors、精确数字/日期和事实类型计数，继续作为可见召回证据。
3. 将 typed expansion 从固定候选上限改为 query-conditioned edge budget、边际分数停止和高出度惩罚；semantic neighbor 默认最低优先级。
4. 先按路由卡覆盖会话，再在会话内检索 facts；为时间、多跳、更新题设置最小跨会话配额。
5. 在 pre-pack 候选池执行 operator，强制保留所有 operands 及其 source leaf，删除泛化 adjacent leaf。
6. 对确定性 operator 使用受约束输出和本地一致性校验，避免正确 ledger 被模型重写。
7. V3 不再使用本次 blind 单题调规则。建立新的交叉验证或一次性全 500 验收；只有 held-out 通过后才报告达到 90%。

## 验收状态

- 构建最大值 ≤300K：通过。
- 回答最大值 ≤10K：通过。
- reasoning token 为 0：通过。
- judge/embedding 排除预算：通过。
- 图来源与结构完整性：通过。
- retrieval sufficiency ≥95%：未通过（开发集 91.67%）。
- Mem0 judge 实测总准确率 ≥90%：未证明。
