# GraphMem V3.5 全错题审计与下一阶段升级方案（2026-07-28）

## 评估口径

- 冻结基线：V3.4。
- LongMemEval：Mem0 固定 judge prompt；基线 326/500（65.2%），冻结错题 174 道。
- LoCoMo：mem0ai/memory-benchmarks judge；Category 1–4 基线 1328/1540（86.23%），冻结错题 212 道。
- 本轮只回放冻结错题，因此“基线正确题不回退”只能作为上界，不能当作最终全集成绩。
- Backbone/judge：gpt-5.4-mini，thinking disabled；judge token 与 embedding token 均排除在回答预算外。

## 全错题回放结果

| Benchmark | 路径 | 修复 | 剩余 | 回答最大 token | 超限 | reasoning token |
|---|---|---:|---:|---:|---:|---:|
| LongMemEval | coarse Top-8 → local dense turns → deterministic QueryIR → lossless answer | 58/174 (33.3%) | 116 | 8,524 | 0 | 0 |
| LoCoMo | deterministic QueryIR → 10 graph seeds → typed recovery → mixed evidence | 93/212 (43.9%) | 119 | 9,424 | 0 | 0 |

若假设旧正确题全部保留（仅为错误回放上界），LongMemEval 为 384/500（76.8%），LoCoMo 为 1421/1540（92.27%）。该上界没有经过全集控制题回归验证。

## LongMemEval 错误分布

| 官方类型 | 原错题 | 修复 | 剩余 | 修复率 |
|---|---:|---:|---:|---:|
| temporal-reasoning | 65 | 19 | 46 | 29.2% |
| multi-session | 61 | 14 | 47 | 23.0% |
| knowledge-update | 24 | 9 | 15 | 37.5% |
| single-session-preference | 11 | 6 | 5 | 54.5% |
| single-session-user | 7 | 6 | 1 | 85.7% |
| single-session-assistant | 6 | 4 | 2 | 66.7% |

QueryIR 最大剩余簇：count 39、duration 22、lookup 19、earliest 11。粗路由已命中全部 gold 会话的错题为 167/174；新证据包覆盖全部 gold 会话的题为 157/174，但其中仍剩余 102 道错误。因此 LongMemEval 的主瓶颈已经不是粗召回，而是跨会话状态/时间/集合语义和最终执行。

## LoCoMo 错误分布

| Category | 原错题 | 修复 | 剩余 | 修复率 |
|---|---:|---:|---:|---:|
| 1 | 47 | 22 | 25 | 46.8% |
| 2 | 62 | 22 | 40 | 35.5% |
| 3 | 41 | 16 | 25 | 39.0% |
| 4 | 62 | 33 | 29 | 53.2% |

QueryIR 最大剩余簇：lookup 52、date 20、list 17、duration 9。最终证据包覆盖全部官方 evidence 的旧错题只有 85/212；79 道完全未包含官方 evidence，45 道仅部分包含。LoCoMo 的主瓶颈仍有显著的细召回/证据打包成分；但 85 道 evidence 全覆盖题仍剩 40 道错误，说明日期解析和答案执行也需要升级。

## 已确认的架构问题

1. `complete` 证书语义不足。它当前主要证明“所选 operand 的 provenance 收齐”，没有证明 query entity、relation、time scope 和 answer slot 都绑定正确。旧错题中有 86 道 catalog operator 标记 complete，但只有 24 个 operator value 与 gold 有直接词法一致。
2. 语义相似度越过实体边界。已复现 `autographed baseball` 被计为 `autographed football`。本轮已增加通用 sibling-entity 排斥：共享修饰词但核心实体头不同，不能仅靠 embedding 相似度匹配。
3. 细图扩展不是证据角色闭包。当前 typed recovery 会返回很多 graph node，但不保证保留 old/new state、每个计数成员、时间端点、否定对照和相邻问答各自的配额。
4. 最终 pack 仍接近平面排序。相同主题的高分行会占满预算，导致旧状态、第二端点或反证被挤掉；缩小 Top-K 反而使固定集从 23/36 降到 18/36。
5. LLM session selector 不可复现。同配置固定 36 题出现 21–26/36 波动，temperature=0 和 seed 并未消除网关侧方差，因此不应作为最终主控制器。
6. LoCoMo 与 LongMemEval 需要不同的图宽度，但路由条件应由 memory scale、QueryIR 和证据证书决定，不能按 benchmark 名称或 topic 分支。

## V3.6 建议实现顺序

### P0：语义证书重做

把 `complete` 拆为四个独立证书：`entity_binding_complete`、`relation_binding_complete`、`scope_closure_complete`、`provenance_complete`。只有四者同时满足才允许 authoritative operator。所有 operator 保存正证据、反证据、未覆盖候选和 source span；embedding 只负责召回候选，不能独立通过实体绑定。

### P0：Evidence Role Packer

由单一 Top-K 改为固定角色槽：query-direct lossless、old/new state、temporal endpoints、collection members、negative/near-match contrast、adjacent reply、coarse routing card。每个槽先去重再分配 token；最后才用 MMR 填充余量。count/list 必须输出 collection-closure manifest，duration/ordering 必须输出 endpoint manifest。

### P0：状态链与事件链重建

状态链 key 从 `(subject,predicate,context)` 扩展为 `(owner,entity,attribute,context)`；每个更新边记录 operation、effective time、observation time、polarity、modality。时间边只在共享 event identity 或显式因果/参与关系下建立。计划、建议、完成和取消使用不同 event status，禁止相互覆盖。

### P1：确定性 coarse-to-fine 控制器

第一层取 routing card / BM25 / dense / entity index 的 RRF；第二层按 QueryIR 选择关系模板；第三层只在候选 session 内用 embedding 找 lossless turn。小 memory + lookup 可用较宽 typed closure；大 memory + state/count/time 必须走 role closure。条件只使用 memory size、IR 和证书缺口。

### P1：日期与数量 algebra

统一 relative-time anchor、inclusive/exclusive interval、month/day unit、target-current、add/remove/replace、distinct occurrence。算子先输出 operands 和单位，再输出值；任何未绑定 operand 都使证书 incomplete。LoCoMo duration 旧错题本轮 0/9 修复，应作为首个独立验收簇。

### P1：安全替换门

新答案只有在证书完整且 lossless evidence 支持时替换 V3.4；证书不足时走旧路径。先在全部冻结错题 + 分层正确控制集上学习/冻结阈值，再一次性跑完整全集。不能根据盲测单题继续改规则。

## 下一轮验收门槛

1. LongMemEval 错题回放至少修复 100/174 后才值得跑全集；目标 125 需要状态/时间/集合算子重建，当前检索调参不够。
2. LoCoMo 错题回放已经修复 93/212，但必须在至少 200 道分层正确控制题上将回归压到 20 题以内，才可能稳定超过 90%。
3. 每题回答 max 不超过 9.5K 的内部安全线，硬上限 10K；reasoning 0；embedding/judge 独立统计。
4. 删除所有 benchmark 名称、topic 词和题目 ID 分支；新增跨域 synthetic sibling/state/time/list 测试。
