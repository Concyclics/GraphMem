# GraphMem V3.6 单版本 90% 重构结论

## 单版本约束

V3.6 的构建索引、查询规划、图扩展、证据打包和回答必须来自同一套实现。
禁止读取 V2/V3.1/V3.4 的答案候选或按题拼接不同版本结果。旧版本只用于离线诊断。

## 复现实验结论

当前 V3.4 图的 LongMemEval 固定开发集由 24 道旧错题和 12 道旧正确控制题组成。

| 实验 | 错题修复 | 控制通过 | 最大回答 token | 结论 |
|---|---:|---:|---:|---|
| self proposal A | 9/24 | 11/12 | 8,863 | 两阶段本身无收益 |
| graph recovery B | 9/24 | 11/12 | 8,673 | 修复了“多数题未走图”的 bug，但图边语义不足 |
| role packer C | 11/24 | 9/12 | 8,924 | 跨会话覆盖提高，但干扰增加 |
| certified pack I | 13/24 | 10/12 | 9,635 | provenance pinning 有效但不足 |
| semantic certificate J | 15/24 | 7/12 | 9,525 | 候选答案锚定导致严重回退 |
| 10.5K semantic certificate K | 11/24 | 9/12 | 10,070 | 增加 token 不能解决不稳定性 |
| session route + lossless L | 11/24 | 10/12 | 10,293 | 前置 LLM 路由不稳定 |
| deterministic one-call M | 11/24 | 10/12 | 10,012 | 当前图上的稳健上限仍不足 |

旧 V3.5-M 曾得到 15/24、11/12，但在相同模型和固定 judge 下无法稳定复现，
不能作为 90% 验收依据。当前实现不能诚实声称达到单版本 90%。

## 已确认的结构问题

1. `session_dense_rerank` 曾错误关闭绝大多数 typed graph recovery；现已修复。
2. 关系图可以命中正确会话，但边主要连接“相似节点”，不能保证主体、关系、状态、
   数值、单位、时间端点与来源作为不可拆证据组进入 prompt。
3. catalog/operator 已给出完整 source IDs 时，packer 仍可能只保留一个 operand；
   已加入 query-bound provenance hydration 和 pinning。
4. lexical graph recovery 每题扫描整个 question scope，增加并发后 CPU 成为瓶颈；
   必须改为持久 BM25/倒排表。
5. LLM answer proposal 会形成错误锚定，且当前兼容端即使 temperature=0/seed=0
   仍有可见非确定性。默认架构不应依赖前置模型猜答案。
6. V2 的高分依赖大量 topic-specific operator 和匹配规则，不能迁移为最终方案。

## 必须重建的唯一 V3.6 索引

### RoleFrame / EvidenceGroup

每个事实不再只是独立 claim，而应形成可检索、可验证的角色超边：

- `frame_id`, `relation_key`, `semantic_type`
- `actor/owner`, `predicate`, `patient/item`, `context`
- `polarity`, `modality`, `lifecycle_status`, `state_op`
- `quantity`, `unit`, `multiplier`
- `event_time`, `observation_time`, `start/end`
- `source_turn_ids`
- `completeness_mask` 与 `provenance_complete`

同一事实的角色成员和原文来源作为一个 `EvidenceGroup` 原子打包，不能被 token
截断拆开。状态更新、集合、时间区间和比较分别建立有向超边。

### 通用查询代数

删除 benchmark/topic 名称分支，查询只编译到以下通用操作：

`SELECT → FILTER → PROJECT → GROUP → DEDUP → ORDER → REDUCE`

其中 `REDUCE` 只允许通用的 `count/list/sum/argmax/diff/duration/latest/exists`。
操作器只读取 RoleFrame 字段和完整性证书，不匹配 airline、fitness、品牌或题目 ID。

### Coarse-to-fine 召回

1. routing card dense + BM25 + entity/relation/date/state 倒排，RRF 选择 4–8 个会话。
2. 在命中会话内检索 RoleFrame/EvidenceGroup，而不是扁平 turn top-k。
3. 从组节点沿 `source/state_history/same_event/temporal/collection/reference`
   做 typed best-first 两跳扩展。
4. 一个组被选中时，完整 pin 所有必需角色和 source turns。
5. 最终只调用一次回答模型；确定性 QueryIR 和证书不消耗 LLM token。

### 预算

- 正常目标：每题回答总量 ≤10,000。
- 临时硬上限：≤10,500；所有 >10,000 题必须单列。
- 最终输入按 provider 实际前序 token 与保守 tokenizer 误差动态打包。
- embedding 与 judge 继续排除，reasoning token 必须为 0。

## 下一轮门槛

先重建固定 36 题涉及的会话索引，但不得把问题或答案送入构建提示。
开发集至少达到错题修复 18/24、控制通过 11/12，且无 topic/ID 分支，才允许：

1. 跑 LongMemEval 封存错题；
2. 跑 LoCoMo 分层控制子集；
3. 最后各跑一次全集并用固定 benchmark judge 验收。

