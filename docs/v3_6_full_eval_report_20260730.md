# GraphMem V3.6 全集评估与错误归因（2026-07-30）

## 结论

V3.6 的 500 题 LongMemEval 与 1,986 题 LoCoMo 回答均已完整生成并通过唯一 ID 校验。固定 judge 完成后，LongMemEval 为 **328/500（65.60%）**，LoCoMo Category 1–4 为 **1,224/1,540（79.48%）**。两者均未达到 90% 目标，因此当前版本只能视为完整、可追溯且预算基本受控的研究基线，不能作为已达标版本发布。

## 准确率

### LongMemEval

| 题型 | 正确/总数 | 准确率 | gold session 全命中率 | 错题中 gold session 已全命中 |
|---|---:|---:|---:|---:|
| knowledge-update | 54/78 | 69.23% | 97.44% | 23/24 |
| multi-session | 75/133 | 56.39% | 84.21% | 44/58 |
| single-session-assistant | 51/56 | 91.07% | 100.00% | 5/5 |
| single-session-preference | 17/30 | 56.67% | 86.67% | 9/13 |
| single-session-user | 59/70 | 84.29% | 98.57% | 11/11 |
| temporal-reasoning | 72/133 | 54.14% | 85.71% | 51/61 |

总体 gold session 全命中 453/500；在全命中题上准确率仍仅 68.43%，未全命中题为 38.30%。172 道错题中有 143 道已命中全部 gold session，说明主要瓶颈不是 coarse routing，而是 session 内细粒度证据绑定、状态/时间/集合组装和最终回答。

最弱题型是 temporal-reasoning（54.14%）、multi-session（56.39%）和 single-session-preference（56.67%）。single-session-assistant 达到 91.07%，说明 lossless 对话回复在目标明确时有效。

### LoCoMo（memory-benchmarks Category 1–4）

| Category | 正确/总数 | 准确率 | gold session 全命中率 | 错题中 gold session 已全命中 |
|---|---:|---:|---:|---:|
| 1 | 199/282 | 70.57% | 44.68% | 29/83 |
| 2 | 239/321 | 74.45% | 91.28% | 58/82 |
| 3 | 51/96 | 53.12% | 55.21% | 23/45 |
| 4 | 735/841 | 87.40% | 96.91% | 90/106 |

Category 3 推断题最弱（53.13%）；Category 1 事实/集合类仅 70.57%，其 gold session 全命中率只有 44.68%，是明确的细粒度召回与人物归属问题。Category 2 和 4 的 session 命中较高，但仍有大量错误，表明正确 session 内选错事件、时间或对象。

| Conversation | 正确/总数 | 准确率 |
|---|---:|---:|
| locomo00 | 129/152 | 84.87% |
| locomo01 | 74/81 | 91.36% |
| locomo02 | 122/152 | 80.26% |
| locomo03 | 150/199 | 75.38% |
| locomo04 | 134/178 | 75.28% |
| locomo05 | 98/123 | 79.67% |
| locomo06 | 115/150 | 76.67% |
| locomo07 | 152/191 | 79.58% |
| locomo08 | 124/156 | 79.49% |
| locomo09 | 126/158 | 79.75% |

## Token 与预算

| Benchmark | 阶段 | 平均 | P50 | P95 | 最大 | 超目标/异常 |
|---|---|---:|---:|---:|---:|---:|
| LongMemEval | 构建/题 | 241325 | 240599 | 264993 | 294011 | >300K: 0 |
| LongMemEval | 回答/题 | 10090 | 10038 | 11952 | 12748 | >10K: 255; >10.5K: 211; >12.1K: 11 |
| LoCoMo | 回答/题 | 6448 | 6066 | 8992 | 12729 | >10K: 27; >10.5K: 15; >12.1K: 1 |

LongMemEval 回答平均 10,090 token，接近约 10K 的放宽目标，但 11 题超过 12.1K，最大 12,748；LoCoMo 平均 6,448，只有 1 题超过 12.1K，最大 12,729。实际执行保险上限临时设为 15K，所有 >12.1K 题仍按异常报告，没有计为预算通过。

LongMemEval 构建最大 294,011，500 题全部低于 300K。LoCoMo 每组 conversation 构建如下：

| Conversation | cache miss 输入 | cache hit 输入 | 输出 | 总计 |
|---|---:|---:|---:|---:|
| locomo00 | 51579 | 0 | 25535 | 77114 |
| locomo01 | 43741 | 0 | 23622 | 67363 |
| locomo02 | 79130 | 0 | 40420 | 119550 |
| locomo03 | 71080 | 0 | 36463 | 107543 |
| locomo04 | 70501 | 0 | 37358 | 107859 |
| locomo05 | 69808 | 0 | 34340 | 104148 |
| locomo06 | 71106 | 0 | 38609 | 109715 |
| locomo07 | 76261 | 0 | 42879 | 119140 |
| locomo08 | 64335 | 0 | 31167 | 95502 |
| locomo09 | 75109 | 0 | 37286 | 112395 |

所有 backbone 与 judge reasoning token 均为 0；题级 token accounting 全部有效。Judge 调用全部标记 excluded_from_budget=true。LongMemEval 抽取 parse error 为 1/23,867 session，LoCoMo 为 0/272，均低于目标。

## 固定 80 题与全集不一致

锁定弱类 80 题独立跑为 67/80（83.75%）；同一批 ID 在本次全集重建中为 60/80（75.00%），其中 multi-session 31/40、temporal 29/40。两次运行有 29/80 的 prediction 变化；相同 prediction 下只有 1 次 judge 不一致。净回退为旧正确新错误 8 题、旧错误新正确 1 题。因此主要问题是抽取/索引重建的非确定性，而非 judge 随机波动；开发集也高估了全集分布。

## 主要错误机制

1. **Completeness certificate 只验证角色存在，不验证语义绑定。** LongMemEval 172 道错题中约 98 道被标为 complete；LoCoMo 316 道错题中约 252 道被标为 complete。证书会把来自相邻事件或同一人物的错误 value/time 当作完整证据。
2. **命中 session 后没有可靠锁定目标 source span。** LongMemEval 83.1% 的错题已经命中全部 gold session，但仍会输出错误值或 abstention。当前 fine ranking 会把同 session 的多个 RoleFrame、EvidenceGroup 和 turn 混入，回答器容易选到旧状态或相邻事件。
3. **状态更新、集合和时间比较没有形成封闭证据域。** 典型错误包括 mortgage 额度选旧值、5K 成绩选错版本、集合漏成员、时间差绑定到错误端点。图边存在，但 expansion 只保证取到某类角色，不能保证角色属于同一 event identity 和同一 scope。
4. **LoCoMo 的人物归属和指代仍弱。** Category 1 session 全命中率仅 44.68%；两人对话中名字、speaker、listener 与事实 owner 的绑定仍会丢失，导致人物事实被路由到另一方或未进入候选。
5. **回答 prompt 过度保守且证据排序噪声高。** 多个 gold session 已命中且 certificate complete 的题仍回答 memory does not establish。相反，在推断题和 preference 题中又会抓取无关类比，说明 abstention 与推断阈值没有由可验证 source span 控制。
6. **80 题调优存在结构性过拟合，但未发现 topic/题号硬编码。** 静态扫描未发现 LongMemEval、LoCoMo、gold 字段、题号或具体 topic 分支；问题来自参数和结构在小开发集上选择后未覆盖全集多样性，而不是显式单题规则。

## 下一阶段建议（保持跨 benchmark 通用）

1. 将 certificate 从 role-presence 改为 **source-span binding certificate**：每个 target entity、relation、value、time、scope 必须指向同一 event identity 或 state chain 的可引用原文 span；不同事件的角色不得拼接成 complete。
2. 在 coarse card 命中后增加确定性的 **session-local lossless re-ranker**：对选中 session 的所有原始 turn 做 exact/FTS/dense 融合，再按 entity-relation-time 联合约束选 span；RoleFrame 只导航，不拥有最终事实优先权。
3. 对 update/temporal/count 建立 **candidate set first** 流程：先收齐同一实体+属性+上下文的全部版本/成员/端点，再执行 latest、diff、distinct；禁止从单个最高分 frame 直接作答。
4. 为 LoCoMo 增加通用 speaker graph：显式存储 speaker、listener、mentioned person、possessor，并用 dialogue_pair/reference 链做局部 coreference；不引入人物名或 benchmark 规则。
5. 重新校准 abstention：只有 source-span certificate 失败才允许 abstain；certificate 通过时给回答器紧凑的唯一候选表，避免在 8K 噪声上下文中重新选择事实。
6. 消除构建非确定性：缓存 session extraction 作为正式可复现实验资产；validator 只修复缺失 span，不允许整 session 重采样；对 alias/event identity/state chain 使用确定性本地归一化。
7. 下一轮开发集必须从两个 benchmark 联合分层抽样，并保留完整盲测；参数选择以跨运行稳定性、source-span recall 和错误上界为先，不再只看单次小样本 judge 分数。

## 产物

- 机器可读汇总：runs/v3_6_locked_20260730/v3_6_full_eval_report_20260730.json
- LongMemEval：runs/v3_6_locked_20260730/lme500/hierarchical_role_graph_v3_6/
- LoCoMo：runs/v3_6_locked_20260730/locomo1986_complete/hierarchical_role_graph_v3_6/

