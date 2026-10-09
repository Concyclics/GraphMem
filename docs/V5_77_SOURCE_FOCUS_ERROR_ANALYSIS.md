# V5.77 自适应召回与 Best-of-N 错误分析

## 结论

V5.76 的 label-free 两阶段候选池在 LoCoMo 1,540 题上的 evaluator-only
Best-of-N 上界为 1,404/1,540（91.17%）。在现有候选哈希可追溯的 1,400
道正确题与 140 道未解题上，新增 32-turn source-focus 独立视图又找回 34
题，其中 29 题位于当前自适应扩展门内。因此：

- 保持现有门控时，候选可用率的保守区间为 92.79%--93.05%；
- 对全部请求提供 focus 候选时，保守区间为 93.12%--93.38%。

区间来自旧实验中 4 道题的候选文本缺少可直接关联的 judge hash；在补齐全量
focus 产物和这 4 道 verdict 前，不将区间端点表述成正式单点成绩。运行时门控和
候选生成均不读取 gold、reference answer 或 judge；Best-of-N 是候选可用率上界，
不是在线 selector 的单答案准确率。

## 错误归因

对 140 道可追溯未解题进行 gold evidence 隔离：

- 97 道可在只提供标注证据时被 Best-of-8 答对；其中 64 道原 pack 的证据不全，
  32 道原 pack 已 all-hit，主要损失来自布局、噪声或答案采样，另 1 道无证据标注；
- 43 道在 gold-only 的八个回答中仍未答对，不能直接归为检索失败；
- 给 gold 无条件增加相邻对话后只答对 95 道。闭包新增救回 10 道，但同时使 12
  道 gold-only 可答题失败，说明邻居应由 query-focused seed 定向触发，不能全局扩张。

原候选池中共有 180 道题在 Best-of-N 内同时出现正确与错误回答。123 道已经
all-hit，说明其中一部分是模型采样/执行波动；另有 57 道证据不完整。正确候选
只在 expanded 视图出现的有 89 道，只在 base 视图出现的有 17 道，两种视图都能
答对的有 74 道。因此波动也明显依赖检索视图和上下文组织，并非单纯模型随机性。

## 修复

新增 `build_source_focus_plan`，直接在不可变原始 turn 上执行 BM25、短语、显式
speaker 和稀有词排序，再只对高分 seed 补齐相邻问答，并按 session/source order
成组呈现。该通路不读取图分、抽取事实、先前答案或评测标签，用作图检索之外的
独立候选视图，不替换原图证据。

新增 `build_source_focus_prompt`，只呈现 source-focus turn、QueryIR operator 和
对应回答义务，避免把 previous answer、图排名或未经验证的 worksheet 值作为事实。
在 140 道未解题的计算捷径实验中，该视图找回 34 道：20 道 focus pack all-hit、
7 道 partial、7 道按官方标注为 no-hit。后 7 道表明部分官方 evidence ID 只覆盖
对话的一侧，source 邻居可提供真实回答所需的补充上下文。

## 正式验收

当前结果足以证明候选层可以越过 90%，但正式报告单点仍需：对门控覆盖的全部题
物化 focus 候选；补 judge hash 缺口；生成统一 manifest；最后评测不使用 judge 的
在线 selector。source-focus 只应在 QueryIR/闭包风险、候选分歧或 verifier 低置信度
时进入候选池，以控制平均 Token 和延迟。

## 同一最终 Prompt 的 Best-of-N 对照

为隔离回答模型采样，另构造了每题唯一且冻结的 Graph+source-focus Prompt，并在
一次本地 API 请求内设置 `n=8`。所有八个回答共享完全相同的 system prompt、证据
ID、顺序和 payload hash，Luna-medium 只承担离线判分。

第一版平均划分 Graph 32 / focus 32 个席位，因截断 expanded graph 后半段，
Pass@1 为 83.83%，Best-of-8 为 86.43%。修复后完整保留最多 64 个 Graph turns，
再追加最多 16 个去重 focus turns；其结果为：

| N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Accuracy | 85.13% | 86.56% | 86.95% | 87.14% | 87.53% | 87.60% | 87.79% | 87.92% |

最终仍错误的 186 题中，86 题 all-hit、32 题 partial、68 题 no-hit。该实验否定了
“约 93% 完全是同一 Prompt 下回答采样上界”的解释：多视图候选池的提升不仅来自
模型随机性，也来自不同证据预算和不同读取布局避免互相干扰。因而正式系统应把
Graph、expanded 与 source-focus 保持为隔离的物理读取视图，再由 label-free
selector/verifier 合并决策；不应将三者简单拼成一个更长的在线 Prompt。
