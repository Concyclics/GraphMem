# GraphMem V3 分层超图子集实验报告（2026-07-25）

## 结论

V3 已完成 role-neutral L0/L1/L2/L3 数据结构、会话级抽取、重叠主题、带 provenance 的超边、dense/BM25/exact RRF、多层直接种子、双向 node↔hyperedge 扩展、查询聚焦打包、局部时间/计划 hint，以及 DeepSeek 分阶段 token 硬门。

当前结果证明 V3 在两个 benchmark 上都能工作，并且 token 预算充足；但尚不能据此宣称全集 90%。LoCoMo 固定顺序小子集已经达到 95%，LongMemEval 固定 12 题统一回放为 83.3%，主要剩余问题是高度相似 distractor session 的 scope discrimination，而不是证据压缩或预算不足。

## 实验选择

- LongMemEval：按原始顺序、每类前 2 题组成固定 12 题，不按 topic 或 V3 对错选择。
- LoCoMo：conversation 0 的前 20 题，保持原始顺序，包含 category 1/2/3。
- Backbone 与 judge：`deepseek-v4-flash`，thinking disabled。
- Embedding：本地 `Qwen3-Embedding-0.6B`，1024 维。
- Judge：LongMemEval 使用固定 Mem0 prompt；LoCoMo 使用 `mem0ai/memory-benchmarks` prompt。
- Judge 与 embedding token 均排除在构建/回答预算之外。

## LongMemEval 结果

统一 12 题回放目录：

`runs/v3_20260725/lme_smoke12_v3_final`

- Mem0 judge：10/12，83.3%。
- 构建总 token 最大值：219,594。
- 回答总 token 最大值：9,497。
- 构建/回答预算超限：0。
- reasoning token：0。

后续局部 operator 修复：

- 最新 scalar：从已打包时间账本按真实 session date 选出 `25:50`，记录三个 source node；聚焦回放通过。
- duration：修复 `how many days` 分类、同日证据去重和相对时间锚定后输出 `7 days`；聚焦回放通过。
- count/list：增加数目与列举一致性、模态过滤和显式谓词约束。

仍未稳定解决的 hard negative：

- “projects led/currently leading”在多个 distractor session 中存在多组语义近似项目证据。继续围绕该题添加实体/topic 规则会造成 benchmark 过拟合，因此保留为 scope discrimination 问题，后续应通过 query-conditioned episode/session posterior 和闭包置信度解决。

## LoCoMo 结果

统一 20 题回放目录：

`runs/v3_20260725/locomo_conv0_first20_final`

- Memory Benchmarks judge：19/20，95%。
- 官方 token-F1（初始 20 题回放）：0.4193；该指标对简短同义答案较敏感，仅作辅助口径。
- conversation 级构建：55,518 token。
  - cache miss input：1,485。
  - cache hit input：29,312。
  - output：24,721。
- 单题回答最大值：8,667。
- 单题回答 P50：7,874.5。
- 构建/回答预算超限：0。
- reasoning token：0。

统一回放唯一错误是 planned-date 模态选择；加入 L0 lossless 回退后，局部 hint 为：

- `event_time`: `next month`
- `anchor_date`: `25 May 2023`
- source：原始 Melanie turn

聚焦回放输出 `June 2023` 并通过 judge。

首轮 4 个 LoCoMo 错误的通用修复包括：

- `where` → location answer contract。
- `What do X like?` → explicit preference list。
- counterfactual → causal counterfactual contract。
- `last week` 等相对时间使用 session date 锚定。
- possessive、复数、`-ing/-ed` 基础归一化。

## 图与召回审计

- L0 turn 始终 lossless；坏 JSON/空抽取不会丢原文。
- L1 claim/event 必须引用现有 source turn。
- L2 episode 和重叠 theme 同时参与 dense/BM25/exact 检索，不只用粗图选点后向下展开。
- 查询从 turn/claim/event/episode/theme/hyperedge 同时取 seed。
- 扩展沿 incidence 双向执行，深度上限 2；trace 记录每一步 `via_hyperedge` 和 graph rescue。
- 打包采用查询词 set-cover，避免一个子事件占满强保护配额。
- temporal/planned hints 只读取打包后的局部证据，不扫描全局索引。

已观察到的召回缺陷：

- LoCoMo “Where has Melanie camped?” 没有召回包含 beach 的 session 6，但召回了 mountains/forest；judge 接受部分答案，retrieval completeness 仍不足。
- 当前超边扩展已经真实使用关系，但 relation 选择仍主要依靠通用 query overlap/temporal gain，尚需加入 domain-neutral 的 relation-type posterior 和分支停止条件，减少大型 theme/participant 分支噪声。
- JSON 全量缓存反序列化和每题重复建立本地排名结构是当前延迟瓶颈；高问题并发会放大 CPU/内存开销。

## 反过拟合约束

- V3 core 静态测试禁止出现 `longmemeval`、`locomo`、category ID、question ID 分支、topic rule table。
- 新规则只处理语言操作和证据状态：location、duration、ordering、counterfactual、preference list、planned date、latest scalar。
- 新测试使用相互无关的相机/兰花/窑炉/徒步等替换场景验证 metamorphic behavior。
- 当前全套测试：275 passed。

## 下一步

1. 为相似 distractor 引入 query-conditioned episode/session posterior；闭包只在高置信 scope 内完成。
2. 为超边关系加入 domain-neutral relation posterior、每关系 beam 配额和 closure-aware stop。
3. 将 BM25、dense matrix、incidence adjacency 持久化并按 conversation 共享内存映射。
4. 固定参数后扩展到 LongMemEval 50 题和 LoCoMo 200 题盲测；不得按单题再修改规则。
5. 记录每批 build/answer 的 miss/hit/output/max，并分别做 graph expansion on/off、coarse on/off 消融。
