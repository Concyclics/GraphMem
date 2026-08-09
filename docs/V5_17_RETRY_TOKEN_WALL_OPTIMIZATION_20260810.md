# V5.17 构建 Retry、Token 与 Wall-Time 优化审计

日期：2026-08-10

## 结论

V5.10 严格原子抽取的大量 retry 并非服务不稳定，而是构建协议内部的三个问题叠加：原子扫描误报、事实预算与 coverage 合同不相容，以及并发前预算低估。V5.17 将 retry 从构建主路径移除，以窄原子合同、批量抽取和 raw-source fallback 保证可追溯性。在同一排序的前 16 个 memory 上，全新重建结果为：

| 指标 | V5.10 审计基线 | V5.17 最终 16 题 |
|---|---:|---:|
| 平均 Token / memory | 455,608 | 191,921 |
| 最大 Token / memory | 473,600（14 个已完成样本） | 199,344 |
| retry calls | 568 / 14 memories | 0 / 16 memories |
| LLM calls / memory | 约 87.4 | 16.6 |
| 构建 wall time | 10 个完成需 15.2 min | 16 个完成需 5.8 min |
| memory throughput | 0.66 / min | 2.76 / min |

最终运行 16/16 成功，Token mean/p50/p95/max 分别为
191,921 / 191,406 / 196,464 / 199,344，全部低于 230K。

## Retry 根因

1. 旧原子扫描器把 assistant 长回复中的列表序号、模态词 `may` 和普通句首大写词当作必须覆盖的单元。V5.10 已完成样本中，每 scene 平均 66.9 个原子单元，95% 来自 assistant；entity 占 61.5%。
2. V5.10 每 scene 自适应事实上限常达到 24，但仍要求至少 95% 单元被 fact 或 unresolved 覆盖。655 个主请求中 568 个触发 retry，触发率 86.7%。
3. retry 的旧择优分数首先最小化 missing，因此把 missing 改写为 unresolved 即可“获胜”，即使没有增加图上的可导航事实。审计中，retry 减少的 10,146 个 missing 里有 9,171 个只是转成 unresolved。
4. 旧预算只按原文字符和 900 expected output 预留，未计入长 system prompt、原子标注及真实输出。实际/预留比主请求均值为 2.07，retry 为 2.45；256 并发会在任何请求 settle 前一起放行，从而突破预算。

## V5.17 方法

- 过滤 numbered-list marker、模态 `may` 与句首普通大写词；原子合同仅保留 date、duration、number、negation、quoted item。
- 采用 12-scene strict batching、每 scene 最多 2 facts、单批最多 3000 output tokens；不生成重复的模型 scene summary/entity list，summary 从已验证 facts 确定性编译。
- unresolved 只输出 unit ID，不重复自然语言理由。
- `semantic_max_retries=0`。未被原子事实覆盖的来源由 raw-source fallback 标记，查询端仍可通过 scene、FTS 与 embedding 到达原文。
- 并发前使用完整 system + serialized payload + 最大输出上限做 hard reservation，安全因子为 1.02；16 个审计 memory 的离线最坏预留为 224,866 Token。
- retry 机制本身仍保留可配置的 critical-kind admission 与 grounded-gain acceptance，供后续消融使用；它不能再靠 missing→unresolved 获胜。

## 校准过程

32-scene / 3-fact 校准被提前终止：34 个打满请求平均只返回
15.5/32 scenes，继续运行只会诱发逐 scene repair。

16-scene / 2-fact 单题校准完成于 1.1 min、230,895 Token，但包含
17 次 retry。其主抽取本身只消耗约 200K；retry 额外消耗 30,916 Token。
解析后的 scene 原子 coverage 均值仅 8.9%，证明在 2-fact 预算下用 80% coverage
触发 retry 是不可满足合同，而非值得再次调用 LLM 的偶发失败。

最终 12-scene / 2-fact 单题门禁为 0.7 min、192,023 Token、0 retry、
0 truncation、204/204 scenes 完整解析；随后才扩到 16 题。

## 最终运行与剩余风险

最终数据库：
`../artifacts/report/v5_17/budget230_retry_optimized_dev16_v3_20260810/graph/graphmem.sqlite`

报告：
`../artifacts/report/v5_17/budget230_retry_optimized_dev16_v3_20260810/build_report.json`

266 个主请求中 12 个触及 3000 output-token 上限，截断 batch 平均返回
9/12 scenes；全局约 98.85% scenes 被主抽取完整解析。截断尾部使用确定性 fallback，
不会再触发 LLM retry。下一步准确率消融应比较当前 latency profile 与
`batch=10, output=2600` 的 completeness profile；只有后者在 hard-set answer
accuracy 上显著更好时，才接受其额外固定 prompt 与请求数。
