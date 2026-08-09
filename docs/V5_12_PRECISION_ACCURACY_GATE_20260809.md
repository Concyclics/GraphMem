# V5.12：从 Gold Coverage 到 Recall–Precision–Answer Gate

## 1. 为什么旧指标会误导

旧评测主要报告 candidate gold coverage、turn all-hit 和 recall。当候选池接近整段
memory 时，这些指标会自然饱和：dev200 中候选池平均返回 510.0 个 turn，占 memory
的 91.71%，因此 candidate all-hit 达到 99.50%、recall 达到 99.88%。但相对官方
gold turn 的 precision 只有 0.47%，F1 只有 0.94%。这不能说明检索排序足以支持一个
有界的答案上下文。

官方 evidence 标注是 sufficient、未必 exhaustive。因此本文中的 precision 是
**annotation-scoped lower bound**：未标注但真实有用的 turn 会被当成 false positive。
它不适合解释为绝对相关率，但适合同一批标注上的配对方法比较。

## 2. 新评测契约

对 gold turn 集合 $G$ 和返回集合 $R$，统一报告：

\[
P=\frac{|G\cap R|}{|R|},\qquad
R_c=\frac{|G\cap R|}{|G|},\qquad
F_1=\frac{2PR_c}{P+R_c}.
\]

评测拆成五层，不能再用后一层的大池 coverage 替代前一层的质量：

1. **Graph reachable**：原文 gold 是否存在可追溯图路径，衡量 source→index 损失。
2. **Full candidate reservoir**：仅作候选上界和诊断，不作为成功指标。
3. **Ranked candidate**：P/R/F1@8、16、32、64、128，外加 mAP、MRR、
   R-Precision、nDCG@8/16/32、last-gold-rank。
4. **Final evidence pack**：macro/micro precision、recall、F1、all-hit、turn 数、
   evidence/prompt token、candidate→pack recall loss 和 precision gain。
5. **Answer conversion**：同一 judge 下的逐题 0→1/1→0 转移、McNemar exact test、
   paired bootstrap CI，并按 benchmark/stratum 分层。

建议以答案准确率为最高优先级，采用约束 Pareto gate，而不是只最大化 F1：

- answer accuracy 不出现显著下降，且没有题型级系统回退；
- evidence/prompt token 至少下降 20%；
- query latency 的 paired CI 不证明显著恶化；
- 同时公开 precision、recall、F1 和 all-hit，不允许只挑一个指标。

## 3. dev200 Recall–Precision 曲线

候选排名本身的质量为：mAP 28.60%、MRR 36.74%、R-Precision 21.01%，
nDCG@8/16/32 分别为 32.47%/35.77%/38.71%。

| 工作点 | 平均 turn | All-hit | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Top-8 | 8.0 | 29.50% | 41.63% | 10.06% | 15.44% |
| Top-16 | 16.0 | 40.00% | 51.96% | 6.31% | 10.85% |
| Top-32 | 32.0 | 50.00% | 62.63% | 3.94% | 7.25% |
| Top-128 | 128.0 | 67.50% | 80.53% | 1.37% | 2.67% |
| Full reservoir | 510.0 | 99.50% | 99.88% | 0.47% | 0.94% |

这条曲线说明当前的主要瓶颈是**前 32 名内部的排序质量**。扩大池子会提高 recall，
但 precision 单调下降，且不会自动转化为答案准确率。

## 4. Precision-aware packer 的三轮结果

第一版将 direct/temporal/exhaustive 的上限压到约 12/16/24，并做在线 MMR。虽然
precision 和 F1 上升，但 recall 下降 9.9pp，LongMemEval temporal 答案准确率下降
8pp，证明统一激进截断不安全。

第二版用 QueryIR answer kind 加查询时间词识别 temporal reasoning，将其下限提高到
24。最终版进一步删除在线 MMR：同预算逐题诊断中，MMR 每题只净救回 0.005 个 gold，
却增加约 89ms。最终实现保留 fused ranking、最小 obligation floor 和 query-aware
bounded stopping。

最终离线检索 gate：

| Arm | Turns | Evidence tokens | Latency | All-hit | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32.0 | 2428.0 | 254.9ms | 49.50% | 62.38% | 3.92% | 7.22% |
| precision bounded | 21.8 | 1793.4 | 263.0ms | 43.50% | 56.07% | 5.06% | 8.97% |
| delta | -10.2 | -26.1% | +8.1ms | -6.0pp | -6.31pp | +1.14pp | +1.75pp |

Latency 的均值差为 +8.05ms，95% paired CI 为 [-54.92ms, +67.26ms]，不能认为存在
显著回退。去掉 MMR 后，pack 与同条数 Top-N 的 selection churn 仅 0.02%。

## 5. Answer conversion

在 source-time prompt 相同、Qwen3-30B answer/judge 相同的 200 条配对实验中，第二版
（仍含 MMR）的结果如下：

| Scope | Baseline | Precision arm | Delta | 95% paired CI |
|---|---:|---:|---:|---:|
| Overall | 50.50% | 52.50% | +2.00pp | [-3.50pp, +7.50pp] |
| LongMemEval | 44.00% | 43.00% | -1.00pp | [-9.00pp, +7.00pp] |
| LoCoMo | 57.00% | 62.00% | +5.00pp | [-2.00pp, +13.00pp] |
| LME multi-session | 34.00% | 38.00% | +4.00pp | [-8.00pp, +16.00pp] |
| LME temporal | 54.00% | 48.00% | -6.00pp | [-16.00pp, +4.00pp] |
| LoCoMo multihop | 56.00% | 66.00% | +10.00pp | [0.00pp, +20.00pp] |
| LoCoMo temporal | 58.00% | 58.00% | 0.00pp | [-12.00pp, +12.00pp] |

总体 +2pp 尚不显著，不能表述为 accuracy improvement；当前可以成立的是
**约 26–27% token 降低下的精度/成本 Pareto 工作点**。LME temporal 的负向趋势要求
在新 holdout 上继续验证 temporal budget 和 state/update reranking。

还观察到 judge noise：个别明显包含错误数值的 baseline prediction 被判为正确。
正式报告应对所有 0↔1 discordant cases 做第二 judge 或人工复核，并增加数字一致性、
normalized exact/F1 等确定性 sanity check。

## 6. 工程状态与下一步

- 所有 precision-aware、QueryIR soft fallback、candidate cap 均为 opt-in，未改变冻结默认。
- source-time normalization 也改为独立 opt-in prompt/cache contract；它不能与旧 prompt
  结果混写，需单独完成时间题消融后再决定是否启用。
- 下一步优先优化 Top-8/16/32 排序，而不是继续扩大 full reservoir：typed edge 真正参与、
  owner/predicate/time/state 特征校准、按 operator 训练/校准 stopping confidence。
- 构造一小批 exhaustive relevance pool（而不只是 sufficient gold），才能得到更接近
  绝对值的 precision；现阶段继续使用 annotation precision 做 paired lower-bound。
- 任何新配置必须同时产出 retrieval Pareto、token/latency 和 answer conversion 三组结果。

## 7. 可复现产物

- `../artifacts/report/v5_12/precision_gate_dev200_temporal24_no_mmr/summary.json`
- `../artifacts/report/v5_12/precision_gate_dev200_temporal24_no_mmr/per_question.jsonl`
- `../artifacts/report/v5_12/precision_gate_dev200_temporal24_no_mmr/pareto.md`
- `../artifacts/report/v5_12/precision_gate_dev200_temporal24_no_mmr/recall_precision_pareto.pdf`
- `../artifacts/report/v5_12/answer_dev200/paired_temporal24/summary.json`
- `../artifacts/report/v5_12/answer_dev200/paired_temporal24/transitions.jsonl`
- `../artifacts/report/v5_12/precision_answer_conversion_v510/summary.md`

