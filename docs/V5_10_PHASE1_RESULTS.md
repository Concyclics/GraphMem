# GraphMem V5.10 Phase 1 实现与实验记录

日期：2026-08-09

## 1. 实验纪律

- 仅使用固定开发集做实现选择：LongMemEval multi-session 50、LongMemEval temporal 50、LoCoMo multi-hop 50、LoCoMo temporal 50。
- 检索 A/B 固定数据库快照、QueryIR、候选生成、图遍历、5K Token 上限和随机种子，只切换 evidence packing。
- 所有统计保留逐题结果、paired bootstrap 95% CI 和 transition；测试集不用于本轮调参。
- V5.10 路径均为 opt-in，封版 baseline 默认行为不变。

## 2. Phase 1A：Lossless Atomic Extractor

### 2.1 已实现机制

1. 在 LLM 抽取前扫描 `date/duration/number_unit/negation/modality/state_change/entity/item` 信息单元。
2. 以信息单元、实体和时间/数值密度计算自适应 fact cap。
3. fact 记录 `information_unit_ids` 和 exact evidence span；scene 记录 covered、implicit、unresolved、missing 与 raw fallback。
4. coverage 低于阈值时重试；仍失败时保留 mandatory raw evidence group。
5. 长 turn 使用 lossless sentence chunks，合并后按 canonical fact 去重。
6. atomic 模式下取消固定 `confidence=0.5` 的伪置信度。

### 2.2 Missing-fact 109-turn Gate

主 artifact：`artifacts/report/v5_10/atomic_gate_v3/summary.json`。

| 指标 | V5.10 结果 |
|---|---:|
| 信息单元 | 320 |
| covered / missing | 308 / 10 |
| unit coverage | 96.25% |
| facts / turn | 3.78 |
| raw-fallback turns | 8 |
| 固定 0.5 confidence facts | 0 |
| build tokens / ceiling | 197,560 / 390,000 |
| V5.10 sufficiency | 75.27% |
| previous augmented sufficiency | 73.12% |
| current index sufficiency | 18.28% |
| raw oracle sufficiency | 89.25% |

按类型 coverage：date 95.12%、duration 100%、entity 98.13%、item 100%、modality 88.89%、negation 86.67%、number/unit 95.45%、state-change 93.10%。

结论：coverage contract、adaptive cap、exact span 和 raw fallback 已工作，且在 missing-fact 集上比 previous augmented 提升 2.15pp；但总体 sufficiency 75.27% 尚未达到方案中的 80% Gate，negation/modality 也未达到 97% 目标。因此 Phase 1A 是“机制完成、硬 Gate 部分通过”，不能宣称抽取问题已经解决。后续应在 typed contradiction/update 与 QueryIR polarity obligation 中继续修复这两类缺口。

## 3. Phase 1B：Obligation-aware Span Packer

### 3.1 最终算法

最终 packer 使用四段式顺序：

1. 以 frozen full-turn pack 作为 monotonic non-regression floor；
2. 将 exact cited span 或 deterministic salient span materialize，并保留数字、日期、否定和状态句；
3. 用剩余容量补齐 QueryIR obligation/operand 的完整 proof unit；
4. 按原 candidate rank 填充，multi-turn proof unit 能容纳时保持原子性，不能容纳时证书明确 incomplete。

目标可写为：

\[
S_{base}\subseteq S_{span},\qquad
\sum_{t\in S_{span}} C(span_t)\leq B,
\]

其中第一项是结构化 non-regression invariant，第二项是 Token budget。证明义务只作为 coverage floor，不再对全候选池做 cost-normalized 重排。

### 3.2 开发迭代中发现的两个反例

- 初版以 `coverage / token^0.72` 全局贪心，10 题 smoke 将 Token 降低 77%，但 all-hit 40%→10%，因为短而低排名的 turn 被系统性提升；该版本废弃。
- 第二版保持 rank prior，在 200 题中得到 6 个 0→1、1 个 1→0。唯一回退题的第六条 gold 位于 candidate rank 40；full-turn baseline 因跳过昂贵 turn 而命中 rank 40，span 路径填满 rank 1–32 后反而漏掉。由此引入 `S_base ⊆ S_span` 单调性约束。

### 3.3 最终固定 200 题结果

主 artifact：`../artifacts/report/v5_10/packer_gate_sparse_dev200_turn32_monotone/`。

预算：最多 32 turns、最多 5,000 evidence tokens。

| 指标 | Full-turn baseline | V5.10 span pack | 差异 |
|---|---:|---:|---:|
| strict all-hit | 43.0% | 45.5% | +2.5pp，95% CI [+0.5,+5.0] |
| mean gold recall | 56.63% | 59.93% | +3.30pp，95% CI [+1.25,+5.80] |
| mean evidence tokens | 3,289.6 | 2,441.7 | -847.9（-25.8%） |
| p95 evidence tokens | 5,000 | 3,925 | -21.5% |
| mean retrieval latency | 1,280.8ms | 1,296.3ms | +15.4ms，CI 跨 0 |
| false-complete | 18.0% | 14.5% | -3.5pp |
| mean packed turns | 26.87 | 32.00 | +5.13 |

Transition：109 个 0→0、86 个 1→1、5 个 0→1、**0 个 1→0**。

| Stratum | all-hit baseline → V5.10 | recall baseline → V5.10 | Token baseline → V5.10 |
|---|---:|---:|---:|
| LME multi-session | 34% → 38% | 58.37% → 64.37% | 4,985 → 3,273 |
| LME temporal | 74% → 80% | 76.07% → 83.27% | 4,999 → 3,374 |
| LoCoMo multi-hop | 14% → 14% | 32.93% → 32.93% | 1,581 → 1,555 |
| LoCoMo temporal | 50% → 50% | 59.17% → 59.17% | 1,594 → 1,565 |

### 3.4 结论与未通过项

Phase 1B 的核心假设成立：在相同 Token budget 下，span materialization 能无回退地保留 baseline，并为长 turn memory 释放容量，从而同时降低 Token 和提升 evidence recall。LoCoMo 的 turn 很短且 32-turn cap 已饱和，所以本阶段只能略降 Token，无法改变候选 membership。

原方案 Gate C 中 LME multi-session ≥48% 和 LoCoMo difficult ≥42% 尚未达到；最终分别为 38% 和 14%。这不是继续调 packer 能解决的问题：当前 candidate ceiling 为 100%，但前 32 排序质量不足，尤其 LoCoMo multi-hop recall 只有 32.93%。因此 Gate C 的剩余缺口转入 Phase 2 typed relation/HNSW candidate ranking 和 Phase 3 QueryIR obligations，不能把它包装成 packing 收益。

## 4. 下一阶段锁定输入

- Phase 1A extractor 与 Phase 1B packer 保持 opt-in；后续 A/B 不再修改其算法。
- Phase 2 固定使用 monotone span packer，单独比较 `ANN-only / hierarchy / hierarchy+typed restoration`。
- Phase 2 必须重点报告 LoCoMo multi-hop 的 top-32 recall、gold-pair ≤2-hop reachability、relation precision/recall、build complexity、p95 latency 与 RSS。
