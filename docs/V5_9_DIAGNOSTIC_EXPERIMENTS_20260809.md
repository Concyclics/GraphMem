# GraphMem V5.9 全链路诊断实验与方法论

日期：2026-08-09

## 1. 目标与原则

本轮实验不把“最终错误”统一归因于检索，而是沿以下链路分解：

```text
Raw turn → Scene/CanonicalFact → Hierarchical relation index
         → QueryIR/seed/graph traversal → Evidence pack
         → Algebra/candidate → Answer model → Judge
```

实验遵守四项原则：

1. **固定干预**：每个 arm 只改变一个可描述的变量。
2. **逐题配对**：binary metric 使用 exact McNemar test；连续 recall 差使用 paired bootstrap。
3. **设置空操作对照**：用未改变实际 prompt token 的 `span128` 估计 Qwen-FP8 运行间噪声。
4. **严格区分结论层级**：确定性日志结果、开发集确认性结果和待 holdout 验证目标分别报告。

所有机器可读结果位于：

- `artifacts/report/v5_9/error_chain/error_chain.json`
- `artifacts/report/v5_9/path_retention/path_retention.json`
- `artifacts/report/v5_9/extraction_rescue/extraction_rescue.json`
- `artifacts/report/v5_9/diagnostic_summary/summary.json`

复现实验入口：

- `scripts/audit_v5_9_error_chain.py`
- `scripts/measure_v5_9_path_retention.py`
- `scripts/measure_v5_9_extraction_rescue.py`
- `scripts/summarize_v5_9_diagnostic_experiments.py`

## 2. E0：全量错误漏斗

### 2.1 方法

在冻结 V5.9 SQLite 图上，将 2,040 条 retrieval row、answer、judge 和 gold turn provenance 逐题连接。该实验不调用模型，重复运行应产生完全相同的计数。

核心边界指标定义为：

\[
R_{fact}=\frac{\#\text{gold turns with CanonicalFact}}
                 {\#\text{gold turns}},
\]

\[
R_{pack}=\frac{\#\text{questions whose all gold turns are packed}}
                 {\#\text{annotated questions}}.
\]

### 2.2 结果

| 数据集 | Accuracy | Gold Fact recall | 问题 Fact 全覆盖 | Candidate all-hit | Packed all-hit |
|---|---:|---:|---:|---:|---:|
| LongMemEval | 72.60% | 78.80% | 67.00% | 100.00% | 56.00% |
| LoCoMo | 81.95% | 82.88% | 77.23% | 99.87% | 58.06% |

错误分解：

- LME 100 个带 turn gold 的问题中有 50 个错误，其中 35 个缺少 packed gold，15 个在 gold 全部进入 prompt 后仍错误。
- LoCoMo 1,533 个有 evidence 标注的问题中有 277 个错误，其中 233 个缺少 packed gold，44 个在 gold 全部进入 prompt 后仍错误。
- LoCoMo 中 Fact 全覆盖问题准确率为 84.46%，未全覆盖为 73.35%，差 11.11pp；bootstrap 95% CI 为 `[6.10pp, 16.18pp]`。
- LME 的 100 题 Fact 覆盖条件差不显著且方向相反，不能从小样本推断因果。

图结构同时显示：

- 62,811 条 `coarse_related` 中仅 117 条可识别为跨 session，比例 0.186%，只覆盖 33/510 个 memory。
- 62,808 条关系位于 level 0，只有 3 条位于更高层。
- 目标图 embedding 数为 0，因此冻结全量结果并未测试完整 HNSW/dense 路径。
- 216,070 条 CanonicalFact 的 confidence 全部为 0.5，置信度没有校准能力。
- 135,372 个 collection manifest 中 82.57% 只有一个 member，不能自动当作可靠集合闭包。

### 2.3 结论

候选池不是主要瓶颈。当前主要损失顺序为：

1. Raw→Fact 信息缺失；
2. 图上跨会话关系不足；
3. candidate→pack 丢失；
4. Temporal/Count/Multi-hop 答案执行错误。

## 3. E1：修正 C1 多跳路径指标

### 3.1 方法

原 C1 将 gold-edge recall 直接记录为 `multi_hop_path_retention`。本轮在完全相同的合成 workload 和候选构建器上分别测量：

- 直接 gold-edge recall；
- gold pair 在两跳以内是否可达；
- gold semantic component 是否在内部边上保持连通。

### 3.2 N=20K 结果

| Method | Candidate edges | Gold edge recall | ≤2-hop reachability | Internal component connected |
|---|---:|---:|---:|---:|
| ANN-only | 83,810 | 100.00% | 100.00% | 100.00% |
| Flat sparse | 124,375 | 75.00% | 100.00% | 75.00% |
| CIR | 108,820 | 86.38% | 86.91% | 80.96% |

原有复杂度结论仍成立：CIR candidate exponent 为 0.984，接近线性；但质量结论需要调整。当前合成 workload 上 ANN-only 同时拥有更低 wall time、更低 relation-token envelope 和更高路径质量；flat sparse 即使丢失直接边，也能通过两跳路径恢复全部 gold pair。

因此该 workload 只能验证“减少候选比较”，不能证明 CIR 对 ANN-only 的关系表达优势。下一版必须加入：

- typed relation precision/recall；
- 跨 session bridge recall；
- path query 和 Multi-hop QA；
- 与 flat ANN 在相同 embedding、相同 token 和相同 evidence budget 下的端到端对照。

## 4. E2：Raw→Fact 查询无关补抽

### 4.1 方法

从 Fact 未全覆盖的问题中，按五个直接依赖记忆的 stratum 最多各抽 20 题。LoCoMo Cat3 因需要 open-domain knowledge，不纳入 evidence-sufficiency 审计。

抽取器只看到单个 source turn、speaker 和 timestamp，不看到 benchmark question 或 reference answer。它被要求保留每一个独立命题、人物/所有权、数字、单位、时间、否定、状态和 modality。随后同一个本地 Qwen-30B 对三种 evidence condition 判断能否推出 reference answer：

1. current CanonicalFacts；
2. current facts + missing-turn re-extraction；
3. annotated raw turns，作为 oracle ceiling。

这是确认性实验而非最终 benchmark：extractor 与 judge 使用同一 backbone，需异构 judge 或人工复核。

### 4.2 结果

- 93 个问题；
- 109 个缺失 gold turn；
- 新生成 557 条原子 fact；
- 一个长 turn 即使 2,048 output-token 重试仍未形成完整 JSON；
- 395 次调用，共 114,598 prompt token、69,733 completion token。

| Stratum | Current sufficient | Augmented sufficient | Raw oracle | Rescue / rescuable |
|---|---:|---:|---:|---:|
| LME Multi-session | 18.75% | 62.50% | 81.25% | 7/10 |
| LME Temporal | 5.88% | 70.59% | 88.24% | 10/14 |
| LoCoMo Cat1 | 55.00% | 70.00% | 90.00% | 4/7 |
| LoCoMo Cat2 | 10.00% | 65.00% | 90.00% | 11/16 |
| LoCoMo Cat4 | 0.00% | 95.00% | 95.00% | 19/19 |
| Overall | 18.28% | 73.12% | 89.25% | 51/66 |

在 raw 足够而 current facts 不足的 66 题中，查询无关补抽救回 51 题，条件救回率 77.27%。这证明 extraction completeness 是高优先级杠杆。

## 5. E3：真实 dense-off/on 固定索引检索

### 5.1 方法

复制冻结图，仅在副本中为困难集涉及的 110 个 memory 建立 source-turn embedding。共索引 55,323 个向量，消耗 10.63M embedding input token、921 个 batch。使用同一 C23 runner、H10、adaptive hierarchy、32-turn/5K-token budget，只切换 dense search。

### 5.2 结果

| Metric | Sparse | Dense | Delta |
|---|---:|---:|---:|
| All-hit | 43.0% | 47.0% | +4.0pp |
| Mean turn recall | 56.63% | 62.33% | +5.70pp |
| Gold-session route recall | 79.86% | 79.86% | 0 |
| Retrieval p95 | 2,037.7 ms | 2,440.3 ms | +19.8% |
| False-complete / certified | 56.82% | 52.27% | -4.55pp |

All-hit 的 paired McNemar new-only/old-only 为 10/2，`p=0.0386`；turn-recall delta 的 paired bootstrap 95% CI 为 `[3.02pp, 8.72pp]`。

结论：dense channel 确实能补 seed，但不能改善物理 hierarchy route；当前 SQLite/vector 实现增加约 20% p95，且 certificate safety 仍不可接受。生产路径需要真正的 HNSW/ANN index，而不是每次读取大量向量后做 Python 侧融合。

## 6. E4：答案与打包干预

### 6.1 设计

困难开发集包含 LME Multi-session/Temporal 各 50 题、LoCoMo Cat1/Cat2 各 50 题。答案和 judge 均使用冻结全量实验相同的 Qwen-30B、temperature=0、pinned prompt。

三个 arm：

- `candidate_off`：只关闭 algebra candidate 注入；
- `span128`：设置 span window，但本次实际 prompt/evidence token 逐题未变化，因此作为空操作噪声对照；
- `turn64`：将 max evidence turns 从 32 提高到 64。

### 6.2 结果

| Arm | Dataset | Baseline | Arm | Delta | New/old only | McNemar p |
|---|---|---:|---:|---:|---:|---:|
| candidate-off | LME | 50% | 46% | -4pp | 3/7 | 0.3438 |
| candidate-off | LoCoMo | 54% | 63% | +9pp | 11/2 | 0.0225 |
| span128 empty control | LME | 50% | 49% | -1pp | 4/5 | 1.0000 |
| span128 empty control | LoCoMo | 54% | 60% | +6pp | 8/2 | 0.1094 |
| turn64 | LME | 50% | 49% | -1pp | 2/3 | 1.0000 |
| turn64 | LoCoMo | 54% | 65% | +11pp | 16/5 | 0.0266 |

空操作 arm 没有改变任何一题的 prompt token，但 LME/LoCoMo 分别只有 31%/48% 的 prediction 与冻结基线字面相同，并出现 LoCoMo +6pp 波动。这说明 FP8、并发和服务调度下的 temperature=0 仍不足以保证运行确定性。

因此：

- `candidate_off` 的 +9pp 不能解释为纯 candidate 因果收益；空操作已经产生 +6pp。
- 直接将 candidate-off 与本轮 candidate-on 空操作对照相比，LME 为 -3pp（`p=0.5078`），LoCoMo 为 +3pp（`p=0.4531`），均不显著；当前没有证据证明 candidate 注入的平均净效应。
- LoCoMo Temporal 中 candidate-off +16pp，但空操作也达到 +14pp，进一步说明差异主要受运行噪声影响。
- turn64 的确定性 retrieval 变化更可信：LoCoMo all-hit 32%→37%（+5pp，`p=0.0625`），mean evidence token 增加约 1,389；答案 +11pp 中只有一部分可归因于证据变化。
- LME turn64 的 all-hit 完全不变，因为瓶颈是 5K token cap 和排序，不是 32-turn cap。

下一次答案消融必须至少三次独立重复，或对两个 arm 使用完全相同的 frozen rendered prompt/candidate patch，再用异构 judge；报告主结果应在 untouched holdout 上确认。

## 7. E5：现有系统微基准的有效范围

已测 synthetic workload：单 memory、256 sessions、每 session 4 个模板化 turn；不调用 semantic extractor、entity merge 或 relation refine。

### 可证明结果

- 8 个 persistent process worker 在 concurrency=32 时达到 160.55 QPS、p95 245.3 ms；单进程 hierarchy 为 17.09 QPS、p95 9.20 s。
- concurrency=64 时 process plane 为 162.29 QPS、p95 426.6 ms，吞吐已经饱和。
- affected-path authority commit p95 约 4.52 ms、触及 4 行；full snapshot p95 527.35 ms、触及 11,114 行。
- 发布期间 64,793 次 reader operation 无错误，old immutable view 未被修改。

### 不可据此声称

- 新 session/新 memory 的完整写入延迟；
- semantic extraction、embedding delta、cross-session relation maintenance；
- partition split/merge 或 hierarchy rebalance；
- worker crash、SQLite authority crash、磁盘错误或跨节点 failover；
- 总 worker RSS；当前 31.6 MiB 只统计 snapshot cache 估算；
- 包含 answer LLM 的端到端多用户 QPS。

## 8. 证据优先级与当前结论

### 已被直接支持

1. Raw→Fact 覆盖是实质瓶颈，补抽具有很高可恢复空间。
2. Dense seed 在困难集上显著提升检索 recall，但当前实现增加约 20% p95。
3. Candidate pool 基本包含 gold，pack 是主要信息损失点。
4. 当前 CIR 跨 session 边极少，现有合成 C1 不能证明优于 ANN-only。
5. certificate 不能安全驱动 bypass；完整实验中所谓 closed-form 仍调用 answer LLM。
6. 进程分片对 Python CPU-bound retrieval 有效，但不等于高可用。

### 仍需确认

1. 查询无关补抽对独立 judge 和最终 QA accuracy 的净提升。
2. obligation-aware span pack 能否以相同 token 获得高于 turn64 的 all-hit。
3. typed cross-session relation 是否能在 ANN-only 之上提高真实 Multi-hop QA。
4. unified QueryIR + deterministic temporal executor 的答案收益和安全 bypass 率。
5. 新写入、持续混合负载和故障场景的 SLO。

## 9. 最小复现命令

在 `GraphMem` 根目录、报告环境解释器可用的条件下：

```bash
../.conda-envs/graphmem-v58/bin/python scripts/audit_v5_9_error_chain.py
../.conda-envs/graphmem-v58/bin/python scripts/measure_v5_9_path_retention.py

LOCAL_API_KEY=local PYTHONPATH=src \
  ../.conda-envs/graphmem-v58/bin/python \
  scripts/measure_v5_9_extraction_rescue.py --workers 32 --per-stratum 20

../.conda-envs/graphmem-v58/bin/python \
  scripts/summarize_v5_9_diagnostic_experiments.py
```

答案消融的完整 CLI、输入路径、config hash、tokenizer hash 和输出 manifest 均保存在 `../artifacts/v5_9/diagnostic_ablations/`。Dense 实验必须使用冻结图副本并显式开启 `--embedding --allow-subset`，禁止向权威图写 embedding。
