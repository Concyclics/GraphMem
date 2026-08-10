# V5.19 多属性边消融与双模型 Benchmark

## 固定实验入口

```bash
bash scripts/run_v5_19_attribute_ablation.sh
bash scripts/run_v5_19_full_benchmark.sh
```

两个入口均为可恢复运行。默认 Python 为工作区
`.conda-envs/graphmem-v58/bin/python`，产物根目录分别为：

- `../artifacts/report/v5_19/attribute_ablation_dev200`
- `../artifacts/report/v5_19/full_benchmark`

运行前必须已有本地 8001 embedding 与 8002 Qwen3-30B 服务；脚本只做健康检查，
不会创建或接管服务。Luna judge 与 GPT-5.4-mini 从 `.env` 读取 `SGAO_API_KEY`
和 `SGAO_BASE_URL`，密钥不进入 manifest。

## 六臂构建契约

| 臂 | `enabled_relation_signals` |
|---|---|
| Full | scene、entity、state、collection、temporal、lexical |
| No-Scene | 除 scene 外全部 |
| No-Entity-family | scene、temporal、lexical |
| No-Temporal | 除 temporal 外全部 |
| No-Lexical | 除 lexical 外全部 |
| Semantic-only | 仅 scene/semantic |

Full 从原文冷构建 110 个 Memory。其余臂通过
`prepare_v5_19_ablation_arm.py` 复制 source turns、语义抽取 cache 和 vectors，删除所有
graph/evidence/version/outbox/run-ledger 表后重新粗化和建边。`enabled_relation_signals`
同时约束 coarse/multiview/atomic 候选、逐层 mask 交集、物化和 edge provenance；
多个属性写入同一 edge source，不产生属性重边。

每个臂使用 H11 accuracy64：64 turns、12K evidence Token、QueryIR soft fallback、
source-time normalization、per-memory FAISS、obligation-aware packing、无 completion
上限。回答模型为本地 Qwen3-30B，LongMemEval 和 LoCoMo 均由 GPT-5.6-luna judge。

## 全量与 Token 契约

全量构建以 Full 的 110-memory 冷构建为 seed，只增量摄入剩余 400 个 Memory，
但最终构建账本和 fallback 诊断必须覆盖全部 510 个 Memory。门禁为：

- 510/510 graph versions；
- 510 条生成式 Token ledger；
- 0 request retry、0 semantic extraction retry；
- 每个 Memory 的 uncached input + output 不超过 230K；
- fallback 与 budget degradation 显式进入 build report。

随后为 510 个 Memory 预编译 FAISS，只执行一次 navigation/packing，并冻结 2,040
条 `PreparedAnswer`。Qwen3-30B 直接完成这些请求；GPT-5.4-mini 通过
`replay_v5_prepared_answers.py` 重放同一 messages/evidence IDs/order。打包始终使用
冻结 Qwen tokenizer；回答成本使用各自 API usage。构建和回答的 input、output、total
均保留逐样本记录，主表的 mean/p95/p99/max 使用 nearest-rank。

## Fail-closed 验收与报告导入

`audit_v5_19_experiment.py` 独立复算 Token 百分位与总和，并检查：

- 禁用 signal 未进入候选或任何 typed/refined/coarse provenance；
- Full 含六种 signal 且存在单边多属性，Semantic-only 仅含 scene signal；
- 非 Full 语义抽取 100% cache hit，aggregate graph checksum 不等于 Full；
- 六臂回答/judge 各 200 题；全量每模型回答/judge 各 2,040 题；
- 双模型 question ID 与逐题 Prompt hash 完全一致；
- Luna 模型、prompt commit/hash、temperature、seed 与 reasoning 状态固定；
- LoCoMo Category 5 未进入 judge。

只有通过审计的 `ablation_summary.json` 和 `benchmark_manifest.json` 才由
`render_v5_19_report_assets.py` 写入报告。GPT 运行中显示“运行中”，Mem0 或其他缺失值
显示“待补”，绝不把 `null` 渲染为 0。
