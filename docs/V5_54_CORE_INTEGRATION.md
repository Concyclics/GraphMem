# V5.54 核心代码固化说明

V5.54 不再依赖 V5.43--V5.54 的离线 Prompt materializer 才能得到最终请求。
已验证的 winner policy 现在由 `AnswerStage.prepare()` 在证据打包之后直接执行。

## 固化范围

- `src/graphmem/answer/readout_policy.py`
  - V5.45：匿名双角色 typed readout；具名多方 what/when/who/which
    strongest-last block layout；
  - V5.49：受限聚合 execution card、具名 date-difference 单行停止、
    modal 且非 counterfactual 的 grounded inference；
  - V5.52：具名 temporal query-overlap block layout；
  - V5.53：带宽泛年份范围、但非 `on <date/year>` 精确日期形式的
    what/which block layout；
  - V5.54：匿名 temporal query-overlap block layout。
- `AnswerConfig.v5_54()` 固定与最终实验一致的 answer contract：
  source-time normalization、topological evidence、32-row aggregation
  availability、source reserve、preference synthesis、contextual question date、
  compact topology、2K completion ceiling、无 Candidate Answer 注入。
- `configs/v5/runtime_v5_54_accuracy64.json` 固定查询面：H11、64 turns、
  12K evidence、native seed fusion、named relational view、speaker/Query witness、
  QueryIR soft fallback、dense/FAISS，并关闭已被否决的 coarse lexical-only
  graph traversal、generic exact priority、relation consensus 和 dialogue closure。
- `scripts/run_v5_6_answer.py` 支持 `--runtime-config` 和
  `--answer-policy v5_54`，manifest 记录实际生效而不是 CLI 默认值。

## 不变量

V5.54 readout policy 保证：

1. 不读取 benchmark、gold、prediction 或 judge verdict；
2. 不改变 evidence turn 集合；
3. 只允许整块重排，不改变块内顺序；
4. Prompt Token 不高于进入 policy 前的请求；
5. 每条最终请求记录 route、Token delta 和 evidence-set audit；
6. lexical IDF 求和和 tie-break 均确定化，不再受 `PYTHONHASHSEED` 影响。

## 复现审计

- 以 V5.40 的 2,040 条冻结 PreparedAnswer 为输入，核心 policy 与 V5.54
  最终 artifact 有 2,037 条完整匹配：messages、evidence IDs/order、Prompt
  Token、Prompt hash 和 payload hash 全部一致。
- 其余 3 条来自旧 materializer 对 Python `set` 迭代顺序的依赖：相同
  coverage/IDF 的 block 会随进程 hash seed 交换。核心实现使用排序后的
  term 求和和原 block index 稳定破平；证据集合、Token 和 Prompt contract
  不变，但不继续复制不可复现的随机顺序。
- 使用 V5.21 safe-witness authority graph 重新执行 `dfde3500` 的完整
  navigation/packing/core-readout，得到与 V5.54 artifact 完全相同的 evidence
  集合、顺序、Prompt payload hash 和 9,888 packing Token。
- 全仓测试：`477 passed`。

## 正式运行入口

```bash
python scripts/run_v5_6_answer.py \
  --source-db <v5.21-safe-witness-graph.sqlite> \
  --config configs/v5/v5_17_budget230.json \
  --runtime-config configs/v5/runtime_v5_54_accuracy64.json \
  --answer-policy v5_54 \
  --lme <longmemeval.json> --locomo <locomo.json> --gold <gold.jsonl> \
  --output-root <output> --full
```

旧实验继续使用默认 `--answer-policy legacy`，因此已有 artifact/cache identity
不会被 V5.54 的核心固化隐式改写。
