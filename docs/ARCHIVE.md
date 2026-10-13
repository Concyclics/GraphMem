# 代码清理记录

5.81.1 以持久化 Memory SDK / HTTP 服务及其完整 LoCoMo 复现为维护入口。

- 移除 V2–V4、V5 历次消融、付费评测编排、报告绘制、GPU heartbeat 占用、重复模型启动和旧 Neo4j 部署入口。
- 移除未接入产品链路的多候选 ensemble、回答后扩容控制器、增量回答合并器，以及对应实验专属测试。
- 移除旧 HTML / PDF / 实验报告、过期绝对路径、LLMLingua 等不再使用的依赖。
- 当前建图仍依赖递归层次、原子事实、R3 projection、QueryIR 和 witness 模块；这些有效算法保留。版本编号不是判定代码废弃的依据。
- 15 份旧配置移至 `test/fixtures/profiles/`，仅用于回归验证参数兼容性，不作为部署方案。产品配置只有 `config/memory-service.json`，完整默认值随 wheel 发布。

历史完整代码保存在清理前提交 [`5ced959`](https://github.com/Concyclics/GraphMem/tree/5ced95927b938ee734f373205ee074d4b784cf28)。需要历史实验时可创建独立工作树：

```bash
git worktree add --detach ../GraphMem-research 5ced95927b938ee734f373205ee074d4b784cf28
```

本地模型权重、原始数据、数据库、`.env` 和 benchmark 原始结果不提交到 Git；`runs/` 保存本次可审计产物。
