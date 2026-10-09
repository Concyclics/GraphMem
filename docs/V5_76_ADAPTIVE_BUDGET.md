# V5.76 QueryIR/闭包驱动的自适应预算

## 1. 目标与结论

固定 32-turn 虽然节省 Token，但容易漏掉多跳、集合和时间问题的必要
witness；固定 64-turn 则让简单 lookup 承担不必要的上下文开销。V5.76
将其改写为一个**回答前、单次调用、可审计**的预算控制问题：

1. 用 32-turn / 2,200 evidence Token 生成精度优先的基础证据包；
2. 根据 QueryIR、逻辑闭包和打包后的 Evidence Certificate 判断证据是否完整；
3. 只对存在可观测缺口的请求重排候选，并扩展到 64 或 80 turn；
4. 只有基础证据包实际触碰 Token 上限时，才将 evidence Token 提高到
   3,800 或 4,500；
5. 在最终证据包确定后仅调用一次回答模型。

控制器不读取 gold turn、reference answer、候选答案或 judge verdict。所有升级
理由、预算和证据变化均写入逐题 trace，因此线上路由与离线评测严格分离。

## 2. 双轴三级预算

| 层级 | evidence turns | evidence Token | 适用请求 |
|---|---:|---:|---|
| B0 | 32 | 2,200 | 证据闭包完整的普通请求 |
| B1 | 64 | 3,800 | QueryIR 或 witness 显示普通缺口 |
| B2 | 80 | 4,500 | 多个缺口叠加的高严重度请求 |

turn 和 Token 是两个正交控制量。扩大 turn 意味着从已排序候选池中寻找更完整的
witness，并不自动扩大最终上下文；只有打包器确认基础 Token cap 已耗尽且仍存在
缺口时，才增加 Token 配额。这样既允许“用更多候选替换噪声”，也避免把所有风险
请求直接变成长 prompt。

## 3. 回答前闭环

### 3.1 QueryIR 与缺口度量

QueryIR 提供算子、operand、时间约束、集合/聚合要求、编译置信度以及解析告警。
检索完成后生成两个证书：粗排后的 closure certificate 和最终打包后的 packed
certificate。严重度由线上可观测信号累加：

\[
s(q)=2I_{\mathrm{closure\ incomplete}}
    +I_{\mathrm{packed\ incomplete}}
    +I_{\mathrm{QueryIR\ uncertain}}
    +I_{|\mathrm{operands}|>1}
    +I_{\mathrm{critical\ witness\ missing}}.
\]

critical witness 包括时间端点、顺序、状态历史、集合成员、实体绑定和来源证明。
当前 Pareto gate 为

\[
G(q)=I_{\mathrm{packed\ incomplete}}\land\left(
 I_{\mathrm{complex\ closure}\land s\ge3}
 \lor I_{\mathrm{lookup}\land s\ge2}
 \lor I_{\mathrm{QueryIR\ soft\ fallback}}\right).
\]

其中第一项覆盖多跳、时间、聚合、集合和推理请求；第二项修复原方案中“lookup
被一律视为简单题”造成的漏召回；第三项在最终证书仍不完整且 QueryIR 降级为
soft fallback 时主动补充候选，而不是把编译不确定性留给回答模型猜测。

### 3.2 预算决策

turn 预算由

\[
T(q)=
\begin{cases}
32,&G(q)=0,\\
80,&G(q)=1\land s(q)\ge6,\\
64,&\text{otherwise}
\end{cases}
\]

给出，并受全局 hard limit 和实际候选池大小约束。Token 预算为

\[
B(q)=
\begin{cases}
2200,&\neg H_{\mathrm{cap}},\\
4500,&H_{\mathrm{cap}}\land s(q)\ge6,\\
3800,&H_{\mathrm{cap}}\land s(q)\ge2,
\end{cases}
\]

其中 \(H_{\mathrm{cap}}\) 表示基础包确实触碰 Token 上限且证书仍不完整。若只有
QueryIR 不确定性但无需增加 turn，控制器允许在 B0 内重排：保护前 20 条强证据，
用定向 witness 替换弱尾部。

### 3.3 重新检索、打包与单次回答

触发升级后，系统复用同一次图导航得到的候选 reservoir，并按目标档位重新编译
dual-lane 顺序，再执行 obligation-aware packing；这不是把初始 32 条证据和额外
证据机械拼接。QueryIR/证书只决定“是否升级”，不会用未经验证的全局 rerank 改写
目标档位的排序契约。打包器按 operand 覆盖、时间端点、关系路径和来源完整性重新
竞争预算，从而同时提高 recall 并抑制噪声。最终 prompt 确定后才调用回答模型，
因此每题仍只有一次完整回答请求。

设三个预算层级的请求成本为 \(C_0,C_1,C_2\)，路由概率为
\(p_0,p_1,p_2\)，则

\[
\mathbb E[C]=p_0C_0+p_1C_1+p_2C_2,
\qquad p_0+p_1+p_2=1.
\]

相较固定 64-turn，节省来自 \(p_0(C_{64}-C_0)\)；相较固定 32-turn，准确率
增益来自风险请求上的定向 witness 补全。两者由同一个证书驱动，而不是两个互不
一致的启发式开关。

## 4. 回答候选与部署档位

默认 `pareto` 档位对最终证据包调用一次回答模型，并直接输出结果。若产品允许
额外生成成本，可以在完全相同的证据包上使用多候选采样；这只改变回答侧采样数，
不改变检索 gate。

`candidate oracle` 仅衡量“多个随机候选中是否存在正确答案”，用于分离检索上界与
模型选择误差，不是可部署 selector 的成绩。任何正式线上数字都必须同时报告单候选
或 label-free selector 的结果。

## 5. 实验诊断与设计取舍

全量检索审计表明，回答前 gate 将 1,540 题中的 480 题保留在 B0，1,012 题
路由到 B1，48 题路由到 B2；平均打包 54.52 个 turn。带 gold 标注的 1,533 题
结果如下：

| 预算策略 | Any-hit | All-hit | Recall | Precision |
|---|---:|---:|---:|---:|
| 固定 32-turn | 86.76% | 73.26% | 80.17% | 2.89% |
| V5.76 自适应 | **89.89%** | **77.56%** | **83.97%** | **2.32%** |
| 固定 64-turn | 90.80% | 78.67% | 85.11% | 1.85% |

自适应控制器分别弥合固定 32 与 64-turn 间 77.5% 的 any-hit 缺口和 79.5%
的 all-hit 缺口，同时保留高于固定 64-turn 的 precision。其平均 packing prompt
为 5,426 Token，相较固定 64-turn 的 6,176 Token 减少 12.14%；对留在 B0 的
480 个简单问题，平均从 6,254 降至 3,728，减少 40.40%。

回答阶段采用相同 V5.73 prompt policy、seed、采样参数和 Luna-medium judge 的
三臂配对运行，避免将采样波动误作检索增益：

| 预算策略 | Candidate-1 | Best-of-8 oracle | API prompt mean | API total mean |
|---|---:|---:|---:|---:|
| 固定 32-turn | 83.57% | 85.97% | 4,446 | 4,655 |
| V5.76 自适应 | **84.55%** | **86.69%** | **5,439** | **5,657** |
| 固定 64-turn | 85.32% | 87.27% | 6,189 | 6,407 |

因此自适应单候选相对固定 32-turn 提高 0.97pp，并弥合 55.6% 的 32→64
准确率差距；相对固定 64-turn，平均 prompt 和总 Token 分别减少 12.11% 和
11.70%。B0 简单题的真实 API 总 Token 从 6,475 降至 3,950，减少 39.01%。
best-of-8 只度量候选可用性，其 oracle 相对固定 32-turn 提高 0.71pp。

两条探索路径未进入默认方案：

- 二次“增量续答”只发送新增证据时，模型容易把未重发的基础证据误判为不存在；
  即使加入基础答案和 evidence anchors，扩展子集仍显著低于一次性完整证据回答。
- 将基础证据与全部扩展证据直接合并虽然能提高部分难题上界，但会把平均输入推高
  到约 8.25K Token，违背自适应预算的系统目标。

两阶段保留 compact/expanded 两种答案的候选集合，离线 candidate oracle 可达到
90.13%；但现有 label-free selector 只有 85.97%，且重复发送上下文会增加成本。
因此 90.13% 只能作为“正确候选存在”的诊断上界，不能作为线上准确率。若要求可
部署的 90% 以上结果，下一瓶颈是回答候选选择或更强回答模型，而不是继续无约束扩大
turn：固定 64-turn 的同模型 best-of-8 上界也只有 87.27%。

因此最终方法选择“回答前证书门控 + 重新打包 + 单次回答”，而不是依赖回答后的
不确定性猜测，也不重发完整上下文。

## 6. 审计与验收

每题记录以下字段：

- QueryIR route、编译置信度、soft-fallback 和解析告警；
- closure/packed certificate、缺失 obligation、严重度和升级理由；
- base/target turn 与 Token、是否命中 Token cap、最终 evidence IDs；
- prompt hash、真实 API usage、retry、截断与候选数量。

验收要求包括：控制器输入不含 label；32/64/80 三个层级均受 hard limit 约束；
简单题相对固定 64-turn 有显著 Token 节省；风险题的 witness recall/all-hit 提升；
API usage 总和可从逐题账本复算；服务重启后可基于 checkpoint 断点续跑。

核心实现位于 `src/graphmem/retrieval/adaptive_recall.py` 和
`src/graphmem/retrieval/navigator.py`，运行配置为
`configs/v5/runtime_v5_76_adaptive_budget.json`。回答候选控制和探索性续答实现
保留在 `src/graphmem/answer/`，用于复现实验及诊断，不属于默认在线路径。
配对全量结果及输入哈希保存在
`../artifacts/report/v5_76_adaptive_budget/preanswer_target_lane_v573/paired_summary.json`。
