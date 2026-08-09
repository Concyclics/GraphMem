# GraphMem V5.10 系统强化方案

日期：2026-08-09

> 实施状态：本计划的 P0/P1 主路径已完成首轮实现、dev Gate、2,040 题全量评测、多租户压测与 fault probe。完整结果、未通过 Gate 和下一轮优先级见 [`V5_10_FINAL_RESULTS_AND_ROADMAP.md`](V5_10_FINAL_RESULTS_AND_ROADMAP.md)。当前全量结果使用冻结 P8 fact projection，不代表 V5.10 atomic extractor 的端到端收益。

## 1. 总体目标

V5.10 不再把优化目标写成单一 accuracy，而是优化以下受约束目标：

\[
\max \; \mathrm{QAAcc}
-\lambda_t C_{token}
-\lambda_l L_{p95}
-\lambda_m M_{RSS}
\]

并满足：

\[
R_{fact}\geq R_{fact}^{min},\quad
R_{path}\geq R_{path}^{min},\quad
R_{pack}\geq R_{pack}^{min},\quad
\mathrm{FCR}_{cert}\leq \epsilon.
\]

其中 `FCR_cert` 是 certified 但 gold evidence 不完整的 false-complete rate。当前 dense/sparse 困难集均超过 50%，所以真正 bypass 必须保持关闭。

建议将三个 Contribution 组织为一个层级系统：

1. **Token-efficient lossless index construction**：自适应原子抽取 + HNSW coarse assignment + typed cross-session relation restoration。
2. **Low-latency compiled graph retrieval**：单一 QueryIR、operator-aware route、native ANN/typed postings 和 obligation-aware packing。
3. **Accurate and scalable execution plane**：确定性 Temporal/Set algebra、证书安全门控、增量写入和多租户进程数据面。

## 2. P0：索引正确性优先

### 2.1 Lossless Atomic Extractor

当前问题：固定每 scene 4 facts、confidence 全为 0.5、同一场景中的数字/人物会相互竞争。

实施内容：

1. 在 scene 抽取前执行信息单元扫描，至少识别 `entity/owner/number/unit/date/duration/negation/modality/state-change/item`。
2. facts cap 改为自适应：

   \[
   K_s=\min(K_{max},\lceil \alpha I_s+\beta E_s+\gamma T_s\rceil),
   \]

   其中 \(I_s\) 为独立命题数、\(E_s\) 为实体数、\(T_s\) 为时间/数字表达数。
3. 每个高显著性信息单元必须产生 fact、明确标记 `unresolved`，或记录“为何不可抽取”。
4. confidence 进入严格 schema，并用人工标注集做 reliability calibration；在完成前删除默认 0.5 的伪语义。
5. 长 turn 不再只保留首尾：按句法/标点切 chunk，完成后合并去重。
6. 永远保留 lossless evidence pointer；低置信或 coverage contract 未通过时，检索自动回退 raw span。

验收 Gate A：

- LME/LoCoMo gold-turn Fact recall ≥95%；
- 数字、日期、否定、实体属性分别 ≥97%；
- 抽取 precision ≥95%；
- 单 memory extraction token 增长不超过 30%；
- missing-fact holdout 的 augmented sufficiency ≥80%，由异构 judge + 20% 人工复核确认。

预期：LoCoMo 全量中 349 个问题存在 Fact 缺失，当前条件准确率差约 11.1pp。仅把该组提升到全覆盖组水平，对总体 accuracy 的经验上界约为 2.5pp；考虑检索连锁收益，保守目标为 +1.5–3.0pp，不能与其他优化直接相加。

### 2.2 真正的 Coarsening + Relation Restoration

当前问题：`coarse_related` 仅 0.186% 跨 session；新图无 embedding；所有新关系为通用类型；428,211 个 ambiguous candidate 未 refine。

实施内容：

1. 使用真实 HNSW 对 Session Card/Scene embedding 做 coarse assignment，禁止用任意 node-id fill 补齐 cluster。
2. 每个 parent pair 若被接受，必须生成可下钻 portal，或继续检查最有 bridge value 的 child pair；不能在高 parent score 时停止关系细化。
3. 为每个 session 设置跨 session neighbour quota，避免同 session 相似边耗尽 degree cap。
4. 将关系恢复拆成两阶段：

   ```text
   ANN candidate / parent gate
        → cheap typed classifier
        → uncertainty × bridge-value scheduler
        → bounded LLM refine
   ```

5. relation type 至少覆盖 `same_entity_state`, `temporal_continuation`, `causal`, `collection_member`, `contradiction/update`, `co_reference`。
6. ambiguous refine 预算按预期 QA value 分配：

   \[
   priority(e)=U(e)\cdot B(e)\cdot Q(e)/C_{token}(e),
   \]

   其中 \(U\) 为不确定性、\(B\) 为跨分区 bridge value、\(Q\) 为历史 query demand。

验收 Gate B：

- relation build candidate exponent ≤1.15；
- 跨 session typed edge precision/recall 均 ≥85%；
- real gold-pair ≤2-hop reachability ≥90%；
- Multi-hop path retention 必须直接测路径，禁止用 edge recall 代替；
- 与 ANN-only 相同 token/latency budget 下，Multi-hop QA 至少 +2pp，否则层级关系模块只能作为可选增强，不能作为核心收益主张。

## 3. P0：检索、打包与执行统一

### 3.1 单一 QueryIR 执行计划

当前 legacy operands 和 AST 并行，binding 通过位置 remap；certificate/packer/algebra 并非共享同一计划。

目标数据流：

```text
Query → AST + compiler confidence
      → principal/owner binding
      → retrieval obligations
      → packed obligations
      → algebra witnesses
      → post-pack certificate
```

实施内容：

- 删除执行路径上的 legacy operand；legacy 只保留 shadow comparison。
- binding 直接引用 AST operand ID，不允许 positional zip。
- 修复第一人称和当前用户 principal mapping；低置信 owner 不得通过 permissive overlap 伪绑定。
- compiler 输出 confidence、parse warning 和 fallback policy；低置信时关闭 closed form。
- 每个 trace 记录 AST、binding、operator obligation、selected witness 和 certificate failure reason。

验收：AST/legacy divergence 有独立集；owner precision/recall ≥98%；任何 certificate complete 都能从最终 pack 重放全部义务。

### 3.2 Obligation-aware Token Packer

当前 LoCoMo 全部触发 32-turn cap，只用约 29% token；turn64 以 evidence token 近翻倍换来 +5pp all-hit，效率不足。LME 则受 5K token cap 限制。

将 packing 写成带约束最大覆盖：

\[
\max_{S}\sum_o w_o\mathbf{1}[o\text{ covered by }S]
-\lambda\sum_{i,j\in S}\mathrm{redundancy}(i,j),
\]

subject to：

\[
\sum_{i\in S} token(i)\le B,
\]

并加入 operand/session/time-stage 的最小覆盖约束。

具体策略：

- 默认使用 cited span + 小窗口；无 span 的 turn 先生成 deterministic salient spans。
- 每个 AST operand 至少保留一个 witness；Multi-hop 每个 gold-like session cluster 设 floor。
- 以“新增 obligation coverage / token”贪心选择，并用 MMR 去重。
- 对数字、日期、否定、状态变化设置不可丢字段。
- turn cap 只作为安全上限，主要预算是 token。

验收 Gate C：

- 相同 5K token 下，LoCoMo difficult all-hit 至少从 32% 提升到 42%；
- LME Multi-session all-hit 从 38% 提升到 ≥48%；
- evidence token 不超过 baseline +10%；
- 对 turn64 达到 all-hit non-inferiority，同时至少节省 35% evidence token。

### 3.3 Deterministic Temporal/Set Executor

当前 normalized interval 已进入图但没有进入答案平面；closed-form candidate 仍调用 LLM，且错误 candidate 会锚定答案。

实施内容：

- normalized fact 以 typed value 进入 algebra，而不是重新从 raw text 解析。
- relative time 永远绑定 source timestamp，不绑定 question date，除非表达明确相对于提问时刻。
- Count 只在 collection scope 被证明且 member 去重后执行；single-member manifest 不能自动代表 closed world。
- executor 输出 value、unit、provenance、interval uncertainty 和 contradiction status。
- candidate draft 在 shadow 中运行；与 LLM 不一致时记录，不注入 prompt。
- 只有白名单算子在独立 holdout 达到 precision ≥99.5%、false-complete ≤0.5% 后才能真正 bypass。

## 4. P1：低延迟数据面

### 4.1 优先优化 Seed Fusion

全量日志中 seed fusion 占 retrieval latency：LME 约 70%，LoCoMo 约 93%；hierarchy route 只有约 1ms。

实施顺序：

1. 将 query normalization/tokenization、predicate postings 和 source-turn FTS plan 预编译到 immutable read view。
2. embedding query 批处理；HNSW index 常驻进程，不从 SQLite 逐行读取向量。
3. score fusion 改为 NumPy/native array 或 Rust/C++ 扩展，避免 Python object 排序。
4. 为同 tenant 的 query 复用 principal、memory metadata 和 hot postings。
5. 记录 snapshot construction，与 traversal latency 分开；当前 `graph_read_view` 名称不能掩盖 cold build。

目标：

- LME retrieval p95 3.60s→≤1.0s；
- LoCoMo p95 1.99s→≤0.75s；
- dense-on 相对 sparse 的 p95 overhead ≤5%；
- hierarchy route p95 维持 ≤5ms。

### 4.2 多租户调度

当前 `ProcessPoolExecutor` 接收无界任务，没有 backpressure、deadline、tenant fairness 或 consistent memory affinity。

新增：

- bounded per-tenant queue；
- global admission controller；
- deadline/cancellation propagation；
- consistent-hash `memory_id→worker`，提高 snapshot affinity；
- per-tenant concurrency/token/RSS quota；
- worker heartbeat、crash detection、restart 和 in-flight retry；
- hot tenant 隔离，防止单用户占满所有 worker。

系统指标必须同时报告 QPS、p50/p95/p99、error/drop/timeout rate 和总进程 RSS，不能只报告 snapshot cache estimate。

## 5. P1：真实增量写入

当前 affected-path 只支持修改已有 partition 的 routing row；新 session 会明确报错。

推荐写入状态机：

```text
RECEIVED
  → RAW_DURABLE            # WAL/authority ACK
  → FACT_INDEXED           # atomic facts + embeddings visible
  → RELATION_INDEXED       # local/cross-session relations visible
  → ROUTE_PUBLISHED        # new immutable read version
```

同步路径仅完成 raw durable；semantic extraction 和 global relation maintenance 异步执行。查询可根据 consistency level 选择：

- `raw_latest`：立即包含新 raw turn；
- `semantic_committed`：只读已完成 fact index 的 version；
- `relation_committed`：要求完整关系版本。

需要实现：

- 新 turn append；
- 新 session local partition insertion；
- HNSW delta index + background merge；
- local typed relation update；
- partition split/merge 与 parent summary 重编译；
- optimistic expected-version retry；
- outbox 驱动异步任务，所有状态幂等可重放。

SLO 建议：raw durable p95 ≤20ms；fact-visible p95 ≤2s；relation-visible p95 ≤5s。LLM extraction 延迟必须单独报告，不能混入 4ms authority delta commit。

## 6. P2：高可用与可观测性

### 6.1 高可用

- SQLite authority 保留单 writer，但通过 append-only change log 和 immutable snapshot 做 follower replication。
- 每个 graph version 带 checksum、source offset 和 schema/config hash。
- query worker 只读本地 snapshot；primary failure 时由 follower 在确认 offset 后提升。
- 建议目标：durable raw RPO=0；semantic index RPO≤1 committed batch；单节点 RTO≤60s。

故障注入必须覆盖：worker `SIGKILL`、writer crash at commit boundary、磁盘 full、corrupt delta、embedding/LLM timeout、网络 partition 和 stale replica。只测成功发布期间 reader 无错误不等于高可用。

### 6.2 Trace 与 checkpoint

每个问题记录：

- graph version/checksum；
- QueryIR AST、compiler confidence、owner binding；
- seed IDs/channel scores；
- reached/candidate/packed turn IDs；
- obligation coverage 与 drop reason；
- algebra witnesses、candidate、certificate phase；
- stage latency、token 和 cache hit。

benchmark runner 每题 append/checkpoint，支持按 question ID resume；单题模型/JSON 失败不得中止整个 shard。本轮 extraction 实验已经复现了这一工程风险。

## 7. 验证路线与发布门槛

### Phase 0：实验有效性（先做）

- 冻结 rendered prompt；同一 arm 重复 3 次；
- 加入异构 judge，并人工复核所有 discordant 样本；
- 报告均值、运行间方差和 paired CI；
- untouched validation/final split，开发集不得继续充当最终测试集。

### Phase 1：Extraction + Pack（最高收益）

- 上线 coverage contract、adaptive facts 和 span packer；
- 做 `raw-only / current-fact / augmented-fact / fact+raw fallback` 消融；
- 预期非叠加净收益目标：LoCoMo +4–7pp，LME hard types +3–6pp。

### Phase 2：Graph + QueryIR

- 真实 HNSW coarse assignment、typed cross-session restore、单一 AST；
- 做 `ANN-only / CIR / CIR+typed / CIR+selective-refine` 对照；
- 必须同时报告 build token、relation precision、path recall、QA、p95 和 RSS。

### Phase 3：Execution + Serving

- temporal/set executor 保持 shadow，达到安全门槛后再 bypass；
- seed fusion native 化、多租户 bounded scheduler、真实 incremental writer；
- 30–60 分钟 Zipf mixed workload + fault injection。

### 最终发布 Gate

- Accuracy：独立 holdout 显著提升，McNemar `p<0.05`，并报告多重比较策略；
- Token：相同或更低 mean/p95 prompt token；
- Retrieval：目标 p95 达标，dense overhead≤5%；
- Correctness：certificate false-complete≤0.5%；
- System：真实多用户 mixed workload 零数据错读，timeout/error 有明确 SLO；
- Availability：RPO/RTO 经故障注入验证，而不是设计声明。
