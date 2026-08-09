# V5.15 单边多属性与稀有词粗边实验

## 1. 假设与设计边界

本轮验证两个假设：

1. 同一节点对只材料化一条 `coarse_related`，并在 edge source 的 relation mask 中同时
   携带 `scene/entity/state/temporal/collection/lexical` 属性，避免重边重复占用 degree、
   frontier 和内存；
2. LongMemEval gold session pair 高频共享稀有词，因此可以把稀有词用于长区域之间的
   粗连边，而不是用于短 query 到 session 的直接排序。

第二个边界来自已有反例：稀有词在 session→session 上具有 14--21x gold/random
分离度，但作为 query→session router 时 top-3 all-hit 只有 54.5%，低于 BM25 的 70%。
因此本轮保留原 lexical/dense seed，只增加 `lexical_rare` 关系属性。

## 2. 实现

### 2.1 构建规则

对每个 memory，以 session 为文档统计原文词项 DF。连接词集合为

\[
R=\{w\mid 2\leq \mathrm{DF}_{session}(w)
\leq \max(2,0.05|S|)\}.
\]

DF=1 的词不能形成跨 session 边，直接删除；高 DF 词和对话模板 stopword 也删除。
两个不相交区域共享至少三个词时产生 `lexical_rare` signal：

\[
s_{lex}(u,v)=\min\left(1,
\frac{|R(u)\cap R(v)|}{2K}\right),\qquad K=3.
\]

候选由倒排表生成，不枚举全量区域对。每端 lexical view quota 为 6，最终
`coarse_related` degree cap 为 16。属性沿粗图自底向上聚合，并在父 gate 下钻时与
子节点属性取交集。

### 2.2 单边多属性

候选可能同时由 semantic HNSW、entity posting、state key 和 rare lexical posting
提出，但最终按 `(src,dst,coarse_related)` 去重，只生成一个 edge ID。所有越过在线
阈值的属性排序后写入同一个 source，例如：

```text
relation_mask:lexical_rare,scene_similar,shared_entity
```

查询侧对属性 bonus 做加和，所以多属性边可以优先于只有 scene similarity 的边；mask
只影响 routing priority，不能直接完成 proof obligation。

### 2.3 查询约束

`lexical_rare` 只在 QueryIR 需要多 operand、state history、temporal endpoint、ordering
或 collection 时获得 0.75 的小幅 bridge bonus。普通 lookup 不加 bonus，避免把已被
实验证伪的 rare query router 重新引入。跨区命中后允许一次结构下钻。

## 3. 200 题构建结果

| 指标 | V5.14 relation mask | + lexical_rare | 差值 |
|---|---:|---:|---:|
| 总边数 | 333,889 | 354,410 | +20,521（+6.15%） |
| direct session-pair recall | 56.52% | 81.04% | +24.51pp |
| 2-hop session-pair recall | 67.78% | 96.12% | +28.34pp |
| session-pair question all-hit | 63.0% | 93.0% | +30.0pp |
| direct fact-pair recall | 33.78% | 33.78% | 0 |

新增/组合的 lexical edge 共 20,864 条：

- 纯 `lexical_rare`：20,494；
- lexical + scene：307；
- lexical + entity：55；
- lexical + scene + entity：7；
- lexical + entity + state：1。

端点分布为 RoutingCard→RoutingCard 13,043、Scene→Scene 7,808、跨层 13。最大端点
degree 为 13、平均 3.3，未越过 16 的硬上限。完整构建比较量从约 10.84M 增至
11.00M（+1.54%），没有 LLM 调用或 token 开支。

结果说明 rare lexical 是高召回、近线性的粗图连接规则；但绝大多数边只有 lexical
属性，真正的多属性交集只有 370 条。

## 4. 固定预算检索结果

相对 V5.14，在相同 QueryIR、seed、beam=2、hop decay=0.3 和 32/48-turn pack 下：

| 指标 | 32 turns | 48 turns |
|---|---:|---:|
| all-hit 差值 | 0 | 0 |
| recall 差值 | 0 | 0 |
| precision 差值 | 0 | 0 |
| gold hits/题差值 | 0 | 0 |
| graph-only turns/题 | +0.61 | +0.61 |
| graph-only gold hits/题 | +0.005 | +0.005 |
| visited nodes/edges | -1.215 | -1.215 |

粗关系 walk 从 435 增到 1,275，`scene_contains` 从 4 增到 257；只有一道 LME
multi-session 题的 graph-only gold 从 2 增到 3，而该 gold 已被 raw lexical/dense
通道召回并进入原 evidence pack。因此 session path 的巨大提升没有转化为新的 packed
gold。

进一步把 typed-region hydration 放宽到三层时，lexical 图的 `scene_contains` 增到
543，仍只有同一个冗余 gold；32/48-turn accuracy 不变，并使旧 entity/state arm 的
一个 32-turn 命中消失。该修改已回退。

## 5. 结论与后续条件

“单边携带多个属性”是正确实现，应保留。`lexical_rare` 也通过了构建 coverage、复杂度
和 degree gate，但没有通过最终 packed-accuracy gate，因此保留为独立、默认关闭的
`rare_lexical_relation` 实验开关。

瓶颈不是缺少 Session/Scene 路径，而是 coarse edge 只保存了 `lexical_rare` 这个布尔
标签，查询端不知道两端具体共享了哪些词，也无法用这些词选择 region 内的正确 child/
Fact。下一步应先实现：

1. 在有界 sidecar 中保存每条边的 top shared rare anchors 与 IDF，而不是把词面塞进
   edge source；
2. query seed 命中一端后，以 `query/seed anchors ∩ edge anchors` 对边做条件化选择；
3. 下钻时使用同一 anchor 的 child posting，执行
   `anchor region -> lexical bridge -> matching child -> source turn`；
4. 以 `new packed gold / added edge` 和 precision 为 gate，再比较 K=3/5、quota=4/6。

## 6. 实物

- 图：`../artifacts/report/v5_15/relation_mask_lexical_dev200/report_graph.sqlite`
- 构建审计：`../artifacts/report/v5_15/relation_mask_lexical_dev200/build_audit/summary.json`
- 一层 hydration 检索：`../artifacts/report/v5_15/accuracy_budget_relation_mask_lexical_dev200/summary.json`
- 配对结果：`../artifacts/report/v5_15/lexical_rare_paired_vs_v5_14.json`
- 三层 hydration 负实验：
  `../artifacts/report/v5_15/accuracy_budget_relation_mask_lexical_hydration3_dev200/summary.json`
