# V5.16 关系跳与层级深度解耦的约束式自顶向下检索

## 1. 问题

旧调度器把所有边统一计入 `max_hops`。当一次 `coarse_related` 已经从区域 A
跳到区域 B 后，剩余 hop 很容易被 `refines_to` 消耗，最终停留在 RoutingCard/Scene，
无法到达 CanonicalFact 和 source turn。两跳因此同时承担了两种不同语义：

- **relation hop**：跨实体、场景、时间或粗区域的横向推理；
- **hierarchy depth**：RoutingCard → 子 RoutingCard → Scene → Fact 的纵向物化。

这两个维度不能共用一个计数器。否则增加图层会反而缩短有效关系推理距离。

## 2. 方法：二维有界搜索

对路径 \(p\) 分别定义关系跳数 \(h_r(p)\) 和结构深度 \(h_s(p)\)：

\[
h_r(p)=\sum_{e\in p}\mathbb{1}[e\notin\mathcal E_s],\qquad
h_s(p)=\sum_{e\in p}\mathbb{1}[e\in\mathcal E_s],
\]

其中结构边集合
\(\mathcal E_s=\{\texttt{refines\_to},\texttt{scene\_contains}\}\)。
在线约束变为

\[
h_r(p)\le H_r,\quad |V_{visit}|\le B_V,\quad
|E_{visit}|\le B_E,\quad |F|\le B_F.
\]

结构下探不消耗 \(H_r\)，但仍消耗节点、边和 frontier 三个硬预算，因此不是全量展开。
结构边只沿构建方向读取，禁止通过 inverse edge 重新向上回游。

### 2.1 每层 rerank 与剪枝

关系扩展和结构下探使用独立 beam：

\[
R(v)=\operatorname{TopK}_{B_r}\{s_r(v,u): (v,u)\in
\mathcal E\setminus\mathcal E_s\},
\]

\[
D(v)=\operatorname{TopK}_{B_d}\{s_d(v,c):(v,c)\in\mathcal E_s\}.
\]

默认 \(B_r=2\)、\(B_d=1\)。关系分数由 QueryIR obligation、relation mask、
query/node lexical overlap、collection key 和 Fact bonus 组成。结构分数在此基础上加入
父节点 `child_postings` 命中：

\[
s_d(v,c)=s_{lex}(q,c)+2\sum_{w\in q}
\mathbb{1}[c\in P_v(w)]+s_{fact}(c)+s_{type}(q,c).
\]

每到一层都重新计算 \(D(v)\)，不能在上层一次性选择整个子树。被选中的结构分支插入
frontier 头部，优先连续下探，直到没有结构子节点；随后继续同层/跨区关系扩展。

### 2.2 避免重复下探

seed fusion 已经通过 `route_hierarchy` 完成一次 query→root→leaf 搜索。因此生产默认
只在**非结构关系到达新区域**时开启新的结构 corridor，不对所有 RoutingCard seed
重复下探。这使修复精确对应“关系扩展到区域后没有下探”的缺陷。

安全预算具有最高优先级：极端情况下若全局 node/edge cap 已耗尽，分支会报告
exhaustion，而不是越过服务级资源上限。正常预算内，relation beam 一旦接纳区域，
结构 hop cap 不会再阻止它到达叶层。

## 3. 三版消融与最终选择

固定 200 题、相同 QueryIR、relation beam=2、hop decay=0.3、32/48-turn evidence pack。
下表 precision 是仅基于官方 gold turn 的下界。

| 调度 | 32 all-hit | 32 recall | 32 precision | 48 all-hit | 48 recall | visited nodes | visited edges | coarse walk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧统一两跳 | 56.0% | 69.28% | 4.484% | 61.0% | 72.79% | 66.31 | 33.38 | 435 |
| V1：所有节点无约束下探 | 58.0% | 70.07% | 4.547% | 62.0% | 74.10% | 95.63 | 78.73 | 4 |
| V2：seed/关系到达均下探 | **58.5%** | **70.32%** | **4.594%** | **62.5%** | 73.78% | 90.41 | 64.00 | 529 |
| V3：仅关系到达下探（默认） | 56.5% | 69.24% | 4.484% | 61.5% | 73.21% | 69.90 | 36.98 | **962** |

结论：

1. V1 证明“允许跨层下探”能提高 gold coverage，但结构边淹没了关系边，95.6/96
   节点预算几乎耗尽，不能作为系统实现；
2. V2 是 accuracy-heavy 点，相比旧调度 32/48 all-hit 提升 2.5/1.5pp，但平均多访问
   24.1 个节点和 30.6 条边；
3. V3 是默认 Pareto 点，相比旧调度只增加 3.59 个节点/边，32/48 all-hit 均提升
   0.5pp，48-turn recall 提升 0.42pp，precision 不退化；
4. V3 的 `coarse_related` walk 从 435 增至 962，同时真正产生 729 次 `refines_to`
   和 1,038 次 `scene_contains`，说明关系扩展和证据物化不再互相抢 beam。

逐题 paired bootstrap 中，V3 在 32 和 48 turns 各新增 1 道 all-hit，均无 all-hit
退化；48-turn recall 增量为 +0.42pp，95% CI `[0,+1.08pp]`。

这里不采用本轮 wall-clock latency 做结论：两张图并行运行并共享本地 embedding 服务，
OS cache 和服务排队会污染逐题延迟。visited node/edge 是本轮可复现的检索工作量代理；
正式 latency/QPS 需要在固定 worker、预热、单独运行条件下重测。

## 4. 稀有词边在新调度下的结果

在 V3 上启用 V5.15 `lexical_rare` 图后：

- `coarse_related` walk：962 → 1,936；
- visited nodes/edges：69.90/36.98 → 84.58/51.66；
- 32/48 all-hit：均不变；
- gold hits/题：均仅 +0.005。

因此新调度已经能利用这些边并下探，当前瓶颈不再是“路径走不到叶层”，而是 edge
只记录 `lexical_rare` 布尔属性，不记录共享 anchor 及其 parent-local child posting。
更多路径主要产生额外候选，没有稳定转化为 packed evidence。该构图开关继续默认关闭。

## 5. 工程接口

- `expansion_beam`：每个节点的非结构关系 fanout，默认 2；
- `hierarchy_descent_beam`：关系到达后每层保留的结构子分支数，默认 1；
- `max_hops`：只计非结构关系 hop；
- `max_visited_nodes/max_visited_edges/max_frontier`：横向和纵向搜索共享的硬安全预算；
- `hierarchy_root_beam/hierarchy_child_beam`：seed 阶段初次自顶向下 routing，和关系到达后的
  `hierarchy_descent_beam` 不同。

四个 runtime profile 已显式写入 `hierarchy_descent_beam=1`。结构/关系两个 beam 可以独立
做 accuracy-cost 消融，不再通过一个参数隐式耦合。

## 6. 下一步实验

1. 增加 `completed_leaf_corridor_rate`、关系到达层级、每层候选数和 pruning margin 遥测，
   明确区分“无叶节点”“排序选错子节点”“全局预算中断”；
2. 固定 V3，消融 \(B_d\in\{1,2,4\}\)、\(B_r\in\{1,2,4\}\)、
   \(H_r\in\{1,2,3\}\)，绘制 all-hit/precision 对 visited edges 的 Pareto 图；
3. 为 coarse edge 保存有界 top shared anchors+IDF sidecar，并用
   `query/seed anchors ∩ edge anchors` 条件化边选择和 child posting 下探；
4. 分题型统计 relation→leaf 的新增 packed gold，特别检查 LoCoMo temporal 当前的负担；
5. 在隔离服务上补做单 worker 与 1/4/16/64 并发的 warmed latency/QPS，比较旧调度、
   V3 默认和 V2 accuracy-heavy 三个运行点。

## 7. 实物

- 默认 V3 基础图：
  `../artifacts/report/v5_15/accuracy_budget_relation_mask_layered_v3_dev200/summary.json`
- V3 稀有词图：
  `../artifacts/report/v5_15/accuracy_budget_relation_mask_lexical_layered_v3_dev200/summary.json`
- 默认 V3 与旧调度配对：
  `../artifacts/report/v5_15/layered_v3_paired_vs_v5_14.json`
- V2 accuracy-heavy：
  `../artifacts/report/v5_15/accuracy_budget_relation_mask_layered_v2_dev200/summary.json`
