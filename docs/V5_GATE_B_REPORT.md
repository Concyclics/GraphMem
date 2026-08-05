# GraphMem 5.0 Gate B implementation and ablation report

## Decision

Gate B is implemented and verified on the frozen 200-question development set.
The selected configuration is **B5 deterministic hierarchy with explicit L2
fanout-8 routing cards, portals, and N5 set-cover navigation**. Selective 30B
refine is disabled in the winner because it added edges and tokens without a
measurable quality gain.

Gate A remains frozen at commit `3cc914d`. Gate B was developed on
`codex/graphmem-v5-gate-b`; no V4/V4.1 implementation or frozen artifact was
deleted or overwritten.

## Implemented vertical slice

- Typed Scene, EventSkeleton, CanonicalEntity, TimeAnchor, StateHead,
  CollectionScope and L0-L3 RoutingCard contracts with deterministic IDs and
  multi-evidence provenance.
- SQLite WAL authority for lossless turns, FTS, canonical graph, embeddings,
  LLM cache/calls, outbox and run ledger.
- GraphReadView with relation-specific forward/inverse adjacency, entity/time
  indexes and role/provenance bitsets.
- N1-N5 navigation: normalized exact/FTS/dense fusion, session and adjacent-turn
  propagation, terminal provenance closure, certificate-guided priority, and
  budgeted set cover.
- B0-B5 construction: 2-8 turn scenes, at most three event skeletons per scene,
  deterministic coreference, bounded typed candidates, value-gated single-model
  Qwen3-30B refine, L2 fanout-8 routing and cross-session portals.
- Qwen3-Embedding-0.6B indexing with content-hash reuse. Embedding usage is
  audited separately from memory-backbone tokens.
- Neo4j 5.26.2 winner-only projection with 1000/2000 node/edge batches,
  constraints, checksum/count validation and three-runtime parity.

Online build, retrieval, runtime and storage packages do not import answers,
gold sessions/turns or question categories. Neo4j contains no raw text, LLM
payload or embedding/vector property.

## Calibration results: fixed 40 questions

The fixed SHA-256/seed-42 calibration contains ten questions from each stratum.

| Configuration | Equal-stratum exact turn all-hit | Build backbone token | Mean evidence token |
|---|---:|---:|---:|
| B0 lossless + cards | 45.0% | 0 | 1,991 |
| B1 scenes/events | 42.5% | 0 | 1,992 |
| B2 entity/time/state hubs | 42.5% | 0 | 1,992 |
| B3 deterministic typed edges | 45.0% | 0 | 1,842 |
| B4 + selective 30B refine | 45.0% | 423,030 clean-build | 1,842 |
| B5 hierarchy + portals | 47.5% | 423,030 if refine enabled | 1,978 |
| Frozen B6 V4.1 paired reference | 27.5% | frozen legacy accounting | — |

N5 reached 47.5%, versus 45.0% for N1-N4, while using fewer evidence tokens.
The 30B calibration issued 108 uncached and 108 cached batches. It recorded
386,232 uncached input, 36,798 output, 386,232 cached input and zero reasoning
tokens. Across 30 calibration memories, clean selective-refine cost averaged
14,101 backbone tokens per memory. The frozen legacy build averaged about
249,631 build tokens per memory, so the selective design is approximately
94.4% lower per memory; the final no-refine winner uses zero build-backbone
tokens.

B4 produced 973 retained Qwen edges but exactly matched B3 quality. The useful
B5 gain came from deterministic hierarchy/portal structure, not refine.

The one-factor funnel produced identical 47.5% calibration all-hit for fanout
4/8/16. Fanout 4 was the calibration tie-break winner (60.10 visited nodes),
but its required full-200 confirmation reached 51.0%, versus 51.5% for fanout
8, and used slightly more visited nodes (61.46 versus 61.43). Fanout 8 therefore
remains the final winner under the within-one-point efficiency rule. Disabling
cross-session merge reduced calibration all-hit from 47.5% to 42.5%, so merge
remains enabled.

## Full 200-question confirmation

After adding the explicit L2 fanout-8 layer, the final B5 result is:

| Stratum | B5 exact turn all-hit | Frozen B6 | Difference |
|---|---:|---:|---:|
| LongMemEval multi-session | 68% | 60% | +8 pp |
| LongMemEval temporal | 80% | 82% | -2 pp |
| LoCoMo Cat1 multi-hop | 4% | 0% | +4 pp |
| LoCoMo Cat2 temporal | 54% | 10% | +44 pp |
| Equal-stratum mean | **51.5%** | **38.0%** | **+13.5 pp** |

This passes the quality-first rule: the mean is higher and no stratum regresses
by more than two percentage points. Other final metrics are:

- session any-hit 94.0%, session all-hit 71.5%, mean session recall 84.06%;
- turn any-hit 73.5%, mean turn recall 61.56%, mean precision 7.63%;
- candidate all-hit 97.0%, candidate recall 98.94%;
- graph-reachable turn recall 93.41%, path provenance complete 100%;
- 61.43 visited nodes, 52.12 visited edges and 2,022 evidence tokens on average;
- median/p95 total navigation latency 198/436 ms;
- median/p95 seed fusion 102/245 ms, GraphReadView 1.06/1.54 ms,
  provenance closure 5.81/10.18 ms, packing 15.90/30.61 ms.

A paired, stratum-preserving bootstrap with 10,000 seed-42 resamples estimates
the B5-minus-B6 all-hit difference at +13.5 points (95% CI +8.0 to +19.0
points; all resamples favored B5). The calibration Pareto frontier is emitted
with the exact points rather than collapsing quality and cost into a synthetic
token-F1 score.

Failure attribution is 103 successes, 43 pack-drop/budget exhaustion cases, 42
routing misses and 12 seed misses. Candidate recall is already high; the next
quality target should be evidence packing and multi-session routing, especially
LoCoMo Cat1, rather than adding more graph nodes or generic refine edges.

The final graph contains 110 memories, 55,323 source turns, 123,752 semantic
nodes and 174,683 edges. Relation-specific degree is bounded at 12; degree p50
is 2 and p95 is 5.

## Authoritative artifacts

- Final L2/B5 200-question confirmation:
  `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5/navigator_only_20260804T183133Z`
- Complete 40-question single-LLM calibration and calls:
  `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5/gate_b_cal40_20260804T174951Z`
- Pre-L2 B0/B5 full comparison:
  `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5/gate_b_full200_20260804T175820Z`
- Fanout/cross-session one-factor funnel:
  `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5/funnel_scan40_20260804T184812Z`
- Paired bootstrap, Pareto points and recall-vs-evidence plot:
  `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5/gate_b_analysis_20260805`
- Fanout-4 full-200 confirmation:
  `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5/navigator_only_20260804T185043Z`

The final directory contains SQLite, exact per-question metrics, graph manifest,
compressed graph snapshot and Neo4j parity. The calibration directory contains
the full SQLite call ledger plus `llm_calls.jsonl.gz` with concrete request,
response, usage, cache status, latency, retry count, batch size and prompt hash.

Neo4j parity verified 110 memories, 123,752 nodes and 174,683 edges in both
SQLite and Neo4j. Five representative memories returned identical edge paths
for `sqlite_snapshot`, `neo4j_direct` and `neo4j_cached`; forbidden Neo4j
property leaks were empty. The container was stopped with its volume retained.
Embedding and 30B services and heartbeats were stopped; GPU1/2/3 were released.
