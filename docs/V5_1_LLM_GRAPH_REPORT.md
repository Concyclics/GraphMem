# GraphMem 5.1 LLM graph construction report

## Status

GraphMem 5.1 adds grounded Qwen3-30B scene extraction, L1-L3 routing-card
compression, canonical fact/value nodes, relation-specific cross-session
regions, and causal graph-navigation probes. The implementation intentionally
keeps the V5 B5/N5 baseline unchanged and does not promote the new graph based
on the smoke results below.

The frozen holdout protocol uses six/two/two LoCoMo conversations and
30/10/10 LongMemEval questions per stratum for development, validation and
final evaluation. The existing 200 questions are used only for a compatibility
comparison after configuration freeze.

## Grounded semantic construction

- Every scene is sent to Qwen3-30B with turn IDs, speakers, timestamps, text
  lengths and complete text. No query, answer, gold evidence or category enters
  the build prompt.
- Facts require owner, predicate, value, scope, polarity, confidence and valid
  source-turn spans. Invalid offsets are recovered only when the value occurs
  literally in the cited turn; otherwise the fact is discarded.
- L1/L2/L3 cards are compressed from child semantic records and retain child
  postings. Cross-session value aliases may produce `COREFERENCE` edges.
- The new graph uses `HAS_FACT`, `FACT_VALUE`, `SHARED_VALUE`,
  `SAME_ACTIVITY`, and `COREFERENCE`; it does not emit generic `PORTAL` edges.
- Full calls, cache hits, malformed responses, repair calls, token use and
  reasoning-token assertions remain in the SQLite ledger.

## Single-memory probes

The first long LongMemEval smoke exposed an invalid scalar `child_postings`
shape. The parser rejected it explicitly; the implementation now accepts only
validated child-ID arrays. It also exposed output truncation and motivated the
compact scene schema and bounded hierarchy outputs.

The first LoCoMo Cat1 probe (`locomo03_0051`, conversation `conv-42`) used 90
scenes and 29 sessions. With the compact schema it consumed 131,628 clean
backbone tokens, produced 100 canonical facts, 82 values, two virtual regions
and 20 shared-value edges. Relation-only navigation improved the gold-turn
recall from 0% at seed-only to 50%, but did not recover both evidence turns;
frozen N5 remained at 0%. This is evidence that graph edges can recover a key
session/turn, but not evidence that the graph is yet complete.

Adding timestamps and hierarchy alias proposals created a `COREFERENCE` edge,
but the second probe still recovered only one of the two gold turns. That run
mixed two prompt versions in its source ledger; prompt-specific accounting for
the new version was 182,489 tokens, just above the 180K gate. The final default
therefore reduces scene batch size from eight to four to avoid incomplete
multi-scene responses and expensive repair calls.

## Four-stratum engineering probe

The final four-scenes-per-batch setting was exercised on one frozen development
question from each stratum (four questions, three distinct memories) with the
30B service on GPU 2/3 at tensor parallel two. This is an engineering probe,
not a statistically meaningful selection run.

| Profile | final turn all-hit | relation-only all-hit | seed-only all-hit | mean uncached tokens/memory |
| --- | ---: | ---: | ---: | ---: |
| C0 | 0.75 | 1.00 | 0.75 | 0 |
| C1 | 0.25 | 0.50 | 0.50 | 243,150 |
| C2 | 0.25 | 0.50 | 0.50 | 243,150 |
| C3 | 0.25 | 0.75 | 0.50 | 287,084 |
| C4 | 0.25 | 0.75 | 0.50 | 287,084 |

C3/C4 relation traversal added 0.25 mean gold-turn recall beyond seeds, while
the degree-preserving shuffled control did not. This is positive causal
evidence that the semantic relations carry useful routing information. The
fixed N5 candidate ranking and set-cover packer nevertheless dropped those
additional turns: candidate all-hit was 1.00 for every profile, but C3/C4 final
all-hit was only 0.25. The immediate bottleneck is therefore no longer only
graph reachability; relation-derived candidate credit is not preserved through
packing.

Four-scene batching did not meet the token gate. Across the three memories it
used 861,253 uncached tokens: 559,776 for initial scene extraction, 169,675 for
44 repair calls after 114 scene calls, and 131,802 for L1-L3 compression. The
38.6% repair-call ratio and 184,093 scene output tokens show that reducing the
batch size alone does not control verbose or structurally incomplete output.
The next iteration should constrain per-scene output more aggressively and
repair only missing scene IDs, while separately fixing relation-aware packing.

## Promotion decision

The new graph is **not promoted**. Current evidence shows:

- grounded fact nodes and typed relations are constructible with zero reasoning
  tokens;
- relation traversal can add a missing gold turn beyond seed-only retrieval;
- on the four-stratum probe, semantic relations beat the shuffled control but
  their recovered turns are lost during final packing;
- alias/time support addresses real schema gaps without question-specific
  rules;
- the token gate is missed by a wide margin and the full development/validation
  gates have not run.

## Full 200-question hard-set confirmation

The frozen C0 baseline and C3/C4 semantic graphs were compared on all 200 hard
questions (50 per stratum). C3 and C4 were identical, so cross-session merge as
currently implemented adds no measurable navigation behavior.

| Profile | Equal-stratum turn all-hit | LME multi-session | LME temporal | LoCoMo Cat1 | LoCoMo Cat2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0 | 51.5% | 68% | 80% | 4% | 54% |
| C3 | 48.0% | 62% | 76% | 2% | 52% |
| C4 | 48.0% | 62% | 76% | 2% | 52% |

The comparison is paired: 193 questions were unchanged and seven regressed;
no question changed from miss to all-hit. At the same time C3/C4 achieved 100%
candidate all-hit and 100% graph-reachable recall in every stratum. The graph
therefore contains and reaches the gold evidence, but N5 does not preserve it
through scoring and packing.

Relation-only traversal reached 93% all-hit versus 40% for seed-only and 88%
for degree-preserving shuffled edges. Typed relations carry a real but modest
five-point advantage over graph connectivity alone. Removing `FACT_VALUE`
increased relation-only all-hit from 93.0% to 95.5%; removing `SAME_ACTIVITY`
increased it to 94.0%; removing `SHARED_VALUE` had no effect. These relations
should not receive positive ranking credit without stronger selectivity.

The full build used 34,492,165 uncached backbone tokens, or 313,565 per memory:
22,810,454 scene extraction, 6,607,421 repair, and 5,074,290 hierarchy
compression. There were 1,707 repair calls after 4,599 scene calls (37.1%).
The result misses both the quality and token gates and is not promoted.

For the full run, construction was changed to one SQLite shard per memory.
Each shard contains only that memory's raw rows, graph, LLM cache and ledger;
eight shards run concurrently and merge atomically into the run authority.
At 128 in-flight requests, TP2 generation throughput reached roughly 3,629
tokens/s. Shards remain available for per-memory audit and restart.

The next experiment should stay on the frozen development split and first
preserve relation-derived gold candidates through a relation-aware packer.
Separately, it should cap per-scene facts/output and replace broad repair with
missing-scene-only extraction. `FACT_VALUE`, `SAME_ACTIVITY` and
`SHARED_VALUE` should default to neutral ranking credit until precision gates
pass. Until those changes pass the declared quality and token gates, V5 C0
remains authoritative and Neo4j should not receive a V5.1 projection.

## Artifact paths

- Split manifest: `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5_1/split_manifest.json`
- Initial Cat1 C0/C4 probe:
  `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5_1/v5_1_graph_ablation_development_20260805T024204Z`
- Timestamp/alias follow-up:
  `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5_1/v5_1_graph_ablation_development_20260805T024950Z`
- Four-stratum C0-C4 engineering probe:
  `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5_1/v5_1_graph_ablation_development_20260805T030512Z`
- Full 200-question C0 baseline:
  `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5_1/v5_1_graph_ablation_full_20260805T035454Z`
- Full 200-question C3/C4 run with per-memory SQLite shards:
  `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5_1/v5_1_graph_ablation_full_20260805T040219Z`
