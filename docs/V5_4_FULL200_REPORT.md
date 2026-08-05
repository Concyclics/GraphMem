# GraphMem V5.4 full-200 evaluation

Date: 2026-08-05

## Result

V5.4 completed the frozen 200-question development set: 110 memories and four
50-question strata. The run is complete and internally consistent, but V5.4
should remain experimental. It passes the average build-token, retry, reasoning,
and relation-signal gates; it does not pass the 95% relation-only all-hit gate,
and its full-set fixed-N5 turn all-hit is only 49.5%.

Authoritative artifact:

`/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5_4/full200_resume/v5_1_graph_ablation_full_20260805T103058Z`

The directory contains the merged SQLite authority database, 110 per-memory
shards, graph and run manifests, navigation and relation traces, CSV/Parquet
metrics, token ledger, error cases, and the HTML ablation report. The run
manifest records 200 unique question IDs and zero reasoning tokens.

## Fixed-N5 navigation

| Stratum | Session all-hit | Turn all-hit | Turn recall | Candidate all-hit | Candidate recall | Reachable recall |
|---|---:|---:|---:|---:|---:|---:|
| LongMemEval multi-session | 90% | 60% | 72.3% | 98% | 99.0% | 57.7% |
| LongMemEval temporal | 82% | 72% | 78.3% | 98% | 98.0% | 60.5% |
| LoCoMo Cat1 multi-hop | 38% | 6% | 25.7% | 60% | 83.2% | 30.8% |
| LoCoMo Cat2 temporal | 78% | 60% | 68.2% | 96% | 97.0% | 55.2% |
| Equal-stratum aggregate | 72% | 49.5% | 61.1% | 88% | 94.3% | 51.0% |

The packed result retrieves 14,763 nodes and 12,386 edges in total, averaging
73.8 nodes, 61.9 edges, and 2,011 evidence tokens per question. Every question
reports budget exhaustion, so these values describe a saturated navigator rather
than a naturally converged one.

The fixed-40 calibration reported 55% all-hit. The full-set result is 5.5
percentage points lower, showing that the calibration split overestimated
generalization. There is no V5.3 full-200 paired artifact, so no full-set V5.3
delta is claimed.

## Graph relation diagnostics

| Diagnostic | Turn all-hit | Mean recall |
|---|---:|---:|
| Oracle seed | 95% | 98.6% |
| Relation-only | 64% | 76.1% |
| Shuffled relation | 16% | 27.1% |
| Relation-minus-shuffled | +48 pp | +49.0 pp |

Relation-only all-hit by stratum is 78% LongMemEval multi-session, 90%
LongMemEval temporal, 20% LoCoMo Cat1, and 68% LoCoMo Cat2. The corresponding
shuffle gaps are 76, 60, 20, and 36 percentage points. Typed topology therefore
contains a strong real navigation signal, especially on LongMemEval; it is not
merely reproducing seed quality.

Actual traversal uses 13,648 `SCENE_CONTAINS`, 10,554 `HAS_FACT`, 2,622
`PARTICIPATES_IN`, 1,685 `REFINES_TO`, 153 `COLLECTION_CO_MEMBER`, 91 `AT_TIME`,
60 `STATE_NEXT`, and 25 `TEMPORAL_BEFORE` edges. Leave-one-relation-out remains
dominated by `SCENE_CONTAINS`: removing it lowers all-hit to 6.5%, while removing
any one lateral relation does not change the 64% point estimate. The lateral
relations are sparse and/or redundant under the current 96-node budget; their
existence alone does not yet make them causal navigation routes.

There is no abstract-provenance leakage. All terminal paths remain traceable to
SourceTurn evidence.

## Failure decomposition

There are 101 strict all-hit failures:

| Failure stage | Count |
|---|---:|
| Pack drop or budget exhausted | 44 |
| Routing miss | 42 |
| Seed miss | 12 |
| Within-session candidate miss | 3 |

The aggregate CandidatePool reaches all required turns for 176 of 200 questions,
but packing closes only 99. The 38.5-point candidate-to-packed gap is the largest
actionable loss and confirms that adding more extraction tokens is not the first
fix.

LoCoMo Cat1 is different from the other strata: 47 of 50 questions fail, and 20
already lack complete candidates. Its multi-event list/count and cross-scene
closure requirements expose genuine indexing gaps as well as navigation losses.
LongMemEval and LoCoMo Cat2 have 98%, 98%, and 96% candidate all-hit respectively;
their remaining failures are primarily relation scheduling, certificate closure,
and evidence packing.

## Build cost and robustness

- Cold-equivalent backbone input plus output: 23,635,622 tokens total, or
  214,869 tokens per memory on average.
- Per-memory distribution: p50 222,720; p95 244,026; maximum 260,847 tokens.
- 65 of 110 memories exceed 220k individually, although the average remains
  below the stated 220k gate.
- Main scene extraction: 22,802,959 tokens; retry: 832,663 tokens.
- Retry rate: 3.30%; reasoning token: 0.
- Accepted semantic terminal coverage: 57.6%; lossless terminal turn coverage:
  100%; accepted facts per scene: 2.66.

The mean cost gate passes, but the distribution is poorly controlled. Future
acceptance should add a percentile or exceedance constraint rather than relying
only on the mean.

## Run interruption and correction

The initial full run encountered the phrase “6000 years ago”. The relative-time
normalizer attempted to construct a Python date outside its supported range.
V5.4 now bounds relative day/week/month/year magnitudes and preserves extreme
expressions as unresolved raw temporal provenance instead of crashing. A focused
regression test covers this case. The resumed run reused completed immutable
per-memory shards and rebuilt the affected memory; no frozen artifact was
overwritten.

## Recommendation

Do not spend additional backbone tokens on generic scene extraction. The next
iteration should:

1. Schedule relation expansion by unresolved certificate slot, reserving budget
   for collection, temporal, and state closure before bulk `SCENE_CONTAINS`.
2. Make collection/list/count facts first-class query targets for LoCoMo Cat1,
   including bounded cross-scene event aggregation and direct terminal postings.
3. Replace the current uniform evidence packer with marginal slot-gain set cover
   and stop expanding once the evidence certificate is closed.
4. Add a per-memory token p95 gate and shorten or deterministically handle long
   low-value turns before invoking the LLM.

V5.4 demonstrates that the graph contains useful typed structure, but the fixed
N5 navigator does not yet exploit that structure efficiently enough for release.
