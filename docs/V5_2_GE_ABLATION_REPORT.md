# GraphMem V5.2 G0–G4 / E0–E4 calibration report

Date: 2026-08-05

## Scope and reproducibility

This is a calibration experiment, not the final 200-question confirmation. It uses the fixed
development subset of 40 questions (10 questions from each of LongMemEval multi-session,
LongMemEval temporal, LoCoMo Cat1 multi-hop and LoCoMo Cat2 temporal), covering 22 unique
memories. All configurations use the same N5 navigator, embedding index, query budget and
Qwen3-30B-A3B-Instruct-2507-FP8 backbone with reasoning disabled.

Artifacts:

- G0–G4: `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5_2/v5_1_graph_ablation_development_20260805T053904Z`
- E0–E4: `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5_2/v5_1_graph_ablation_development_20260805T060500Z`
- Interrupted E run used only to reconstruct cold token cost:
  `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5_2/v5_1_graph_ablation_development_20260805T054505Z`

Each memory was built in an independent SQLite shard and atomically merged into the run
authority database. The token table below deduplicates calls by `cache_key` across the
interrupted and resumed runs. It therefore does not mistake a cache hit for a zero-cost graph.

## G0–G4 relation graph ablation

| Profile | Semantic nodes | Edges | Final turn all-hit | Relation-only all-hit | Shuffled all-hit | Relation gain |
|---|---:|---:|---:|---:|---:|---:|
| G0 current | 32,338 | 37,976 | 45.0% | 92.5% | 87.5% | +5.0 pp |
| G1 remove noisy value/activity edges | 32,338 | 33,397 | 45.0% | 97.5% | 92.5% | +5.0 pp |
| G2 add time/state edges | 32,969 | 35,943 | 45.0% | 92.5% | 90.0% | +2.5 pp |
| G3 add collection scopes | 33,210 | 36,896 | 45.0% | 92.5% | 82.5% | **+10.0 pp** |
| G4 add cross-session value/coreference | 33,234 | 37,481 | 45.0% | 92.5% | 92.5% | 0.0 pp |

All five variants have 100% candidate all-hit and 100% graph-reachable recall on this subset,
but only 45% survives the final 16-turn evidence pack. Consequently the graph is not losing
gold evidence here; ranking and packing are the immediate bottleneck. The fact that every run
also exhausts its navigation budget supports the same diagnosis.

G3 is the best relation design for further work: its typed structure loses ten points when the
relation labels are shuffled, twice the G0 gap. G4 should not be promoted: cross-session
shared-value/coreference edges remove the entire relation-specific gain, indicating
over-connection rather than useful routing. G1 is a useful cheap control because deleting
4,579 `FACT_VALUE`/`SAME_ACTIVITY` edges raises relation-only all-hit without changing final
quality.

G2's temporal edges do not improve temporal all-hit. The current implementation orders raw
time strings and groups state chains by lexical predicates. A useful temporal graph needs
normalized intervals, relative-date anchoring, canonical predicates and query-aware temporal
operators; adding `AT_TIME` and lexically sorted `TEMPORAL_BEFORE` edges alone is insufficient.

## E0–E4 extraction/compression ablation on G3

| Profile | Cold-equivalent token/memory | vs E0 | Repair calls / scene calls | Final all-hit | Relation / shuffled | Evidence tokens |
|---|---:|---:|---:|---:|---:|---:|
| E0 current extraction + LLM hierarchy | 318,706 | baseline | 351 / 932 (37.7%) | 45.0% | 92.5% / 82.5% | 1,994 |
| E1 facts=4, summary=32, output=1024 | 277,245 | -13.0% | 479 / 932 (51.4%) | 45.0% | 97.5% / 95.0% | 1,974 |
| E2 E1 + JSON object mode | 277,974 | -12.8% | 475 / 932 (51.0%) | **47.5%** | 97.5% / 95.0% | 1,961 |
| E3 batch=2 + per-scene repair | 282,236 | -11.4% | 576 / 1,854 (31.1%) | 45.0% | 97.5% / 87.5% | 1,980 |
| E4 E3 + deterministic hierarchy | **236,567** | **-25.8%** | 576 / 1,854 (31.1%) | **47.5%** | 95.0% / 85.0% | **1,942** |

E4's recorded calls are cache hits because it deliberately shares the E3 scene-extraction
configuration. Its cold-equivalent cost is the unique E3 scene and repair cost (5,204,467
tokens) with the 1,004,716 hierarchy tokens removed, divided by 22 memories. It is not a
zero-token build. All measured reasoning-token counts are zero.

The compact prompt itself did not solve cost. E1/E2 lower output limits caused more missing
scenes, raising repair incidence from 37.7% to about 51%. E3's smaller batches reduce the repair
ratio but double scene-call count and duplicate prompt/context overhead. E4 is the calibration
Pareto choice because deterministic hierarchy removes a whole LLM stage while tying the best
final all-hit, but it misses the planned 30% cost reduction by 4.2 percentage points and must
not yet be declared the winner.

The only all-hit improvement is one LongMemEval temporal question (70% to 80%). LoCoMo Cat1
remains 0% all-hit under every configuration even though its candidate and reachable all-hit
are 100%. This is strong evidence that the next gain should come from multi-evidence set-cover,
slot closure and packing rather than adding more graph edges or a larger construction model.

## Recommended next iteration

1. Use G3 + E4 as the next candidate, while retaining G1 as the sparse control. Validate both
   on the untouched split before a full 200-question confirmation.
2. Replace JSON-object mode with a strict schema and emit exactly one scene object per initial
   call, or use a local deterministic fallback for content-light scenes. The target is repair
   incidence below 5%; simply lowering `max_tokens` is counterproductive.
3. Keep hierarchy construction deterministic. Spend LLM tokens only on grounded scene facts
   that can change an entity/state/time/collection route.
4. Normalize dates into intervals with provenance; anchor relative expressions to session time;
   canonicalize state predicates before creating `STATE_NEXT`; do not order raw time strings.
5. Optimize navigation before judging further graph growth: candidate/reachable recall is already
   100%, final all-hit is 45–47.5%, every query exhausts budget, and LoCoMo multi-hop evidence is
   dropped during ranking/packing.
6. Add per-question pack-loss traces containing candidate rank, slot contribution, marginal
   set-cover gain, token cost and drop reason. This separates seed/routing failure from pack loss
   without using answer or gold labels online.

## Decision

Promote neither configuration to final status yet. G3 establishes the strongest typed-relation
signal; E4 is the current cost/quality Pareto candidate. The next experiment should test strict
single-scene extraction plus deterministic hierarchy, followed by navigator/packer optimization
on the fixed calibration set and confirmation on held-out questions.
