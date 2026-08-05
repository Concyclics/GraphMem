# GraphMem V5.4 navigable graph calibration

Date: 2026-08-05

## Decision

V5.4 remains an experimental graph profile. It meets the cold-token and retry
budgets and improves graph reachability, collection semantics, and the
relation-shuffle gap, but it does not meet the 95% relation-only all-hit gate and
regresses the fixed N5 LongMemEval temporal stratum by one question. V5.3 must
therefore remain the frozen release reference; the 200-question confirmation is
not triggered.

Final fixed-40 artifact:

`/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5_4/calibration40_v5/v5_1_graph_ablation_development_20260805T094433Z`

The interrupted cold run and its resumed authority DB are retained under
`artifacts/v5_4/calibration40_v3*`. The interruption was an embedding endpoint
restart; predicate embedding now retries transient failures three times.

## Implemented changes

- Strict two-scene extraction with exact output cardinality and deterministic
  recovery of duplicated scene aliases.
- Recovery of complete scene objects from a length-truncated JSON root, so only
  the missing scene is retried.
- A 2,500-character per-turn LLM input cap using exact head/tail excerpts. Raw
  SourceTurn text and evidence provenance remain lossless in SQLite.
- Prompt prioritization for counts, ordinals, named objects, wins,
  acquisitions, state changes, temporal expressions, and media-caption facts.
- Canonical predicate/scope families for win, participate, write, receive,
  visit, read, attend, and buy, with conservative exclusions such as
  `recommends reading`.
- Fact modality and polarity partitions. Modality is inferred from the exact
  fact evidence rather than neighbouring sentences.
- Event-instance collections, direct bounded fact-to-fact
  `COLLECTION_CO_MEMBER` projections, and explicit `Scene -> Entity`
  `PARTICIPATES_IN` links for two-hop navigation.
- Relative weekday and duration normalization anchored to SourceTurn
  timestamps. Observation time is not treated as event time.

## Fixed-40 results

| Metric | V5.3 frozen | V5.4 final | Delta |
|---|---:|---:|---:|
| Cold-equivalent backbone token / memory | 180,512 | 215,095 | +19.2% |
| Extraction retry rate | 0.32% | 2.99% | +2.67 pp |
| Reasoning token | 0 | 0 | 0 |
| Fixed N5 turn all-hit | 55.0% | 55.0% | 0 pp |
| Relation-only turn all-hit | 62.5% | 65.0% | +2.5 pp |
| Shuffled turn all-hit | 22.5% | 20.0% | -2.5 pp |
| Relation-minus-shuffled gap | 40.0 pp | 45.0 pp | +5.0 pp |
| Terminal graph-reachable recall | 61.8% | 68.5% | +6.7 pp |

V5.4 stays under the 220k token/memory limit and below the 5% retry limit.
Token stages are 4,582,060 main extraction tokens and 150,027 retry tokens over
22 memories. Semantic terminal coverage is 56.2%, with 2.59 accepted facts per
scene. All SourceTurns still have lossless terminal fallback coverage.

Fixed N5 all-hit by stratum is 70% LongMemEval multi-session, 80%
LongMemEval temporal, 10% LoCoMo Cat1, and 60% LoCoMo Cat2. V5.3 was 70%, 90%,
0%, and 60%, respectively. The overall tie therefore hides a temporal
regression and a Cat1 improvement; V5.4 fails the no-stratum-regression release
gate.

## Real graph inspection

In `locomo:conv-42`, the final graph contains 905 CanonicalFacts, 349 Scenes,
41 CollectionScopes, 59 TimeAnchors, and 26 StateHeads across the two inspected
LoCoMo memories. The semantic edge projection includes `SCENE_CONTAINS`,
`PARTICIPATES_IN`, `HAS_FACT`, `COLLECTION_CO_MEMBER`, `AT_TIME`, `STATE_NEXT`,
and `TEMPORAL_BEFORE`.

The strongest corrected example is Nate's asserted positive `win/tournament`
collection: it has exactly nine members, including values that the extractor
occasionally represented only by a temporal adjunct. `last Friday`, `last
Saturday`, `yesterday`, and `last week` facts have normalized relative
intervals anchored to their conversation timestamps. Planned and negative facts
are partitioned from asserted positive event collections.

Remaining graph defects are real:

- Joanna's “third one” does not explicitly name a screenplay in the local
  evidence and remains in a generic writing collection. Automatically forcing
  it into `screenplay` would be an unsafe benchmark-specific inference.
- Some LLM facts remain verbose or conversational even after filtering.
- Leave-one-relation-out is dominated by `SCENE_CONTAINS`: removing it drops
  all-hit to 17.5%, while removing one lateral relation at a time does not change
  all-hit. Lateral edges are redundant in this probe rather than individually
  causal under the 96-node budget.
- Oracle-seed all-hit is only 92.5%, placing a hard ceiling below the proposed
  95% gate for these exact official evidence sets.

## LoCoMo failure audit

Several official all-hit failures are not clean graph misses:

- `locomo03_0069` (three turtles) includes `D8:3`, a turn about unread books,
  although the three-turtle media caption is in `D28:23`.
- `locomo04_0011` (sports besides basketball) marks three basketball-context
  turns plus the single surfing turn; the extra turns are not required to answer
  the question.
- `locomo04_0020` (Seattle before Chicago) includes an unrelated won-game turn
  and a generic trip-introduction turn in addition to the city-bearing turns.
- `locomo04_0040` requires outside geographic knowledge that the Smoky
  Mountains cross North Carolina/Tennessee; neither state name appears in the
  official evidence.
- `locomo04_0017` answers six wins but supplies five evidence turns, so exact
  turn all-hit does not by itself validate the count.

Other failures are genuine hard cases: multi-event closure for two letters,
nine tournaments, three screenplays and travel locations; temporal linkage for
the third screenplay and professional-career duration. Reports must continue to
publish both strict official all-hit and an adjudicated sufficient-evidence
metric rather than silently changing the benchmark.

## Next engineering target

The next improvement should be in retrieval and certificate closure, not a
larger graph or more extraction tokens. CandidatePool already has 95% all-hit
and 97.5% mean recall, while final N5 all-hit is 55%. A query-conditioned
relation scheduler should reserve budget for collection/state/temporal closure
before expanding all `SCENE_CONTAINS` children, and sufficient-evidence
evaluation should be reported separately for annotation-noisy LoCoMo items.
