# GraphMem V5.3 lean graph calibration report

## Decision

V5.3 substantially improves cold build cost, provenance correctness, and the causal value of typed relations, but it does not pass the predeclared 95% relation-only all-hit gate. The frozen F0/G3+E4 configuration therefore remains the release reference. No 34-question validation or 200-question confirmation was started.

## Prompt decision

Three strict extraction designs were probed with Qwen3-30B-A3B-Instruct-2507-FP8, with reasoning disabled:

- nested numeric offsets caused whitespace/length failures and roughly 540k--585k token/memory;
- quote-grounded max-2 facts reduced retry to 0.216% and averaged 199,131 token/memory on the 22-memory calibration set;
- compact quote-grounded max-3 facts removed model-generated `value_type` and `time`. Value type and temporal expressions are derived locally from the exact quote. This improved semantic coverage while reducing the same calibration set to 180,512 token/memory.

The selected prompt passes local aliases (`s0`, `s0t0`, ...), speaker, observation timestamp `d`, and raw turn text. The timestamp is explicitly marked as observation metadata and cannot be emitted as event time. The model only selects grounded facts and exact quotes; rules extract a temporal phrase from the quote and normalize it against the cited turn timestamp.

## Temporal representation

Each normalized `TemporalInterval` records raw text, start/end, precision, kind, anchor turn, and confidence. `observed_at` and event time are stored separately. Scene, entity, state, and L1--L3 route cards receive deterministic event and observation ranges from their children. `AT_TIME` and `TEMPORAL_BEFORE` are created only for reliable event intervals; a conversation timestamp never becomes an event-order edge by itself.

## Final 40-question calibration

The fixed set contains 10 questions from each of LongMemEval multi-session, LongMemEval temporal, LoCoMo Cat1 multi-hop, and LoCoMo Cat2 temporal.

Final auditable artifacts are under `/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5_3/calibration40_final_artifacts/v5_1_graph_ablation_development_20260805T081726Z`.

| Metric | F0 frozen | V5.3 F4 final |
|---|---:|---:|
| cold-equivalent backbone token/memory | 236,567 | 180,512 |
| extraction retry rate | legacy repair path | 0.324% |
| reasoning token | 0 | 0 |
| terminal turn coverage | legacy/fat provenance | 100% lossless compact refs |
| N5 turn all-hit | 52.5% | 55.0% |
| relation-only all-hit | 95.0% (entity provenance leakage present) | 62.5% |
| shuffled all-hit | 92.5% | 22.5% |
| relation-minus-shuffled | 2.5pp | 40.0pp |

V5.3 relation-only strata were 80% LongMemEval multi-session, 100% LongMemEval temporal, 10% LoCoMo Cat1, and 60% LoCoMo temporal. The Cat1 result is the blocking gate.

The final graph uses a compact terminal `EvidenceGroupRef` for every turn. It stores a deterministic speaker/keyphrase sketch, numeric/time type, and provenance ID, not raw text. A Scene reaches the ref through `SCENE_CONTAINS`. This raises truthful global terminal coverage to 100% without allowing a RoutingCard or CanonicalEntity to expand an entire session. CanonicalEntity is route-only.

## Failure attribution

- The max-2 prompt was over-compressed; max-3 with fewer JSON fields is both cheaper and semantically richer.
- Broad CanonicalEntity provenance produced false coverage and has been removed.
- The original relation probe used edge-ID BFS and lexical-only seeds. It is now a deterministic best-first typed traversal with the same Dense/BM25/Exact turn fusion mapped to Scene routes; all relation ablations share a cached seed computation.
- High-precision portals were sparse and did not improve quality, so F4 without portals remains the candidate.
- LoCoMo Cat1 often requires closure over many turns in multiple scenes. Under 2 hops/96 nodes, current owner/state/collection edges do not close every evidence set. Adding broad shared-value or entity edges would improve the development metric by leakage and was rejected.

## Next bounded experiment

The next construction experiment should target cross-scene evidence-set closure without changing the query budget: fact-to-evidence-ref grounding edges, predicate-compatible CollectionScope postings, and bounded owner/predicate portals. It must be calibrated on held-out memories and retain the 40pp shuffle gap. Until relation-only all-hit reaches the declared threshold, F0 remains the release reference and full-200 rebuilding is not authorized by the gate.
