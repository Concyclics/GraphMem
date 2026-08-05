# V5.6 re-baseline: correcting the LongMemEval gold annotation input (D0)

Every V5.4 and V5.5 run consumed the **pre-adjudication draft** annotation
file, not the finalized asset that ships in `eval_annotations/`. This page
re-measures the frozen V5.4 graph and the V5.5 H6+G2 configuration against
the finalized annotations so V5.6 gates have a defensible baseline.

All four runs use `PYTHONHASHSEED=0`. The retrieval code is unchanged from
commit `1a0779a`; the only variable is the `--gold` input and the source graph.

## The annotation change

| | draft (`lme-v5-dev100-draft-r1`) | finalized (`lme-v5-dev100-r1`) |
| --- | ---: | ---: |
| annotations | 207 | 217 |
| role `aggregation_member` | 0 | 56 |
| role `fact` | 205 | 88 |
| role `negative_scope` | 2 | 14 |
| role `temporal_endpoint` | 0 | 59 |
| sha256 | `9ce371fe0a36ceac…` | `58a931e233746201…` |

At the `(question_id, session_id, turn_index)` granularity that
`navigation_metrics` actually scores, the two files share **142** references;
**75** exist only in the finalized set and
**61** only in the draft. Roughly a third of the
LongMemEval gold turns changed, and the draft had almost no
`temporal_endpoint` or `aggregation_member` roles at all.

## Strict turn all-hit and candidate all-hit, by gold file

`strict` = `turn_all_hit` (all gold turns packed); `cand` = `candidate_turn_all_hit`.

### V5.4 frozen navigator (H0, G0)

| Stratum | draft strict / cand | finalized strict / cand | Δ strict | Δ cand |
| --- | ---: | ---: | ---: | ---: |
| LongMemEval multi-session | 60.0% / 98.0% | **58.0%** / **100.0%** | -2.0pp | +2.0pp |
| LongMemEval temporal | 72.0% / 98.0% | **78.0%** / **96.0%** | +6.0pp | -2.0pp |
| LoCoMo Cat1 multi-hop | 6.0% / 60.0% | **6.0%** / **60.0%** | +0.0pp | +0.0pp |
| LoCoMo Cat2 temporal | 60.0% / 96.0% | **60.0%** / **96.0%** | +0.0pp | +0.0pp |
| **All 200** | 49.5% / 88.0% | **50.5%** / **88.0%** | +1.0pp | +0.0pp |

### V5.5 harness (H6, G0)

| Stratum | draft strict / cand | finalized strict / cand | Δ strict | Δ cand |
| --- | ---: | ---: | ---: | ---: |
| LongMemEval multi-session | 68.0% / 80.0% | **70.0%** / **78.0%** | +2.0pp | -2.0pp |
| LongMemEval temporal | 74.0% / 84.0% | **78.0%** / **86.0%** | +4.0pp | +2.0pp |
| LoCoMo Cat1 multi-hop | 14.0% / 16.0% | **14.0%** / **16.0%** | +0.0pp | +0.0pp |
| LoCoMo Cat2 temporal | 58.0% / 60.0% | **58.0%** / **60.0%** | +0.0pp | +0.0pp |
| **All 200** | 53.5% / 60.0% | **55.0%** / **60.0%** | +1.5pp | +0.0pp |

### V5.5 harness (H6, G2 sidecar)

| Stratum | draft strict / cand | finalized strict / cand | Δ strict | Δ cand |
| --- | ---: | ---: | ---: | ---: |
| LongMemEval multi-session | 68.0% / 80.0% | **70.0%** / **78.0%** | +2.0pp | -2.0pp |
| LongMemEval temporal | 74.0% / 84.0% | **78.0%** / **86.0%** | +4.0pp | +2.0pp |
| LoCoMo Cat1 multi-hop | 18.0% / 82.0% | **18.0%** / **82.0%** | +0.0pp | +0.0pp |
| LoCoMo Cat2 temporal | 62.0% / 96.0% | **62.0%** / **96.0%** | +0.0pp | +0.0pp |
| **All 200** | 55.5% / 85.5% | **57.0%** / **85.5%** | +1.5pp | +0.0pp |

## Reproduction of the published figures

Under a fixed hash seed the draft-gold runs should land on the numbers in the
V5.4/V5.5 reports. Divergence here is the non-determinism of defect D1
(`packer.pack` iterates a `set` of mandatory turn ids).

| Configuration | Stratum | published | draft-gold replay | Δ |
| --- | --- | ---: | ---: | ---: |
| g0/h0 | LongMemEval multi-session | 60.0% | 60.0% | +0.0pp |
| g0/h0 | LongMemEval temporal | 72.0% | 72.0% | +0.0pp |
| g0/h0 | LoCoMo Cat1 multi-hop | 6.0% | 6.0% | +0.0pp |
| g0/h0 | LoCoMo Cat2 temporal | 60.0% | 60.0% | +0.0pp |
| g0/h0 | All 200 | 49.5% | 49.5% | +0.0pp |
| g2/h6 | LongMemEval multi-session | 68.0% | 68.0% | +0.0pp |
| g2/h6 | LongMemEval temporal | 74.0% | 74.0% | +0.0pp |
| g2/h6 | LoCoMo Cat1 multi-hop | 18.0% | 18.0% | +0.0pp |
| g2/h6 | LoCoMo Cat2 temporal | 62.0% | 62.0% | +0.0pp |
| g2/h6 | All 200 | 55.5% | 55.5% | +0.0pp |

## Where the candidate pool went (the H8 defect, quantified)

Measured on the finalized gold, same frozen graph, so the only difference
is the harness itself.

| Candidate pool size | H0 | H6 |
| --- | ---: | ---: |
| p50 | 269 | 46 |
| p90 | 355 | 53 |
| p99 | 422 | 74 |
| mean | 278.3 | 45.3 |
| max | 524 | 81 |

- The H6 pool is a **superset** of the H0 pool on **0/200** questions.
- The H6 pool is a **subset** of the H0 pool on **49/200** questions.
- **56/200** questions had every gold turn in the H0 pool but
  not in the H6 pool: {'lme_multi_session': 11, 'lme_temporal': 5, 'locomo_multihop': 22, 'locomo_temporal': 18}.

H8 must therefore restore a pool that dominates H0's by construction. Sizing
the id-only reservoir at **524+** entries covers H0's widest question;
the narrowing then happens only at hydration and packing, where it is
budget-driven and measured.

## What the G2 sidecar is actually contributing

G2 was built only for the ten LoCoMo memories where a prior H6 run missed
`candidate_turn_all_hit`, i.e. its build scope is selected using gold. On the
honest graph its headline gain largely disappears:

| Stratum | H6 on frozen V5.4 (honest) | H6 + G2 (gold-scoped) | Δ cand |
| --- | ---: | ---: | ---: |
| LongMemEval multi-session | 78.0% | 78.0% | +0.0pp |
| LongMemEval temporal | 86.0% | 86.0% | +0.0pp |
| LoCoMo Cat1 multi-hop | 16.0% | 82.0% | +66.0pp |
| LoCoMo Cat2 temporal | 60.0% | 96.0% | +36.0pp |
| **All 200** | 60.0% | 85.5% | +25.5pp |

LoCoMo Cat1 candidate all-hit is **16%** without the sidecar and 82% with it.
The published "candidate 82%, packed 18%, so the problem is packing" reading
does not survive this: on the honest graph Cat1 fails at routing and seeding
long before packing. PR6 must rebuild those postings globally and
question-independently before any Cat1 packing claim can be made.

## Corrected baseline for the V5.6 gate table

| Metric | A0 = V5.4 H0 | V5.5 H6+G2 |
| --- | ---: | ---: |
| Strict full-200 turn all-hit | 50.5% | 57.0% |
| Mean turn recall | 62.5% | 70.2% |
| Candidate all-hit | 88.0% | 85.5% |
| Candidate recall | 94.4% | 92.0% |
| Evidence tokens (heuristic estimate) | 2,011 | 3,345 |
| Certificate complete (pre-pack) | 100.0% | 67.0% |
| Pack turn cap reached | 0.0% | 52.5% |
| Pack token cap reached | 0.0% | 48.0% |
| LongMemEval multi-session candidate all-hit | 100.0% | 78.0% |
| LongMemEval temporal candidate all-hit | 96.0% | 86.0% |
| LoCoMo Cat1 multi-hop candidate all-hit | 60.0% | 82.0% |
| LoCoMo Cat2 temporal candidate all-hit | 96.0% | 96.0% |

## Paired bootstrap versus H0 (finalized gold, same graph)

| Profile | point | CI low | CI high | significant |
| --- | ---: | ---: | ---: | --- |
| h6 | +0.045 | +0.000 | +0.095 | **no** |

## Run provenance

| Run | source db | gold | run directory |
| --- | --- | --- | --- |
| `g0_draft` | `graphmem.sqlite` | `lme_gold_turn_merged_draft_20260804.jsonl` | `artifacts/v5_6/rebaseline/g0_draft/v5_5_retrieval200_20260805T145611Z` |
| `g0_final` | `graphmem.sqlite` | `longmemeval_v5_dev100_gold_turns.jsonl` | `artifacts/v5_6/rebaseline/g0_final/v5_5_retrieval200_20260805T145902Z` |
| `g2_draft` | `graphmem_g2.sqlite` | `lme_gold_turn_merged_draft_20260804.jsonl` | `artifacts/v5_6/rebaseline/g2_draft/v5_5_retrieval200_20260805T150154Z` |
| `g2_final` | `graphmem_g2.sqlite` | `longmemeval_v5_dev100_gold_turns.jsonl` | `artifacts/v5_6/rebaseline/g2_final/v5_5_retrieval200_20260805T150339Z` |

Every run reports `generative_llm_calls: 0`; only the permitted embedding
query channel was used.

