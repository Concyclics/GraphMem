# V5.6-prebench progress

One page per landed PR, measured on the fixed-200 development set against the
**finalized** LongMemEval annotations (`eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl`)
and the frozen V5.4 authority graph recorded in `V5_6_FREEZE.md`.

Corrected baselines from `V5_6_REBASELINE.md`:

| | strict turn all-hit | candidate all-hit |
| --- | ---: | ---: |
| A0 = V5.4 H0 (frozen navigator) | 50.5% | 88.0% |
| V5.5 H6 on the same graph | 55.0% | 60.0% |
| V5.5 H6 + G2 sidecar (gold-scoped) | 57.0% | 85.5% |

Every number below is `PYTHONHASHSEED=0`, `generative_llm_calls: 0`, embedding
query channel only.

---

## PR0 — re-baseline, freeze, determinism

**What it fixed**

| Defect | Symptom | Fix |
| --- | --- | --- |
| D0 | Every V5.4/V5.5 run scored against the pre-adjudication draft annotations; only 142 of 217 turn references were shared with the finalized set | Re-ran the 2×2 gold × graph matrix; `V5_6_REBASELINE.md` is now the baseline of record |
| D1 | `packer.pack` iterated a `set` of mandatory turn ids, so packed order and (under the turn cap) packed membership varied with `PYTHONHASHSEED` | Mandatory turns keep their declared proof-unit order |
| D2 | `FactBinding.time_interval` held `str(dict)`; canonical JSON sorts keys, so `ARGMIN/ARGMAX/ORDINAL` ranked events by `anchor_turn_id` hash | Typed `TemporalKey` with an explicit `sort_key` |
| D3 | `lookup_facts` truncated to 24 by lexicographic node id | Rank by owner/predicate/lexical relevance, then cut |
| D9/D10 | `destination_priority` evaluated twice per comparison; per-turn linear scans of the whole turn list | Score once; index turns by id and session |
| D11 | `str(attrs.get(key)) or None` produced the literal string `"None"` | `_text()` helper returning a real `None` |
| D12 | `estimate_tokens` was `words × 1.3`, so no token budget was ever actually enforced | `graphmem/tokenization.py` loads the backbone's own `tokenizer.json` offline; `require_exact=True` refuses to guess |

**Verification**

- 968 tests pass, including a cross-process `PYTHONHASHSEED=0/1` byte-identity
  check that asserts its own fixture produces more than one proof unit.
- **H0 parity is exact**: 0.505 strict / 0.880 candidate / 2011.3 evidence tokens,
  unchanged before and after. The legacy path deliberately keeps the historical
  word-count estimate so it remains a faithful frozen baseline; only the harness
  budgets against real tokens.
- H6 moved 0.550 → 0.555 strict once its budget was enforced in real tokens.

**Measured along the way**: 20.4% of harness proof steps are reached via an
inverse edge, so the missing `ProofStep.inverse` flag was misreporting direction
on a fifth of all steps. It is now recorded.

---

## PR1 (H8) — safe-superset candidate reservoir

Run: `artifacts/v5_6/pr1_final/v5_5_retrieval200_20260805T155045Z`

### Result

| Stratum | H0 strict / cand | H6 strict / cand | **H8 strict / cand** |
| --- | ---: | ---: | ---: |
| LongMemEval multi-session | 58.0% / 100.0% | 70.0% / 78.0% | 56.0% / **100.0%** |
| LongMemEval temporal | 78.0% / 96.0% | 80.0% / 86.0% | 78.0% / **100.0%** |
| LoCoMo Cat1 multi-hop | 6.0% / 60.0% | 14.0% / 16.0% | 8.0% / **100.0%** |
| LoCoMo Cat2 temporal | 60.0% / 96.0% | 58.0% / 60.0% | 56.0% / **100.0%** |
| **All 200** | 50.5% / 88.0% | 55.5% / 60.0% | 49.5% / **100.0%** |

Coverage against the frozen navigator, per question:

| | H6 | H8 |
| --- | ---: | ---: |
| pool is a superset of H0's | 0/200 | **188/200** |
| H0 turns missing, per question | 235.9 | **0.09** |
| candidate all-hit regressions vs H0 | 56 | **0** |
| mean pool size | 45.3 | 544.8 |

**LoCoMo Cat1 candidate all-hit reaches 100% with no sidecar at all.** The V5.5
report got 82% there only via G2, whose build scope was selected from a prior
run's `candidate_turn_all_hit`, i.e. from gold. That leakage is now unnecessary.

### What was actually wrong

1. **Pool membership was derived from a normalized score.** Min-max normalization
   maps each channel's weakest hit to exactly 0.0, so a truthiness test silently
   discarded the tail of every channel. Membership now comes from presence.
   This one defect accounted for most of the missing turns.
2. **Session flooding could not be reproduced by imitation.** The legacy navigator
   floods its eight strongest sessions using its own score scale and its own
   tokenizer — `text.terms()` strips a possessive `'s`, `navigator.terms()` does
   not. Rather than chase that, the reservoir floods *every session holding a
   channel hit*, which dominates the legacy choice by construction.
3. **`lookup_facts` truncated to 24 by lexicographic node id**, discarding facts
   for reasons unrelated to the query.
4. **The shared 48-node cap was consumed in operand order**, so operand 2 of an
   intersection could be starved outright. Node budgets are now per-operand and
   round-robin merged.
5. **A wider pool has to be ranked better, not just be wider.** With the V5.5
   fused score, going from 45 to 545 candidates *lowered* strict all-hit: the
   scorer lacked the lexical, session and adjacency terms the legacy scorer used
   to pick 16 useful turns out of 278. Those terms are restored for the wide
   profile and gated off for H2–H6.

### What it does not fix

Strict packed all-hit is **49.5%**, against H0's 50.5% (paired delta −0.010,
CI [−0.055, +0.030] — indistinguishable). Every gold turn is now reachable on
100% of questions, but the packer still selects by fused score rather than by
answer membership, so coverage does not convert into packed evidence. That
conversion is PR7's job, and the candidate→packed gap is now the single largest
remaining loss: **50.5 percentage points**.

The residual 0.09 missing turns/question are reachable only through the legacy
path's own graph traversal, not through seeding; they are a scheduler concern
(PR5), not a reservoir one.

### Ladder integrity

H2–H6 keep the frozen V5.5 seeding (`_seed_narrow`) and the V5.5 fused score, so
H6 still reports 55.5% / 60.0% and every later delta stays attributable. A
regression test asserts the narrow path stays within the V5.5 caps and remains a
subset of the wide one.

---

## PR2b — operator AST compiler, shadow mode

`operators.py` (PR2a) defined the algebra but nothing produced it: `query_ir.py`
still chose a single enum by racing substring tests. This wires a compiler in
front of it — parse the slots, then compose — while leaving execution alone.

`QueryIR.operator` / `.operands` remain exactly the V5.5 values that the
reservoir, scheduler and packer consume. `.ast`, `.ast_operands`,
`.ast_obligations`, `.slots` and `.parse_warnings` are compiled alongside and
written to the trace. A later profile switches execution over; until then the
divergence can be measured before anything depends on it.

### Verified inert

`scripts/compare_v5_6_runs.py` compares two runs on execution-relevant fields
only (packed/dropped/retrieved turns, proof units, certificate, candidate
scores, coverage), ignoring trace keys:

```
h0: IDENTICAL   h6: IDENTICAL   h8: IDENTICAL   →  OK: execution unchanged
```

### What the compiler found

Measured on the fixed 200 (h8):

| | |
| --- | ---: |
| questions where the AST disagrees with the legacy classifier | **48 / 200 (24.0%)** |
| questions whose AST is compositional (nested operators) | **41 / 200** |
| strict all-hit where the two agree | 51.3% |
| strict all-hit where they disagree | **43.8%** |

The disagreements concentrate on the harder questions, which is what you would
expect if the classifier is wrong precisely where the question is complex.

Largest disagreement classes:

| count | legacy | AST |
| ---: | --- | --- |
| 14 | `lookup` | `union_distinct` |
| 12 | `union_distinct` | `lookup` |
| 7 | `lookup` | `count_distinct` |
| 5 | `count_distinct` | **`date_difference`** |
| 3 | `argmax_time` | `argmin_time` |
| 2 | `latest_state` | `ordinal` |

The `count_distinct → date_difference` class is the clearest: the legacy cascade
tests `"how many"` before `"how long"`, so *"how many days between X and Y"*
became a count. `latest_state` drops to zero because `"now"` no longer matches
inside `"know"` and because `"last"`/`"latest"` are ordinals over time, not state
lookups.

### Incidental finding: the dense channel is not reproducible across runs

The first byte-identity attempt failed on **h0**, which PR2b cannot touch. The
cause is the embedding service: `dense_score` drifts by ~3e-3 between runs
(0.90817 → 0.90478), which reorders the tail of the candidate list. Within a
single process it is stable — three successive calls return bit-identical
vectors — so this is cross-run batching/GPU nondeterminism, not a code defect.

Impact on outcomes is negligible: across h0/h6/h8 every headline metric is
identical between the two runs except h6 `evidence_tokens` (3171.17 → 3171.40,
one turn swapped). But it means **cross-run byte-identity is unachievable while
dense retrieval is on**, so the inertness proof above was run with the dense
channel disabled, where all channels are deterministic.

Worth fixing before the frozen full benchmark: persist query embeddings in a
sidecar cache keyed by `(model_id, query_text)` so a frozen run can be replayed
exactly. Until then, any two runs differ by this noise floor.

---

## PR3 — proof funnel and packing oracles

Run: `artifacts/v5_6/funnel2/`, profile h8, full 200, finalized gold.

Each gold turn is walked through every stage it has to survive. A question
counts only if *all* of its gold turns clear the stage.

| stage | h8 |
| --- | ---: |
| R0 gold turn is in the reservoir | 100.0% |
| R1 …scored as a candidate | 100.0% |
| R2 …an EvidenceGroup resolves to it | 100.0% |
| R3 …a CanonicalFact cites that group | **68.5%** |
| R3b …**that fact was actually reached as a node** | **21.0%** |
| R4 …an operand bound to it | 13.5% |
| R5 …the algebra kept the binding | 13.5% |
| R6 …it became a proof unit | 13.5% |
| R7 …it was packed | 49.5% |

Where each question *first* loses a gold turn:

| first loss stage | questions |
| --- | ---: |
| **the fact was never reached as a node** | **95 / 200** |
| no CanonicalFact exists for the turn at all | 63 / 200 |
| reached but did not bind | 15 / 200 |
| lost only at packing | 2 / 200 |
| clears every stage | 25 / 200 |

By stratum (h8):

| stratum | has_canonical_fact | fact_reached | has_binding | packed |
| --- | ---: | ---: | ---: | ---: |
| LongMemEval multi-session | 68% | 12% | 6% | 56% |
| LongMemEval temporal | 66% | 26% | 8% | 78% |
| LoCoMo Cat1 | 54% | 6% | 4% | 8% |
| LoCoMo Cat2 | 86% | 40% | 36% | 56% |

### Three conclusions, all of which redirect the plan

**1. Packing is not the bottleneck — node retrieval is.** Packing is the first
loss on 2 of 200 questions. The dominant cliff is R3 → R3b: the fact exists in
the graph on 68.5% of questions but is reached on only 21.0%. The mechanism is
the exact turn-side problem PR1 fixed, one level up: memories hold **450–500
CanonicalFacts** while the node budget is `max_visited_nodes = 96` with a
per-operand `lookup_facts` limit of 48. The right fact is simply never
retrieved.

**2. Proof-driven packing cannot be turned on yet.** `gold_packed` (49.5%) is
almost four times `gold_in_proof_unit` (13.5%): today's strict score is carried
by fused-score *fill*, not by proof. Switching to strict proof-bundle packing
before fixing reachability would drop strict all-hit toward 13.5%. Proof
packing depends on binding coverage, which depends on fact reachability.

**3. `budget_oracle_fits` is 100%.** Every question's gold turns fit inside 32
turns and 5,000 real tokens. Budget is never the binding constraint, so the
32-turn cap and the 5,000-token cap are not what is costing strict all-hit.

### Revised order

```text
PR4a  fact reservoir  — extend the safe-superset principle to the node side
PR4b  binding discriminant — the 15/200 that reach but do not bind
PR7a  active shortlist   (now safe: proof coverage will be high)
PR7b  proof bundles      (ditto)
PR8   post-pack certificate
Track B extraction — the 31.5% of questions with no CanonicalFact at all
```

The 68.5% canonical-fact ceiling is the answer to "do we need to rebuild the
graph?": **eventually yes**. No retrieval change can lift proof-based strict
all-hit above 68.5% on this graph, because on 31.5% of questions at least one
gold turn has no fact citing it. That is now a measurement rather than a guess,
and it sizes Track B precisely.

---

## PR4a (h9) — semantic fact reservoir

The reservoir principle from PR1, applied one level up: a wide id-only pool of
CanonicalFacts, then a bounded active shortlist that is what actually enters
binding, scheduling and algebra. Widening alone would only move the noise
downstream, so the two layers are separate and separately measured.

Channels, per operand: **reverse projection** (turn → evidence group → citing
facts), composite owner/predicate postings, RoutingCard child postings, and a
fact lexical index over a compact per-fact surface. Dense fact retrieval is
behind an ablation flag and off by default. Nothing hydrates source text.

### Reachability moved; the end metric did not

Measured with the funnel (20-question probe, conditional on a CanonicalFact
existing so the 68.5% graph ceiling does not mask the retrieval question):

| | h8 | h9 |
| --- | ---: | ---: |
| gold fact **in reservoir** \| fact exists | — | **90.3%** |
| gold fact **in active shortlist** \| fact exists | 40.0% | **75.9%** |
| gold_fact_reached (unconditional) | 16.7% | 35.0% |
| gold_has_operand_binding | 8.3% | **10.0%** |

Full 200, strict / candidate:

| Stratum | H0 | H6 | H8 | **H9** |
| --- | ---: | ---: | ---: | ---: |
| LongMemEval multi-session | .580 / 1.000 | .700 / .780 | .560 / 1.000 | .520 / 1.000 |
| LongMemEval temporal | .780 / .960 | .800 / .860 | .780 / 1.000 | .740 / 1.000 |
| LoCoMo Cat1 | .060 / .600 | .140 / .160 | .080 / 1.000 | **.100** / 1.000 |
| LoCoMo Cat2 | .600 / .960 | .580 / .600 | .560 / 1.000 | .580 / 1.000 |
| **All 200** | .505 / .880 | .555 / .600 | .495 / 1.000 | **.485** / 1.000 |

Paired vs H0: h8 −0.010 [−0.055, +0.030], h9 −0.020 [−0.075, +0.040]. h9 and h8
are indistinguishable on strict; the source-turn reservoir stays at 100%
candidate all-hit with 0 regressions.

### Why the gain is invisible, and what it implies

`gold_has_operand_binding` moved only 8.3% → 10.0%. Facts are now reached but
still do not bind, so they produce no proof units, so strict all-hit cannot
move. **PR4a's payoff is gated behind PR4b**, exactly the dependency the funnel
predicted, and it is the reason strict is flat rather than up.

The one place it already shows is LoCoMo Cat1 (.080 → .100), the stratum whose
facts were least reachable.

### Gates: partially met

| gate | target | actual |
| --- | ---: | ---: |
| source-turn reservoir candidate all-hit | 100% | **100%** ✅ |
| conditional gold-fact reservoir recall | ≥95% | 90.3% ⚠️ |
| conditional gold-fact active recall | 75–85% | 75.9% ✅ (low end) |
| overall reached-fact all-hit | 55–60% | 35% ❌ |
| no historical profile regression | — | h0/h6/h8 unchanged ✅ |
| mean active facts | ≤96 | ~32 ✅ |
| reservoir hydrates raw text | 0 | 0 ✅ |

The overall reached-fact target is not met, but it is bounded above by the
68.5% existence ceiling: 35% unconditional against a 55% ceiling on that probe
is 75.9% conditional. The honest reading is that the reservoir is close to
target and the shortlist is at the low end of target, while the unconditional
figure is held down by the graph, not by retrieval.

Next: PR4b (binding discriminant), which is now the sole thing standing between
reached facts and proof units.
