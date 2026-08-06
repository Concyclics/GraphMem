# V5.6 — Is the retrieval proxy measuring the right thing?

First judged answer accuracy ever produced by the V5 tree.

- Run: `artifacts/v5_6/answers/v5_6_answer_h9_baseline_20260806T021127Z`
- Retrieval: h9 (fact reservoir), dense channel on, `max_evidence_turns=32`
- Answer: `graphmem.answer`, prompt `f7a39726…`, temperature 0, thinking disabled
- Judges: **mem0 official prompts, both benchmarks**, judge model = local Qwen3-30B
  - LME: `evaluate_mem0_judge.py`, prompt sha256 `ba8cf60d…`
  - LoCoMo: `evaluate_memory_benchmarks_locomo_judge.py`, pinned commit `4b61c5d3…`, prompt sha256 `8ebac1ef…`
- Question set: the **hard-200** (judge-error selected). Not comparable to a full-set number.

## Headline

| Set | n | Accuracy |
|---|---:|---:|
| Overall hard-200 | 200 | **59.5%** |
| LongMemEval hard-100 | 100 | **56.0%** |
| LoCoMo hard-100 (Cat 1–2) | 100 | **63.0%** |

Prompt tokens per question: mean 4,384 / p50 5,771 / p95 6,441 / max 6,526 — **0 questions over the 10,000 budget**, counted with the backbone's own tokenizer.

## The proxy every prior conclusion rests on

`turn_all_hit` is the metric that ranked h0–h9 and rejected PR4b. Against judged correctness:

| Proxy | ρ | 95% CI | acc when true | acc when false | lift |
|---|---:|---|---:|---:|---:|
| **turn_all_hit** | **+0.230** | [+0.09, +0.37] | 0.711 (n=97) | 0.485 (n=103) | **+22.6pp** |
| graph_reachable_turn_recall | +0.219 | [+0.09, +0.35] | 0.686 | 0.495 | +19.1pp |
| session_all_hit | +0.156 | [+0.03, +0.29] | 0.643 | 0.474 | +17.0pp |
| turn_recall | +0.141 | [+0.01, +0.28] | 0.619 | 0.545 | +7.4pp |
| **certificate_complete** | **+0.103** | **[−0.04, +0.25]** | 0.633 | 0.528 | +10.5pp |
| turn_any_hit | +0.022 | [−0.11, +0.16] | 0.601 | 0.577 | +2.4pp |
| session_any_hit | −0.067 | [−0.19, +0.07] | 0.586 | 0.714 | −12.8pp |

**The proxy is real but weak.** ρ = 0.23 explains ~5% of the variance in correctness. The 7pp spread that separates h0/h6/h8/h9 (.505/.555/.495/.485) maps to roughly **1.6pp of answer accuracy** — inside the noise band. Harness rungs should not be ranked on `turn_all_hit` alone from here on.

**`certificate_complete` does not predict correctness.** Its CI crosses zero. It is a pre-pack flag over "the operand has at least one binding", so this is the expected result, and it is the case for PR8's post-pack certificate rather than against it — but the current field must not be reported as evidence quality.

## Two findings that reframe the plan

### 1. Retrieval reachability is already solved; ranking is the whole loss

**The candidate pool contained every gold turn for 200 of 200 questions.** Not one question fails because the evidence was unreachable. Every retrieval-side loss is in ranking and packing, downstream of a pool that already holds the answer.

### 2. Perfect retrieval does not reach 90% on this set

Of the 97 questions where **all** gold turns were packed, **28 (28.9%) were still judged wrong**. So an oracle packer — one that always achieves `turn_all_hit` — would score about **71%** here, not 90%. The remaining 29% is answer-side: ordering, arithmetic, aggregation, and abstention.

### 3. The temporal stratum does not respond to retrieval at all

| Stratum | n | acc when all gold packed | acc when not |
|---|---:|---:|---:|
| lme_multi_session | 50 | 0.654 (n=26) | 0.250 (n=24) |
| **lme_temporal** | 50 | **0.649 (n=37)** | **0.692 (n=13)** |
| locomo_multihop | 50 | 0.800 (n=5) | 0.556 (n=45) |
| locomo_temporal | 50 | 0.828 (n=29) | 0.476 (n=21) |

On `lme_temporal`, packing all the gold turns is worth **nothing** (65% vs 69%, sign reversed). The evidence is present and the answer is still wrong. This is consistent with the graph carrying 150 `temporal_before` edges corpus-wide (1.4 per memory): the model receives the right turns and cannot order them. Temporal accuracy is a **projection + algebra** problem, not a retrieval one.

`lme_multi_session` is the opposite — a 40pp gap — so it is the stratum where P-series packing work pays.

## Consequences for the remaining plan

1. **Judged accuracy becomes the primary metric.** `turn_all_hit` is demoted to a diagnostic and is only meaningful on `lme_multi_session`.
2. **P-series arms must be scored on judged accuracy**, not turn all-hit, or they will optimize a metric worth 1.6pp.
3. **PR5 (temporal/relational algebra) outranks further packing work**, reversing the plan's ordering. The temporal stratum is 50% of the hard set and is retrieval-insensitive.
4. **90% is not reachable by retrieval work alone on this set.** The oracle-packer ceiling is ~71%.

## Caveat that must travel with these numbers

The judge is **local Qwen3-30B**, while every historical artifact (V3.7's 89.0%/86.2%, V4.1's 72.6%) was judged by **gpt-5.4-mini**. Prompts are byte-identical to mem0's official versions (sha256-enforced on both benchmarks), so judge model is the only changed variable — but a 200-question cross-judge calibration is still required before any of these numbers is placed beside a historical one.

---

# Addendum — why the aggregation fix did not land

Step 1 built the three pieces the aggregation error class needs, and they work
in isolation:

* **P1 projection**: 51,496 `COLLECTION_MANIFEST` nodes and 54,766 `MEMBER_OF`
  edges where the frozen graph had **zero**. 49,477 are single-member and
  **49,641 (96.4%) were invisible** to the build's `>=2 rows and >=2 distinct
  values` rule — the frozen graph held 1,856 `collection_scope` nodes, which is
  exactly the remainder.
* **`ast_algebra.evaluate_ast`**: executes the compiled AST and emits
  `AnswerMember` rows with witnesses; 24 unit tests.
* **h10 wiring**: `NavigationResult.algebra` reaches the answer stage.

The end-to-end result is still `closed_form_rate = 0.0`, and that is now the
**correct** outcome rather than an accident.

## What blocks it

Operand-to-collection identification. For *"How many antique items did I inherit
from my family members?"* the compiled operand carries:

```
predicates: ('ask family members about clocks', 'notes that many people share',
             'plans a monthly family game night')
scopes:     ('antique clocks', 'family activities')
owners:     ('i',)
```

The predicate candidates are **retrieved from the graph by embedding
similarity, not parsed from the question** — nothing encodes *inherit*. Matching
manifests by term overlap against them therefore matches nearly every
collection, and the count ranged over the whole memory: it returned **15
"antique items"** that were actually `Shutterfly`, `grandmother`, `$70` and
similar, with `scope_complete = True`. Gold is 5.

Owner does not rescue it: `owners = ('i',)` means "every fact about the memory
user", which is the whole memory.

## What was changed as a result

A confidently wrong count is worse than no count, so:

* an operand is closed only when a manifest matches it on owner **and**
  predicate, and the count is then restricted to that manifest's members;
* owner alone no longer counts as a constraint;
* `compose()` **withholds an uncertified count entirely** rather than proposing
  "at least 15". A partial *list* still renders — its members are named rather
  than inferred from a scope claim — but a partial *count* is withheld.

## What this means for the plan

The aggregation fix is **blocked upstream of both the manifest and the
algebra**, in query parsing / operand construction. The next step is not more
algebra: it is making the operand carry the question's own predicate and scope
(*inherit*, *antique items*) so a collection can be identified at all.

That is a change to `retrieval/slots.py` and `_ast_operands`, and it is the
precondition for every aggregate operator. Until it lands, `COUNT_DISTINCT`,
`GROUP_BY_OWNER` and `EXISTS_ALL` cannot be trusted on this corpus regardless of
how good the manifests are.
