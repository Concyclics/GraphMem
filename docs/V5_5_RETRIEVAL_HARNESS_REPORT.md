# GraphMem V5.5 Retrieval Harness report

## Decision

V5.5 is an experimental retrieval improvement, not a replacement for the
frozen V5.4 navigator.  On the fixed 200-question graph it raises strict
turn-level all-hit from **49.5% to 55.5%**, and substantially improves LoCoMo
Cat1 candidate coverage (60% to 82%).  It does not meet the promotion gates:
the candidate-to-packed gap remains 30 percentage points and LongMemEval
candidate all-hit regresses because its bounded candidate pool is too narrow.

No V5.4 authority SQLite, graph, or frozen log was changed.  The Cat1 work is
an independent sidecar SQLite for ten residual memories.

## Implemented query-only harness

`GraphNavigator` now orchestrates a deterministic execution chain:

```text
question -> QueryIR -> multi-anchor seeds -> relation scheduler
         -> FactBinding/relation algebra -> certificate
         -> terminal provenance hydration -> proof-unit packing
```

The public domain includes query operators, operand specifications, proof
obligations, bindings, evidence units, certificate states, stop reasons and
per-stage trace fields.  `GraphReadView` compiles fact/owner/predicate/scope,
value, collection, session, and RoutingCard-posting lookups entirely in
memory.  The frozen source is opened read-only; only retrieval artifacts are
written.  Online query code neither imports nor reads answers, gold sessions,
gold turns, or category labels.

H0--H6 implement the requested progression: legacy parity/telemetry,
postings, per-operand RRF seeding, obligation scheduling, algebra plus real
certificate, and proof-unit packing.  Generated-model calls are **zero** for
all V5.5 runs; only the permitted embedding query channel is used and recorded
separately.

## Full-200 result

| Stratum (50 each) | H0 strict / candidate | H6 + G2 strict / candidate |
| --- | ---: | ---: |
| LongMemEval multi-session | 60% / 98% | 68% / 80% |
| LongMemEval temporal | 72% / 98% | 74% / 84% |
| LoCoMo Cat1 multi-hop | 6% / 60% | 18% / 82% |
| LoCoMo Cat2 temporal | 60% / 96% | 62% / 96% |
| **All 200** | **49.5% / 88.0%** | **55.5% / 85.5%** |

H0 parity was verified by an H0/H1 replay: both produce the frozen 49.5%
strict all-hit and 88.0% candidate all-hit result.  The final H6+G2 run has
68.9% mean turn recall, 91.7% candidate recall, 3,345 average evidence tokens,
54.6 visited nodes, and 25.4 visited edges.  A valid certificate stops 67% of
queries.  Its maximum 32-turn cap is hit on all Cat1 questions; the 5,000-token
cap is hit on 96% of both LongMemEval strata.

## G-series sidecar finding

The Cat1 condition was triggered after H6 (candidate all-hit below 75%).

- G1 adds `CollectionManifest` membership for all factual owner--predicate--
  scope chains in the ten Cat1-residual memories.  It does not materially move
  results by itself: certificate closure can still occur before all member
  witnesses are packed.
- G2 adds bounded, lossless owner-to-terminal postings for direct speaker
  evidence groups.  It improves Cat1 candidate all-hit from the H0 60% to 82%,
  demonstrating that the frozen index lacked a direct terminal path for many
  dialogue/list facts.  It does not copy raw text or call a model.

The remaining failure is chiefly packing, not semantic reachability.  For
example, after G2 the two gold turns for `locomo00_0019` are candidates, but
are displaced by many same-owner rescue turns.  Cat1's 82% candidate versus
18% packed all-hit is the clearest form of this issue.

## Gate assessment and next action

| Gate | Status | Evidence |
| --- | --- | --- |
| H0 frozen parity | Pass | H0 and H1 are identical. |
| Online generative LLM token | Pass | 0 calls. |
| Telemetry separates stopping causes | Pass | search/pack/certificate fields in every trace. |
| Cat1 candidate condition | Pass after G2 | 82%. |
| Candidate-to-packed gap <=20pp | Fail | 30pp aggregate; 64pp Cat1. |
| LME candidate no regression | Fail | 80%/84% versus H0 98%/98%. |
| Promote over V5.4 | **Do not promote** | Quality gates above are unmet. |

The next safe iteration is a packer-only H7: make each collection member and
each temporal/state endpoint an atomic mandatory proof unit, partition the
32-turn budget across unsatisfied operands/sessions, and score direct terminal
postings using local clause/keyphrase overlap before owner-level rescue.  It
should retain G2's direct terminal postings, but not rebuild V5.4.  Only if
candidate coverage remains insufficient after that should a graph rebuild be
considered.

## Reproducible artifacts

- Frozen H0/H1--H6 first run: `artifacts/v5_5/retrieval200/v5_5_retrieval200_20260805T121717Z`
- Final 200-question H6+G2 run: `artifacts/v5_5/final3_h6_g2/v5_5_retrieval200_20260805T125416Z`
- G2 independent SQLite and manifest: `artifacts/v5_5/cat1_g2_20260805/`

Each result directory contains the run manifest, per-question navigation
JSONL, metrics CSV/Parquet, token ledger, error cases, traces, and HTML report.
