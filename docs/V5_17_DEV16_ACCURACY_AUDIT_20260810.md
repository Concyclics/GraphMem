# V5.17 16-memory accuracy and quality audit (2026-08-10)

## Scope

This audit uses a fresh V5.17 build of all 16 LongMemEval memories. The graph
was not reused from an older construction implementation.

- Graph: `artifacts/report/v5_17/budget230_retry_optimized_dev16_v3_20260810/graph/graphmem.sqlite`
- Build report: `artifacts/report/v5_17/budget230_retry_optimized_dev16_v3_20260810/build_report.json`
- Original answer arm: `artifacts/report/v5_17/accuracy_dev16_v3_20260810/current`
- Wider answer arm: `artifacts/report/v5_17/accuracy_dev16_v3_20260810/expanded64`

The answer model output cap is disabled in both arms. The changed budget is the
retrieved evidence budget, not the completion length.

## Result

| Arm | Evidence budget | Local Qwen judge | Strict answer audit |
|---|---:|---:|---:|
| Adaptive precision arm | 12/24 turns, 5K tokens | 11/16 (68.75%) | 9/16 (56.25%) |
| Wider evidence arm | 64 turns, 12K tokens | 13/16 (81.25%) | 12/16 (75.00%) |

The strict score overrides two demonstrably invalid positive judgements in the
adaptive arm: `07b6f563` and `07741c45` both abstained even though their gold
answers were answerable. In the wider arm, `06878be2` has the same false
positive pattern. Therefore the local judge score must not be reported without
the deterministic rule: a refusal is wrong whenever the benchmark row is not
an abstention row.

## Per-question strict result for the wider arm

| Question | Type | Result | Main observation |
|---|---|---:|---|
| `0862e8bf` | single-session | correct | Cat name recovered. |
| `001be529` | single-session | correct | Duration recovered. |
| `0862e8bf_abs` | abstention | correct | Hamster near-match rejected. |
| `00ca467f` | multi-session count | wrong | Both March 3 and March 20 turns are packed, but the answer counts only one. |
| `06878be2` | preference | wrong | The answer refuses despite relevant Sony/photography context in the source. |
| `06f04340` | preference | correct | Homegrown basil and mint are used in the recommendation. |
| `07b6f563` | preference | correct | iPhone screen protector, wallet case, and power-bank preferences recovered. |
| `0100672e` | multi-session arithmetic | correct | Computes 60 / 5 = 12. |
| `078150f1` | multi-session arithmetic | correct | Computes 250 - 200 = 50. |
| `07741c44` | knowledge update | correct | Initial `under my bed` state recovered. |
| `031748ae` | knowledge update | correct | Initial 4 and current 5 engineers both recovered. |
| `08e075c7` | knowledge update | correct | Latest 9-month state recovered. |
| `01493427` | knowledge update | correct | Latest total 25 recovered. |
| `06db6396` | knowledge update | correct | Latest fifth project recovered. |
| `031748ae_abs` | entity/role abstention | wrong | `Software Engineer Manager` is conflated with `Senior Software Engineer`. |
| `07741c45` | projected state | wrong | The old `under bed` state wins over the later shoe-rack plan. |

## Source-to-answer failure localization

The original adaptive arm produced a mean 454 candidate turns and packed only
18.6. All 16 queries ended with `budget_exhausted`; most also reached node,
frontier, or hop caps. This is a late-pruning system, not yet a precise graph
router.

For six non-abstention failures, exact gold-turn coverage changed as follows:

| Budget arm | Questions with any gold turn | Gold turns packed | Mean packed turns | Mean evidence tokens |
|---|---:|---:|---:|---:|
| Adaptive 12/24 | 1/6 | 2/11 | 18 | 2,009 |
| 32 turns | 4/6 | 6/11 | 32 | 3,312 |
| 48 turns | 5/6 | 7/11 | 48 | 4,937 |
| 64 turns | 6/6 | 10/11 | 64 | 6,756 |

The wider arm raises mean complete prompt length to 9,478 tokens (max 9,905),
with zero output truncation. It remains well inside the 65,536-token serving
context and does not approach the 400--500-turn candidate reservoir.

## Remaining quality blockers

1. **Fact binding and algebra do not close.** `00ca467f` compiles to
   `CountDistinct`, but no bindings are produced, so deterministic counting is
   not used even though both witnesses are packed.
2. **Exact discriminants are not enforced.** `031748ae_abs` does not require an
   exact role-title match before answering.
3. **Planned state is not projected to the question time.** `07741c45` stores
   the shoe-rack statement with planned modality, but the question is later
   than the planned weekend.
4. **Evidence packing is not monotone.** Increasing capacity can replace a
   previously useful span with other proof units. This explains the wider
   arm's regression on `06878be2`; a larger budget should preserve the smaller
   arm's relevant floor.
5. **Candidate precision remains weak.** The reservoir is near the whole
   memory and graph traversal repeatedly exhausts caps. Relation-aware routing
   should improve before using still larger evidence budgets.

## Promoted accuracy profile

`configs/v5/runtime_v5_17_accuracy64.json` records the measured accuracy
operating point: 64 evidence turns, a 12K evidence-token cap, obligation-aware
packing, no 12/24-turn adaptive precision cap, QueryIR soft fallback, and rare
lexical relations enabled. The frozen 230K build configuration is unchanged.

This profile is an interim accuracy operating point. A release-quality gate
still requires the blockers above, a strict judge guard, and a larger held-out
benchmark run.
