# Frozen development results — November 2026

All LoCoMo categories 1–4 (1,540 questions) and all 500 LongMemEval questions are
retained. Both phases use the same frozen Qwen30B judge and rubric per benchmark.

| Benchmark | Reader | Correct | Accuracy | Gain over archived baseline | All query tokens / question | Source-ID recall |
|---|---|---:|---:|---:|---:|---:|
| LoCoMo | Qwen30B | 1,455 / 1,540 | 94.48% | +22.08 pp | 9,680.88 | 99.1915% |
| LongMemEval | Qwen30B | 464 / 500 | 92.80% | +20.20 pp | 9,877.32 | 99.6809% |
| LoCoMo | Astra | 1,511 / 1,540 | 98.12% | +25.71 pp | 8,440.46 | 99.2567% |
| LongMemEval | Astra | 491 / 500 | 98.20% | +25.60 pp | 8,958.58 | 99.9468% |

The last accepted Qwen adjustment applies the latest-event reader to its entire
five-question family: LME improves from 463 to 464 correct. The accepted calendar
reader uses its model answer; the separately tested deterministic answer-rewrite
variant is **not** part of this result. All 1,609 LoCoMo and 509 LME Qwen calls,
including repairs/fallbacks, are counted. Astra uses one reader call per question.

Source-ID recall is the mean fraction of annotated original turns present in the
actual cumulative evidence set: 1,533 valid LoCoMo questions and 470 non-abstention
LME questions. Astra contains every annotated source for 1,499/1,533 LoCoMo and
469/470 LME questions. Empty, ambiguous or out-of-range annotations are not assigned
perfect recall.

## Construction and baseline scope

| Benchmark | Archived Mem0 build LLM tokens / history | GraphMem build LLM tokens / history | Saving |
|---|---:|---:|---:|
| LoCoMo | 5,318,817.9 | 86,660.5 | 98.37% |
| LongMemEval | 2,461,196.546 | 206,763.714 | 91.60% |

A history is one complete benchmark memory: one LoCoMo conversation or one LME
haystack. It can contain many sessions. These are frozen graph-construction LLM
ledgers, not a per-turn number and not total GPU/indexing cost. New raw/graph-view
embedding costs are recorded separately by the reproduction encoder.

The comparison baseline is the **old-material Mem0 top-200 archive chosen for this
study**, scored by the frozen common judge: 1,115/1,540 (72.40%) on LoCoMo and
363/500 (72.60%) on LME. Its recorded query LLM tokens are 13,515.10 and 11,761.066
respectively; archived embedding usage is unavailable. GraphMem's query numbers
include both embedding views and all reader calls. The archive's exact Mem0 update
mode was not verified, so it should not be advertised as a fully controlled native
Mem0 v2 reproduction. The later 85.97% Mem0 snapshot is a different baseline.

The LoCoMo archive used Qwen30B BF16 / temperature 0.1; the current stage used the
same pretrained family/version at FP8 / temperature 0. Do not describe this as
identical precision/decoding. Astra also changes reader instructions and some
source completion rules, so the phase difference is not a model-only causal test.
Tokenizers differ between Qwen and Astra; native-token totals are not monetary costs.

## Matched oracle diagnostic

With Astra and complete gold-bearing sessions, LME is 491/500 (98.2%), matching
the final GraphMem result. LoCoMo had three unresolved oracle transports; on the
same 1,537 successfully answered question IDs, both GraphMem and oracle have
1,508 correct (98.1132%). Do not divide the full 1,540-question numerator by 1,537.
The full oracle known-correct bound is 1,508–1,511 / 1,540; one failed empty output
was originally judged correct and is not a confirmed successful answer.
This is an empirical reader diagnostic, not a mathematical upper bound.

## Semantic required-information coverage

This separate diagnostic uses the frozen Qwen rubric, not a count of independent
gold facts. Final-source alignment reuses 1,949 unchanged contexts and verifies
54 expanded contexts. Both methods have paired judgments for all 2,003 eligible
questions, including unresolved rubric outputs.

| Benchmark | Usable pairs | GraphMem Astra, usable rubric units | Archived Mem0, usable rubric units |
|---|---:|---:|---:|
| LoCoMo | 1,262 / 1,533 | 99.6412–99.6729% | 51.4699–68.6648% |
| LongMemEval | 339 / 470 | 100% | 61.1357–77.3599% |

Keeping every unresolved rubric as [0,1], the full eligible-cohort bounds are
82.0268–99.7307% vs 42.3712–74.2042% on LoCoMo, with a paired improvement lower
bound of +7.8227 pp. On LME the bounds are 72.1277–100% vs 44.0957–83.6702%; the
paired difference spans −11.5426 to +55.9043 pp, so full-cohort superiority remains
inconclusive. Do not present the usable-pair 100% as coverage over all 500 questions.

These are development-selected benchmark results. Raw transcripts, responses,
private endpoints, authentication material and model downloads are excluded from
Git; aggregate metrics and artifact hashes preserve the result's provenance.
