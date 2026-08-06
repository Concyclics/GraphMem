# V5.6 — Enforcing 220K build tokens per memory

Target: **220,000 LLM tokens per memory** (one LongMemEval question's haystack, one LoCoMo conversation).

## What the frozen build actually costs

Three different numbers can be read off the frozen artifact, and only the third is the cost of the build we ship. Reporting the wrong one led to an earlier claim that the budget was blown 3.9×.

| Measurement | LME mean/memory | What it includes |
|---|---:|---|
| Every uncached call in the artifact | 854,471 | **42 graph versions** across ablation arms sharing one database |
| One pass, deduplicated by scene | 393,432 | still mixes arms: `legacy_batch` repair + LLM hierarchy compression |
| **Only `v5_4_navigable`'s own calls** | **275,261** | the config we actually ship |

The artifact holds `max(graph_version) = 42`. Arms that varied `semantic_max_facts_per_scene`, `summary_tokens` or `extraction_mode` produce different extraction cache keys, so the same scene was re-extracted once per arm — 2 scenes in one sampled memory appear in 89 distinct uncached calls. Those are ablation costs, not build costs.

Isolating calls made under the shipped config (strict JSON schema, `max_tokens=1024`, batches of ≤2):

| Benchmark | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| LongMemEval | **275,261** | 229,202 | 486,110 | 515,632 |
| LoCoMo | 260,318 | 346,390 | 387,297 | 387,844 |

Even that 275,261 still carries cross-arm residue. The only trustworthy number is a **cold build run on purpose**, which `scripts/measure_v5_6_build_budget.py` does into a throwaway database:

| config | mean | p50 | min | max | over 220K | fact coverage |
|---|---:|---:|---:|---:|---:|---:|
| `v5_4_navigable` (control, no ceiling) | 229,038 | 229,321 | 220,504 | 237,008 | **4/4** | 0.5560 |
| `v5_6_budget220k` (ceiling on, quote-free) | **196,605** | 199,285 | 186,713 | 201,138 | **0/4** | 0.5265 |

So the shipped build is **4% over target**, not 289% over. The earlier "3.9× blown" claim came from reading the ablation artifact as if it were a build ledger, and was wrong.

With the ceiling on, every memory lands under 220K with **no scenes skipped and no fallback scenes** — the ladder only ever reached step 2 (a reduced fact cap on 24–29 calls per memory). Builds also ran about 2× faster (51–58s vs 100–103s per memory), since output tokens dominate latency.

The cost is quality: semantic terminal turn coverage falls from 0.5560 to 0.5265, a **5.3% relative drop**. Whether that costs answer accuracy is **not yet measured** — it needs an answer + judge run against a budgeted graph, and until that exists the budgeted config should not be assumed free.

The 196,605 mean also shows the configuration overshoots: only a 4% cut was needed and it took 14%. `semantic_budget_degrade_at` is the dial — raising it toward 0.95 spends more of the ceiling before degrading and should recover most of the lost coverage while still landing under 220K.

Stage split (per LME memory, shipped config):

| stage | in | out | total |
|---|---:|---:|---:|
| `scene_semantic` | 177,079 | 84,973 | 262,052 |
| `scene_semantic_retry` | 8,979 | 4,230 | 13,209 |

Raw conversation text is ~122,285 tokens/memory, so the shipped build runs at **2.25× raw**. 220K would be 1.80×.

## Why nothing stopped it

`ModelConfig.semantic_average_tokens_per_memory` has declared `220000` since V5 and **has no consumer anywhere in `src/`** — the same condition `QueryBudget.max_answer_tokens` was in before V5.6 wired it. The budget existed as documentation.

## The enforcement

`graphmem.build.budget.BuildTokenLedger`, one per memory, shared across the extraction worker pool.

A hard refusal at the ceiling would silently drop the last sessions of every long memory — precisely the late-conversation facts multi-session questions need — so the policy is a ladder:

| spend | behaviour |
|---|---|
| < `degrade_at` (default 75%) | extract normally |
| ≥ `degrade_at` | keep extracting, **halve the per-scene fact cap** (cuts output, the larger half of cost, while still covering every scene) |
| would exceed ceiling | stop calling; emit deterministic scene summaries so later scenes stay in the graph and lose only their distilled facts |

Reservations are held across the fan-out, so 8 workers that each individually fit cannot collectively overrun. Every degradation is counted and written into the build manifest as `build_diagnostics.build_token_budget` — a silent degradation is indistinguishable from an unexplained accuracy drop.

## The structural levers, measured

| Lever | Saving/memory | Basis | Cost |
|---|---:|---|---|
| Drop the `q` exact-quote field | ~22,200 | 26.1% of extraction output bytes (24,805 facts sampled) | span precision within an already-cited turn |
| Batch 2 → 4 scenes/call | ~18,500 | 316-token fixed overhead × 117 calls, halved | more output per call, pushes on `semantic_batch_output_tokens` |

The `q` field only narrows the span **inside a turn that `r` already cites**. `semantic.py` already falls back to locating the fact *value* in the turn, which is the same rule the P5 projection arm applies, so dropping `q` moves span derivation from the model to a deterministic rule rather than losing it. When `q` is absent the evidence resolver now locates the value inside the cited turn instead of falling through to a whole-memory scan, which would have kept the fact but lost its grounding turn.

Batching beyond 2 requires the new `strict_batch` extraction mode, since `strict_pair` hard-codes 2. It is an ablation knob, not a default: larger batches press against the 1,024-token output cap, and output truncation is what generated the repair traffic in the first place.

## Defaults are unchanged

`semantic_max_tokens_per_memory` defaults to `0` (disabled) and `semantic_quote_evidence` to `True`, so every frozen artifact and cached extraction stays byte-identical. Enabling the budget changes `config_hash`, which is correct: a budgeted build is a different artifact and must not reuse a frozen one.

`configs/v5/v5_6_budget220k.json` turns it on.
