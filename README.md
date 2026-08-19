# GraphMem

GraphMem is a graph-structured long-term memory system for AI agents. It turns
multi-session conversations into a sparse hierarchical graph, compiles a user
question into a typed QueryIR plan, and retrieves a bounded evidence set for
answer generation.

The current V5.63 implementation focuses on three system goals:

- **Token-efficient construction.** HNSW-assisted coarsening, reconnecting and
  parent-gated refinement replace all-pairs relation induction. Lossless atomic
  extraction preserves dates, durations, quantities, negation and collection
  items, while an exact response cache and recoverable publication keep the
  510-Memory build inside the 230K generative-token gate with zero SDK retries.
  On the full corpus, the measured coarse-candidate exponent is `0.89` and the
  final relation set is `99.98%` smaller than the all-pairs upper bound.
- **Low-latency retrieval.** Per-memory FAISS indexes, compiled graph views,
  dual-lane coarse-to-fine traversal and deterministic QueryIR execution avoid
  an LLM call on the query path. Operator-aware pruning keeps relevance and
  proof/witness candidates separate until evidence packing.
- **High-precision readout.** Source-time normalization, topological evidence
  blocks and typed aggregation are augmented only for high-confidence temporal,
  preference and arithmetic routes. The V5.63 selector is label-free, never
  reads predictions or judge verdicts, and permits at most 500 additional prompt
  tokens for a selected request.

The graph is stored as a versioned SQLite snapshot. The core package is under
`src/graphmem`; benchmark, audit and reproduction entrypoints are under
`scripts`.

## Qwen3-30B benchmark results

We evaluate the full [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval)
set (500 questions over 500 memories) and LoCoMo
[Category 1–4](https://github.com/snap-research/locomo) (1,540 questions over
10 conversations). LoCoMo Category 5 is excluded. GraphMem and the archived
Mem0 baseline use Qwen3-30B; all answers are judged by the pinned
`gpt-5.6-luna` prompts. Accuracy is end-to-end answer accuracy, not retrieval
recall.

| Benchmark | System | Retrieval budget | Accuracy | Mean build Token | Mean answer Token |
|---|---|---:|---:|---:|---:|
| LongMemEval-S | GraphMem V5.63 | 32 turns | **75.40%** | **189,297** | 6,191 |
| LongMemEval-S | Mem0 baseline | top-50 | 56.80% | 2,461,197 | **5,202** |
| LongMemEval-S | GraphMem V5.63 | 64 turns | **79.20%** | **189,297** | **9,199** |
| LongMemEval-S | Mem0 baseline | top-200 | 57.60% | 2,461,197 | 11,761 |
| LoCoMo Cat. 1–4 | GraphMem V5.63 | 32 turns | **83.70%** | **75,243** | **3,220** |
| LoCoMo Cat. 1–4 | Mem0 baseline | top-50 | 67.66% | 5,318,818 | 5,514 |
| LoCoMo Cat. 1–4 | GraphMem V5.63 | 64 turns | **86.23%** | **75,243** | **5,724** |
| LoCoMo Cat. 1–4 | Mem0 baseline | top-200 | 68.31% | 5,318,818 | 13,515 |

The 32-turn/top-50 and 64-turn/top-200 rows are approximate retrieval-budget
operating points, not claims that their prompts contain exactly the same number
of tokens.

| Matched comparison | Accuracy gain | Build Token saving | Answer Token saving |
|---|---:|---:|---:|
| LongMemEval-S: 32 turns vs top-50 | **+18.60 pp (+32.75%)** | **92.31%** | -19.01% |
| LongMemEval-S: 64 turns vs top-200 | **+21.60 pp (+37.50%)** | **92.31%** | **21.78%** |
| LoCoMo: 32 turns vs top-50 | **+16.04 pp (+23.70%)** | **98.59%** | **41.60%** |
| LoCoMo: 64 turns vs top-200 | **+17.92 pp (+26.24%)** | **98.59%** | **57.65%** |

The percentage in parentheses is the relative accuracy uplift over the matched
Mem0 baseline; `pp` is the absolute percentage-point gain.
`-19.01%` means the LongMemEval-S 32-turn GraphMem prompt costs 19.01% more
answer Token than Mem0 top-50; it is retained to show the complete comparison.
Build Token means are per LongMemEval memory and per LoCoMo conversation.
Answer Token means are per question. Build values count generative API input +
output tokens and exclude embeddings and judge calls; one GraphMem build is
shared by its 32-turn and 64-turn query runs. Answer values count prompt +
completion tokens with a 2,000-token completion ceiling.

The exact p95/p99/max values, per-type accuracy and verdict hashes are generated
by [`scripts/build_v5_63_report_accuracy.py`](scripts/build_v5_63_report_accuracy.py).
The frozen V5.63 report manifest has SHA-256
`57c5e01d9cc4352e862c0343d5819281d4008bbca3289f2a184415e23e9caa11`.
The previous V5.54 result contract remains available as a historical snapshot
in [`docs/results/v5_54_qwen30b.json`](docs/results/v5_54_qwen30b.json).

## Reproduce V5.63

### 1. Install

Python 3.11 or newer is required.

```bash
git clone https://github.com/Concyclics/GraphMem.git
cd GraphMem
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,dense,experiment]"
cp .env.example .env
```

Benchmark datasets are not redistributed. Prepare the official datasets as:

```text
<workspace>/GraphMem/                         # this repository
<workspace>/artifacts/data/longmemeval_s_cleaned.json
<workspace>/artifacts/data/locomo10_graphmem.json
<workspace>/third_party/memory-benchmarks/    # Mem0 judge prompts
```

The LoCoMo conversion must preserve conversation, session, speaker, timestamp
and Category 1–4 question IDs. Gold turn annotations used only for retrieval
auditing are versioned in `eval_annotations/`; they are never passed to the
retriever or answer model.

### 2. Start local model endpoints

Expose OpenAI-compatible endpoints with these frozen models:

```text
http://127.0.0.1:8001/v1  Qwen/Qwen3-Embedding-0.6B
http://127.0.0.1:8002/v1  Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
```

The managed local workflow can be checked with:

```bash
scripts/smoke_longmemeval.sh up
scripts/smoke_longmemeval.sh run
```

### 3. Rebuild all 510 memories

```bash
mkdir -p ../artifacts/report/v5_57/full/graph

python scripts/run_v5_6_full_build.py \
  --target-db ../artifacts/report/v5_57/full/graph/graphmem.sqlite \
  --relation-embedding-db ../artifacts/report/v5_57/full/graph/relation_embeddings.sqlite \
  --lme ../artifacts/data/longmemeval_s_cleaned.json \
  --locomo ../artifacts/data/locomo10_graphmem.json \
  --gold eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl \
  --config configs/v5/v5_57_lossless_atomic.json --profile b5 \
  --embedding --embedding-request-model Qwen3-Embedding-0.6B \
  --memory-workers 8 --max-concurrency 256 \
  --require-zero-retries --require-complete-diagnostics \
  --report ../artifacts/report/v5_57/full/build_report.json

python scripts/precompile_dense_indexes.py \
  --db ../artifacts/report/v5_57/full/graph/graphmem.sqlite \
  --config configs/v5/v5_57_lossless_atomic.json \
  --output ../artifacts/report/v5_57/full/dense_indexes \
  --backend faiss_flat --workers 16
```

V5.63 shares this single V5.57 lossless graph build across both query budgets.
The runner is resumable by graph version and exact LLM-call cache. A valid full
build has 510/510 published memories, zero SDK retries, complete diagnostics
and no memory above the configured 230K generative-token gate. The convenience
runner [`scripts/run_v5_57_full_benchmark.sh`](scripts/run_v5_57_full_benchmark.sh)
automates build, FAISS precompilation and the frozen pre-V5.63 answer arms.

### 4. Run the 32-turn and 64-turn answer arms

```bash
for budget in 32 64; do
  if [ "$budget" = 32 ]; then
    runtime=configs/v5/runtime_v5_63_selective32.json
  else
    runtime=configs/v5/runtime_v5_59_hybrid64.json
  fi

  python scripts/run_v5_6_answer.py \
    --source-db ../artifacts/report/v5_57/full/graph/graphmem.sqlite \
    --relation-embedding-db ../artifacts/report/v5_57/full/graph/relation_embeddings.sqlite \
    --output-root ../artifacts/repro/v5_63/turn${budget} \
    --run-root ../artifacts/repro/v5_63/turn${budget}/answer \
    --lme ../artifacts/data/longmemeval_s_cleaned.json \
    --locomo ../artifacts/data/locomo10_graphmem.json \
    --gold eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl \
    --config configs/v5/v5_57_lossless_atomic.json \
    --runtime-config "$runtime" --answer-policy v5_63 \
    --embedding --embedding-request-model Qwen3-Embedding-0.6B --full \
    --answer-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --answer-base-url http://127.0.0.1:8002/v1 \
    --answer-request-profile qwen --max-output-tokens 2000 \
    --answer-workers 256 --navigate-workers 32 \
    --label v563_qwen30b_turn${budget} --resume
done
```

Runtime JSON is the authority for the evidence budget, QueryIR, hierarchy,
FAISS and traversal settings. The 32-turn profile is frozen directly as V5.63;
the 64-turn profile retains the validated V5.59 retrieval layout and applies
the V5.63 selective readout in core code. The archived 64-turn paper artifact
was also independently composed from frozen full-corpus arms with
[`scripts/materialize_v5_63_selective_accuracy.py`](scripts/materialize_v5_63_selective_accuracy.py).
That audit changed 75/2,040 prompts, used no prediction, gold label, category or
judge result for routing, and added only 2.51 mean prompt tokens over all
questions. The end-to-end 32-turn answer and paired-judge workflow is available
as [`scripts/run_v5_63_selective32_full.sh`](scripts/run_v5_63_selective32_full.sh).

### 5. Judge and audit

With `SGAO_API_KEY` and `SGAO_BASE_URL` set in the ignored `.env`, judge each
arm using the pinned prompts:

```bash
python scripts/evaluate_mem0_judge.py \
  --answers ../artifacts/repro/v5_63/turn64/answer/answers.jsonl \
  --output-dir ../artifacts/repro/v5_63/turn64/answer/judge_longmemeval \
  --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
  --request-profile openai --workers 32 --resume

python scripts/evaluate_memory_benchmarks_locomo_judge.py \
  --data ../artifacts/data/locomo10_graphmem.json \
  --answers ../artifacts/repro/v5_63/turn64/answer/answers.jsonl \
  --output-dir ../artifacts/repro/v5_63/turn64/answer/judge_locomo \
  --memory-benchmarks-repo ../third_party/memory-benchmarks \
  --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
  --request-profile openai --workers 32 --resume
```

Repeat for `turn32`. Percentiles use nearest rank, `ceil(p * N) - 1`.
Generate the final report contract only from completed, audited artifacts:

```bash
python scripts/build_v5_63_report_accuracy.py \
  --v563-root ../artifacts/report/v5_63/selective64_v3 \
  --v563-32-root ../artifacts/report/v5_63/selective32_v1 \
  --output ../artifacts/report/v5_63/latest_accuracy/summary.json
```

The builder verifies question counts, verdict IDs, token additivity, nearest-rank
statistics, answer hashes and the shared build ledger before publishing the
manifest. V5.54 structural factorial results remain documented in
[`docs/V5_54_INDEX_STRUCTURE_ABLATION.md`](docs/V5_54_INDEX_STRUCTURE_ABLATION.md).

## Tests

```bash
python -m pytest test/ -q
```

Unit tests use synthetic fixtures and do not require benchmark data or model
services. The V5.63 release gate passes all 523 tests.

## Repository layout

```text
src/graphmem/   graph construction, storage, retrieval, answer and serving
configs/v5/     frozen build and runtime profiles
scripts/        resumable benchmark, audit and rendering entrypoints
test/           unit and contract tests
docs/           design, deployment, result and ablation notes
```
