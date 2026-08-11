# GraphMem

GraphMem is a graph-structured long-term memory system for AI agents. It turns
multi-session conversations into a sparse hierarchical graph, compiles a user
question into a typed QueryIR plan, and retrieves a bounded evidence set for
answer generation.

The current V5.54 implementation focuses on three system goals:

- **Token-efficient construction.** HNSW-assisted coarsening, reconnecting and
  parent-gated refinement replace all-pairs relation induction. On the complete
  510-Memory corpus, the measured coarse-candidate exponent is `0.89` and the
  final relation set is `99.98%` smaller than the all-pairs upper bound.
- **Low-latency retrieval.** Per-memory FAISS indexes, compiled graph views,
  bounded top-down traversal and deterministic QueryIR execution avoid an LLM
  call on the query path.
- **High-precision readout.** Source-time normalization, witness-aware routing,
  topological evidence blocks and typed aggregation/readout contracts preserve
  multi-hop and temporal evidence under a fixed prompt budget.

The graph is stored as a versioned SQLite snapshot. The core package is under
`src/graphmem`; benchmark, audit and reproduction entrypoints are under
`scripts`.

## Qwen3-30B benchmark results

We evaluate the full [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval)
set (500 questions over 500 memories) and LoCoMo
[Category 1–4](https://github.com/snap-research/locomo) (1,540 questions over
10 conversations). LoCoMo Category 5 is excluded. GraphMem and the archived
Mem0 baseline use the Qwen3-30B answer-model family; all answers are judged by
the pinned `gpt-5.6-luna` prompts. Accuracy is answer accuracy, not retrieval
recall.

| Benchmark | System | Retrieval budget | Accuracy | Mean build Token | Mean answer Token |
|---|---|---:|---:|---:|---:|
| LongMemEval-S | GraphMem V5.54 | 32 turns | **74.00%** | **189,297** | 6,181 |
| LongMemEval-S | Mem0 baseline | top-50 | 56.80% | 2,461,197 | **5,202** |
| LongMemEval-S | GraphMem V5.54 | 64 turns | **76.20%** | **189,297** | **9,112** |
| LongMemEval-S | Mem0 baseline | top-200 | 57.60% | 2,461,197 | 11,761 |
| LoCoMo Cat. 1–4 | GraphMem V5.54 | 32 turns | **71.04%** | **75,243** | **3,205** |
| LoCoMo Cat. 1–4 | Mem0 baseline | top-50 | 67.66% | 5,318,818 | 5,514 |
| LoCoMo Cat. 1–4 | GraphMem V5.54 | 64 turns | **79.35%** | **75,243** | **5,613** |
| LoCoMo Cat. 1–4 | Mem0 baseline | top-200 | 68.31% | 5,318,818 | 13,515 |

The 32-turn/top-50 and 64-turn/top-200 rows are approximate retrieval-budget
operating points, not claims that their prompts contain exactly the same number
of tokens.

| Matched comparison | Accuracy gain | Build Token saving | Answer Token saving |
|---|---:|---:|---:|
| LongMemEval-S: 32 turns vs top-50 | **+17.20 pp** | **92.31%** | -18.83% |
| LongMemEval-S: 64 turns vs top-200 | **+18.60 pp** | **92.31%** | **22.52%** |
| LoCoMo: 32 turns vs top-50 | **+3.38 pp** | **98.59%** | **41.87%** |
| LoCoMo: 64 turns vs top-200 | **+11.04 pp** | **98.59%** | **58.47%** |

`-18.83%` means the LongMemEval-S 32-turn GraphMem prompt costs 18.83% more
answer Token than Mem0 top-50; it is retained to show the complete comparison.
Build Token means are per LongMemEval memory and per LoCoMo conversation.
Answer Token means are per question. Build values count generative API input +
output tokens and exclude embeddings and judge calls; one GraphMem build is
shared by its 32-turn and 64-turn query runs. Answer values count prompt +
completion tokens with a 2,000-token completion ceiling.

The exact p95/p99/max values, model/runtime caveats and audit hashes are in
[`docs/results/v5_54_qwen30b.json`](docs/results/v5_54_qwen30b.json). The Mem0
rows are an archived baseline at commit `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`;
Mem0 used BF16 while GraphMem used the FP8 Qwen checkpoint, so the comparison is
same-family rather than bit-identical inference.

## Reproduce V5.54

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
python scripts/run_v5_6_full_build.py \
  --target-db ../artifacts/repro/v5_54/graph/graphmem.sqlite \
  --relation-embedding-db ../artifacts/repro/v5_54/graph/relation_embeddings.sqlite \
  --lme ../artifacts/data/longmemeval_s_cleaned.json \
  --locomo ../artifacts/data/locomo10_graphmem.json \
  --gold eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl \
  --config configs/v5/v5_17_budget230.json --profile b5 \
  --enabled-relation-signals scene_similar,shared_entity,state_compatible \
  --embedding --memory-workers 2 --max-concurrency 256 \
  --require-zero-retries --require-complete-diagnostics \
  --report ../artifacts/repro/v5_54/build_report.json
```

The runner is resumable by graph version and exact LLM-call cache. A valid full
build has 510/510 published memories, no SDK retry, complete diagnostics and no
memory above the configured 230K generative-token gate.

### 4. Run the 32-turn and 64-turn answer arms

```bash
for budget in 32 64; do
  python scripts/run_v5_6_answer.py \
    --source-db ../artifacts/repro/v5_54/graph/graphmem.sqlite \
    --output-root ../artifacts/repro/v5_54/turn${budget} \
    --run-root ../artifacts/repro/v5_54/turn${budget}/answer \
    --lme ../artifacts/data/longmemeval_s_cleaned.json \
    --locomo ../artifacts/data/locomo10_graphmem.json \
    --gold eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl \
    --config configs/v5/v5_17_budget230.json \
    --runtime-config configs/v5/runtime_v5_54_accuracy${budget}.json \
    --answer-policy v5_54 --embedding --full \
    --max-output-tokens 2000 --answer-workers 256 \
    --label v554_qwen30b_turn${budget} --resume
done
```

Runtime JSON is the authority for the evidence budget, QueryIR, hierarchy,
FAISS and traversal settings. `--answer-policy v5_54` selects the frozen
topological/typed readout in core code; no offline prompt materializer is
required.

### 5. Judge and audit

With `SGAO_API_KEY` and `SGAO_BASE_URL` set in the ignored `.env`, judge each
arm using the pinned prompts:

```bash
python scripts/evaluate_mem0_judge.py \
  --answers ../artifacts/repro/v5_54/turn64/answer/answers.jsonl \
  --output-dir ../artifacts/repro/v5_54/turn64/answer/judge_longmemeval \
  --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
  --request-profile openai --workers 32 --resume

python scripts/evaluate_memory_benchmarks_locomo_judge.py \
  --data ../artifacts/data/locomo10_graphmem.json \
  --answers ../artifacts/repro/v5_54/turn64/answer/answers.jsonl \
  --output-dir ../artifacts/repro/v5_54/turn64/answer/judge_locomo \
  --memory-benchmarks-repo ../third_party/memory-benchmarks \
  --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
  --request-profile openai --workers 32 --resume
```

Repeat for `turn32`. Percentiles use nearest rank,
`ceil(p * N) - 1`. For the structural factorial and its checksum, prompt-hash,
usage-sum and paired-verdict gates, see
[`docs/V5_54_INDEX_STRUCTURE_ABLATION.md`](docs/V5_54_INDEX_STRUCTURE_ABLATION.md).

## Tests

```bash
python -m pytest test/ -q
```

Unit tests use synthetic fixtures and do not require benchmark data or model
services.

## Repository layout

```text
src/graphmem/   graph construction, storage, retrieval, answer and serving
configs/v5/     frozen build and runtime profiles
scripts/        resumable benchmark, audit and rendering entrypoints
test/           unit and contract tests
docs/           design, deployment, result and ablation notes
```
