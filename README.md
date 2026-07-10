# GraphMem

GraphMem is a graph-structured long-term memory system for conversational QA benchmarks such as [LongMemEval](https://github.com/LongMemEval/LongMemEval) and LoCoMo.

It builds a hierarchical memory graph from dialogue sessions, retrieves evidence with hybrid semantic / structured / graph signals, and answers questions with timeline-aware context assembly.

## Architecture

See [`docs/2026-07-08_graphmem_architecture_overview.md`](docs/2026-07-08_graphmem_architecture_overview.md) for the current design overview.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
```

## Data

Benchmark datasets are not bundled in this repo. Place converted JSON files under `data/`, for example:

- `data/longmemeval_s_cleaned.json` — LongMemEval-S (cleaned)
- `data/locomo10_graphmem.json` — LoCoMo converted to GraphMem format

Use `scripts/build_eval_subset.py` to create a fixed evaluation subset once you have the source benchmark file.

## Quick start

Run the main demo CLI:

```bash
python scripts/run_token_demo.py \
  --data data/longmemeval_s_subset50_balanced.json \
  --output-dir runs/my_run \
  --variants direct_session_k16_compact_graphmem \
  --max-questions 10
```

Evaluate generated answers:

```bash
python scripts/evaluate_answers.py \
  --answers runs/my_run/direct_session_k16_compact_graphmem/answers.jsonl \
  --data data/longmemeval_s_subset50_balanced.json \
  --output-jsonl runs/my_run/direct_session_k16_compact_graphmem/auto_eval.jsonl \
  --output-md runs/my_run/direct_session_k16_compact_graphmem/auto_eval.md
```

## Local vLLM workflow

For fully local runs with embedding + LLM services:

```bash
scripts/smoke_longmemeval.sh up      # start services
scripts/smoke_longmemeval.sh run     # smoke test (3 questions)
scripts/smoke_longmemeval.sh stop    # stop services
```

For a fixed 50-question subset with judging and stage audit:

```bash
scripts/run_subset_eval.sh
```

## Tests

```bash
python -m pytest test/ -q
```

Unit tests use synthetic fixtures under `test/fixtures/` and do not require benchmark data.

## Project layout

```
src/graphmem_demo/   Core library (build, retrieve, answer pipeline)
scripts/             CLI entrypoints and evaluation utilities
test/                Unit tests and fixtures
docs/                Design notes and experiment reports
```
