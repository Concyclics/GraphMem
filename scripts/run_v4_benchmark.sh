#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA="${DATA:?Set DATA to a converted LongMemEval or LoCoMo JSON file}"
OUTDIR="${OUTDIR:-${REPO}/runs/graphmem_v4_0}"
MAX_QUESTIONS="${MAX_QUESTIONS:-100000}"
QUESTION_WORKERS="${QUESTION_WORKERS:-32}"
BUILD_WORKERS="${BUILD_WORKERS:-32}"
LLM_MODEL="${LLM_MODEL:-${SGAO_MODEL:-gpt-5.4-mini}}"
LLM_BASE_URL="${LLM_BASE_URL:-${SGAO_BASE_URL:-https://sub2api.sgao.me/v1/}}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://127.0.0.1:8001/v1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen3-Embedding-0.6B}"

cd "${REPO}"
exec .venv/bin/python scripts/run_token_demo.py \
  --data "${DATA}" \
  --question-type all \
  --max-questions "${MAX_QUESTIONS}" \
  --output-dir "${OUTDIR}" \
  --variants hierarchical_hybrid_graph_v4_0 \
  --tree-mode hierarchical_hybrid_graph_v4_0 \
  --llm-model "${LLM_MODEL}" \
  --llm-base-url "${LLM_BASE_URL}" \
  --llm-api-key-env SGAO_API_KEY \
  --llm-request-profile openai \
  --embedding-base-url "${EMBEDDING_BASE_URL}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --question-workers "${QUESTION_WORKERS}" \
  --summary-workers "${BUILD_WORKERS}" \
  --max-inflight-deepseek "${BUILD_WORKERS}" \
  --build-budget-tokens 300000 \
  --answer-budget-tokens 10000 \
  --v36-answer-hard-budget-tokens 12100 \
  --v36-context-token-budget 8000 \
  --qa-max-tokens 512 \
  --reasoning-effort none \
  --resume
