#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "${REPO}/.venv/bin/python" ]]; then
  echo "Missing ${REPO}/.venv" >&2
  exit 2
fi

RAW_DATA="${RAW_DATA:-/mnt/ssd1/yongan/Resources/RefRepos/locomo_benchmark/locomo/data/locomo10.json}"
DATA="${DATA:-${REPO}/data/locomo10_graphmem.json}"
OUT="${OUT:-${REPO}/runs/locomo10_v2_full_20260725}"
MAX_QUESTIONS="${MAX_QUESTIONS:-1986}"
QUESTION_WORKERS="${QUESTION_WORKERS:-32}"
BUILD_LLM_MAX_INFLIGHT="${BUILD_LLM_MAX_INFLIGHT:-512}"
MEMORY_CACHE_DIR="${MEMORY_CACHE_DIR:-${OUT}/memory_cache}"

if [[ ! -f "${RAW_DATA}" ]]; then
  echo "Official LoCoMo data not found: ${RAW_DATA}" >&2
  exit 2
fi

"${REPO}/.venv/bin/python" "${REPO}/scripts/convert_locomo10.py" \
  --input "${RAW_DATA}" --output "${DATA}"
"${REPO}/.venv/bin/python" "${REPO}/scripts/check_v2_services.py"
"${REPO}/.venv/bin/python" "${REPO}/scripts/run_token_demo.py" \
  --data "${DATA}" --question-type all --variants hierarchical_state_graph_v2 \
  --output-dir "${OUT}" --memory-cache-dir "${MEMORY_CACHE_DIR}" \
  --llm-model gpt-5.4-mini --llm-base-url https://sub2api.sgao.me/v1/ \
  --embedding-base-url http://127.0.0.1:8001/v1 --embedding-model Qwen3-Embedding-0.6B \
  --max-questions "${MAX_QUESTIONS}" --question-workers "${QUESTION_WORKERS}" \
  --summary-workers "${BUILD_LLM_MAX_INFLIGHT}" \
  --max-inflight-deepseek "${BUILD_LLM_MAX_INFLIGHT}" \
  --reasoning-effort none --build-budget-tokens 300000 --answer-budget-tokens 10000 \
  --qa-max-tokens 512 --v2-fact-extraction-max-tokens 3072 \
  --v2-consolidation-max-tokens 3072 --v2-context-token-budget 7600 \
  --v2-card-k 6 --v2-fact-k 14 --v2-leaf-k 14 \
  --v2-semantic-k 3 --v2-semantic-floor 0.55 --resume

VARIANT_DIR="${OUT}/hierarchical_state_graph_v2"
"${REPO}/.venv/bin/python" "${REPO}/scripts/evaluate_locomo_official.py" \
  --data "${DATA}" --answers "${VARIANT_DIR}/answers.jsonl" \
  --output-dir "${OUT}/official_eval"
"${REPO}/.venv/bin/python" "${REPO}/scripts/summarize_locomo_v2_tokens.py" \
  --data "${DATA}" --run-dir "${VARIANT_DIR}" --output-dir "${OUT}/token_analysis"
