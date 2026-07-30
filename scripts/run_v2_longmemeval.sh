#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "${REPO}/.venv/bin/python" ]]; then echo "Missing ${REPO}/.venv; run: uv venv --system-site-packages .venv && uv pip install --python .venv/bin/python -r requirements.txt" >&2; exit 2; fi
DEFAULT_DATA="/mnt/ssd1/yongan/Resources/RefRepos/longmemeval_benchmark/LongMemEval/data/longmemeval_s_cleaned.json"
DATA="${DATA:-${1:-${DEFAULT_DATA}}}"
OUT="${OUT:-${2:-/mnt/ssd1/pzx/share/graphmem/hierarchical_state_graph_v2_500}}"
MAX_QUESTIONS="${MAX_QUESTIONS:-500}"
QUESTION_WORKERS="${QUESTION_WORKERS:-8}"
BUILD_LLM_MAX_INFLIGHT="${BUILD_LLM_MAX_INFLIGHT:-256}"
JUDGE_WORKERS="${JUDGE_WORKERS:-64}"
if [[ ! -f "${DATA}" ]]; then echo "LongMemEval data not found: ${DATA}" >&2; exit 2; fi
"${REPO}/.venv/bin/python" "${REPO}/scripts/check_v2_services.py"
"${REPO}/.venv/bin/python" "${REPO}/scripts/run_token_demo.py" \
  --data "${DATA}" --question-type all --variants hierarchical_state_graph_v2 \
  --output-dir "${OUT}" --memory-cache-dir "${MEMORY_CACHE_DIR:-${REPO}/runs/v2_memory_cache}" --llm-model gpt-5.4-mini --llm-base-url https://sub2api.sgao.me/v1/ \
  --embedding-base-url http://127.0.0.1:8001/v1 --embedding-model Qwen3-Embedding-0.6B \
  --max-questions "${MAX_QUESTIONS}" --question-workers "${QUESTION_WORKERS}" --summary-workers "${BUILD_LLM_MAX_INFLIGHT}" --max-inflight-deepseek "${BUILD_LLM_MAX_INFLIGHT}" \
  --reasoning-effort none --build-budget-tokens 300000 --answer-budget-tokens 10000 \
  --qa-max-tokens 512 --v2-fact-extraction-max-tokens 3072 --v2-consolidation-max-tokens 3072 --v2-context-token-budget 7600 --v2-card-k 6 --v2-fact-k 14 --v2-leaf-k 14 \
  --v2-semantic-k 3 --v2-semantic-floor 0.55 --resume
if [[ "${RUN_JUDGE:-1}" == "1" ]]; then
  "${REPO}/.venv/bin/python" "${REPO}/scripts/evaluate_mem0_judge.py" \
    --answers "${OUT}/hierarchical_state_graph_v2/answers.jsonl" --output-dir "${OUT}/mem0_judge" \
    --model gpt-5.4-mini --base-url https://sub2api.sgao.me/v1/ --workers "${JUDGE_WORKERS}" --resume
fi
