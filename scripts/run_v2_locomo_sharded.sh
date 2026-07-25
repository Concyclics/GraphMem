#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DATA="${RAW_DATA:-/mnt/ssd1/yongan/Resources/RefRepos/locomo_benchmark/locomo/data/locomo10.json}"
DATA="${DATA:-${REPO}/data/locomo10_graphmem.json}"
OUT="${OUT:-${REPO}/runs/locomo10_v2_sharded4_20260725}"
SHARD_COUNT="${SHARD_COUNT:-4}"
QUESTION_WORKERS="${QUESTION_WORKERS:-64}"
BUILD_LLM_MAX_INFLIGHT="${BUILD_LLM_MAX_INFLIGHT:-512}"
MEMORY_CACHE_DIR="${MEMORY_CACHE_DIR:-${OUT}/memory_cache}"
SHARD_DATA_DIR="${OUT}/shard_data"
SHARD_RUN_ROOT="${OUT}/shards"
MERGED_DIR="${OUT}/hierarchical_state_graph_v2"

"${REPO}/.venv/bin/python" "${REPO}/scripts/convert_locomo10.py" \
  --input "${RAW_DATA}" --output "${DATA}"
"${REPO}/.venv/bin/python" "${REPO}/scripts/shard_locomo_graphmem.py" \
  --data "${DATA}" --output-dir "${SHARD_DATA_DIR}" --shards "${SHARD_COUNT}"
"${REPO}/.venv/bin/python" "${REPO}/scripts/check_v2_services.py"

pids=()
for index in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_out="${SHARD_RUN_ROOT}/shard_${index}"
  mkdir -p "${shard_out}"
  "${REPO}/.venv/bin/python" "${REPO}/scripts/run_token_demo.py" \
    --data "${SHARD_DATA_DIR}/shard_${index}.json" \
    --question-type all --variants hierarchical_state_graph_v2 \
    --output-dir "${shard_out}" --memory-cache-dir "${MEMORY_CACHE_DIR}" \
    --deepseek-model deepseek-v4-flash --deepseek-base-url https://api.deepseek.com \
    --embedding-base-url http://127.0.0.1:8001/v1 \
    --embedding-model Qwen3-Embedding-0.6B \
    --max-questions 1986 --question-workers "${QUESTION_WORKERS}" \
    --summary-workers "${BUILD_LLM_MAX_INFLIGHT}" \
    --max-inflight-deepseek "${BUILD_LLM_MAX_INFLIGHT}" \
    --reasoning-effort none --build-budget-tokens 300000 \
    --answer-budget-tokens 10000 --qa-max-tokens 512 \
    --v2-fact-extraction-max-tokens 3072 --v2-consolidation-max-tokens 3072 \
    --v2-context-token-budget 7600 --v2-card-k 6 --v2-fact-k 14 \
    --v2-leaf-k 14 --v2-semantic-k 3 --v2-semantic-floor 0.55 --resume \
    > "${shard_out}/run.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "At least one LoCoMo shard failed; inspect ${SHARD_RUN_ROOT}/shard_*/run.log" >&2
  exit 1
fi

"${REPO}/.venv/bin/python" "${REPO}/scripts/merge_locomo_shards.py" \
  --shard-root "${SHARD_RUN_ROOT}" --output-dir "${MERGED_DIR}" \
  --expected-questions 1986
"${REPO}/.venv/bin/python" "${REPO}/scripts/evaluate_locomo_official.py" \
  --data "${DATA}" --answers "${MERGED_DIR}/answers.jsonl" \
  --output-dir "${OUT}/official_eval"
"${REPO}/.venv/bin/python" "${REPO}/scripts/summarize_locomo_v2_tokens.py" \
  --data "${DATA}" --run-dir "${MERGED_DIR}" --output-dir "${OUT}/token_analysis"
