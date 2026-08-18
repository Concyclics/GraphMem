#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO}")"
PY="${PYTHON_BIN:-${WORKSPACE}/.conda-envs/graphmem-v58/bin/python}"
ROOT="${V557_FULL_ROOT:-${WORKSPACE}/artifacts/report/v5_57/full}"
LME="${WORKSPACE}/artifacts/data/longmemeval_s_cleaned.json"
LOCOMO="${WORKSPACE}/artifacts/data/locomo10_graphmem.json"
GOLD="${REPO}/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"
CONFIG="${REPO}/configs/v5/v5_57_lossless_atomic.json"
TOKENIZER="${WORKSPACE}/artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json"
MEMORY_BENCHMARKS="${WORKSPACE}/third_party/memory-benchmarks"
MEM0_RESULTS="${WORKSPACE}/artifacts/report/v5_19/mem0_qwen30_cutoffs.json"
GRAPH_DB="${ROOT}/graph/graphmem.sqlite"
RELATION_DB="${ROOT}/graph/relation_embeddings.sqlite"
BUILD_REPORT="${ROOT}/build_report.json"
BUILD_MEMORY_WORKERS="${V557_BUILD_MEMORY_WORKERS:-8}"
BUILD_CONCURRENCY="${V557_BUILD_CONCURRENCY:-256}"
ANSWER_WORKERS="${V557_ANSWER_WORKERS:-256}"
JUDGE_WORKERS="${V557_JUDGE_WORKERS:-32}"

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi
export GRAPHMEM_TOKENIZER_PATH="${GRAPHMEM_TOKENIZER_PATH:-${TOKENIZER}}"
export PYTHONHASHSEED=0
mkdir -p "${ROOT}/graph"

event() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "${ROOT}/orchestrator.log"
}

wait_local_models() {
  local delay=5
  until curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null \
      && curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null; do
    event "local model service unavailable; waiting ${delay}s for managed restart"
    sleep "${delay}"
    if (( delay < 30 )); then delay=$((delay + 5)); fi
  done
}

run_build() {
  local reservation_args=()
  if [[ -n "${V557_BUILD_RESERVATION_SAFETY_OVERRIDE:-}" ]]; then
    reservation_args=(
      --semantic-request-reservation-safety-override
      "${V557_BUILD_RESERVATION_SAFETY_OVERRIDE}")
  fi
  wait_local_models
  "${PY}" "${REPO}/scripts/run_v5_6_full_build.py" \
    --target-db "${GRAPH_DB}" --relation-embedding-db "${RELATION_DB}" \
    --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" \
    --config "${CONFIG}" --profile b5 --embedding \
    --embedding-request-model Qwen3-Embedding-0.6B \
    --memory-workers "${BUILD_MEMORY_WORKERS}" \
    --max-concurrency "${BUILD_CONCURRENCY}" \
    --llm-request-timeout-seconds 1800 \
    "${reservation_args[@]}" \
    --require-zero-retries --require-complete-diagnostics \
    --report "${BUILD_REPORT}"
}

until run_build >>"${ROOT}/build.log" 2>&1; do
  event "build pass incomplete; preserving graph/cache and resuming"
  sleep 30
done
event "510/510 memories built and audited"

"${PY}" "${REPO}/scripts/precompile_dense_indexes.py" \
  --db "${GRAPH_DB}" --config "${CONFIG}" \
  --output "${ROOT}/dense_indexes" --backend faiss_flat --workers 16 \
  >"${ROOT}/dense_precompile.log" 2>&1
event "510 per-memory FAISS indexes compiled"

answer_arm() {
  local turns="$1"
  local arm="${ROOT}/turn${turns}"
  local runtime="${REPO}/configs/v5/runtime_v5_57_accuracy${turns}.json"
  mkdir -p "${arm}/answer"
  until wait_local_models && "${PY}" "${REPO}/scripts/run_v5_6_answer.py" \
      --source-db "${GRAPH_DB}" --output-root "${arm}" \
      --run-root "${arm}/answer" \
      --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" --full \
      --config "${CONFIG}" --runtime-config "${runtime}" \
      --answer-policy v5_54 --embedding \
      --embedding-request-model Qwen3-Embedding-0.6B \
      --relation-embedding-db "${RELATION_DB}" \
      --answer-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
      --answer-base-url http://127.0.0.1:8002/v1 \
      --answer-request-profile qwen --max-output-tokens 2000 \
      --answer-workers "${ANSWER_WORKERS}" --navigate-workers 32 \
      --checkpoint-every 50 --label "v557_accuracy${turns}_qwen30" --resume \
      >>"${arm}/answer.log" 2>&1; do
    event "turn${turns} answer pass interrupted; resuming from checkpoint"
    sleep 30
  done
  event "turn${turns} answers and frozen PreparedAnswer records complete"
}

judge_arm() {
  local turns="$1"
  local answer="${ROOT}/turn${turns}/answer"
  until "${PY}" "${REPO}/scripts/evaluate_mem0_judge.py" \
      --answers "${answer}/answers_longmemeval.jsonl" \
      --output-dir "${answer}/judge_lme" --model gpt-5.6-luna \
      --api-key-env SGAO_API_KEY --request-profile openai \
      --workers "${JUDGE_WORKERS}" --resume; do
    event "turn${turns} LongMemEval judge unavailable; resuming"
    sleep 30
  done
  until "${PY}" "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
      --data "${LOCOMO}" --answers "${answer}/answers_locomo.jsonl" \
      --output-dir "${answer}/judge_locomo" \
      --memory-benchmarks-repo "${MEMORY_BENCHMARKS}" \
      --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
      --request-profile openai --workers "${JUDGE_WORKERS}" --resume; do
    event "turn${turns} LoCoMo judge unavailable; resuming"
    sleep 30
  done
  event "turn${turns} judges complete"
}

# Run one answer arm at a time so both have the same full local-service budget;
# remote judging overlaps the next arm and therefore does not idle the Qwen GPU.
answer_arm 32
judge_arm 32 & judge32_pid=$!
answer_arm 64
judge_arm 64
wait "${judge32_pid}"
"${PY}" "${REPO}/scripts/summarize_v5_20_budget_benchmark.py" \
  --root "${ROOT}" --mem0 "${MEM0_RESULTS}" \
  --build-report "${BUILD_REPORT}" --output "${ROOT}/summary.json"
event "V5.57 paired 32/64 full benchmark complete"
