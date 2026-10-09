#!/usr/bin/env bash
set -u

# Rejudge the frozen 2026-09-10 LoCoMo top-200 responses with local Qwen3-30B.
# This wrapper waits for the managed vLLM service and keeps each judge output
# resumable when the service is restarted.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GRAPHMEM_ROOT="${ROOT}/GraphMem"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.conda-envs/graphmem-v58/bin/python}"
BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:8002/v1}"
MODEL="${QWEN_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
WORKERS="${QWEN_JUDGE_WORKERS:-128}"
RUN_ROOT="${ROOT}/artifacts/report/v5_76_adaptive_budget/qwen30_rejudge_20260910"
INPUT_ROOT="${RUN_ROOT}/inputs"
DATA="${INPUT_ROOT}/locomo_category_1_4_judge_data.json"
JUDGE_SCRIPT="${GRAPHMEM_ROOT}/scripts/evaluate_memory_benchmarks_locomo_judge.py"
MEMORY_BENCHMARKS_REPO="${ROOT}/third_party/memory-benchmarks"

wait_for_chat() {
  while true; do
    if curl -fsS --max-time 5 "${BASE_URL}/chat/completions" \
      -H 'Content-Type: application/json' \
      -H 'Authorization: Bearer local' \
      --data-raw "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Return OK\"}],\"temperature\":0,\"max_tokens\":8}" \
      >/dev/null 2>&1; then
      return 0
    fi
    echo "$(date -u +%FT%TZ) waiting for Qwen chat service at ${BASE_URL}" >&2
    sleep 10
  done
}

run_one() {
  local name="$1"
  local answer_file="$2"
  local answers="${INPUT_ROOT}/${answer_file}"
  local output="${RUN_ROOT}/judge_${name}"
  mkdir -p "${output}"
  wait_for_chat
  env LOCAL_API_KEY=local "${PYTHON_BIN}" "${JUDGE_SCRIPT}" \
    --data "${DATA}" \
    --answers "${answers}" \
    --output-dir "${output}" \
    --memory-benchmarks-repo "${MEMORY_BENCHMARKS_REPO}" \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --api-key-env LOCAL_API_KEY \
    --request-profile openai \
    --workers "${WORKERS}" \
    --reasoning-effort none \
    --max-tokens 256 \
    --resume
}

# Keep total in-flight calls bounded by two workers x 128 (the usual vLLM
# max-num-seqs setting), while retaining parallelism across baseline methods.
run_one mem0 mem0_top200_answers.jsonl & p1=$!
run_one jiuwen jiuwen_top200_answers.jsonl & p2=$!
wait "${p1}" "${p2}"
run_one jiuwen_memory_turbo jiuwen_memory_turbo_top200_answers.jsonl & p3=$!
run_one graphmem_selection_modal graphmem_selection_modal_answers.jsonl & p4=$!
wait "${p3}" "${p4}"
