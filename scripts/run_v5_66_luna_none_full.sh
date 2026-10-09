#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO}")"
PY="${PYTHON_BIN:-${WORKSPACE}/.conda-envs/graphmem-v58/bin/python}"
ROOT="${V566_ROOT:-${WORKSPACE}/artifacts/report/v5_66/luna_none_full}"
HARD64="${WORKSPACE}/artifacts/report/v5_65/luna_build_hard64/luna_none"
CONTROL_ROOT="${WORKSPACE}/artifacts/report/v5_57/full"
CONTROL_PREPARED="${WORKSPACE}/artifacts/report/v5_63/selective64_v3/prepared_answers.jsonl"
CONTROL_METADATA="${WORKSPACE}/artifacts/report/v5_63/selective64_v3/answer/answers.jsonl"
LME="${WORKSPACE}/artifacts/data/longmemeval_s_cleaned.json"
LOCOMO="${WORKSPACE}/artifacts/data/locomo10_graphmem.json"
GOLD="${REPO}/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"
CONFIG="${REPO}/configs/v5/v5_57_lossless_atomic.json"
RUNTIME_BASE="${REPO}/configs/v5/runtime_v5_59_hybrid64.json"
TOKENIZER="${WORKSPACE}/artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json"
MEMORY_BENCHMARKS="${WORKSPACE}/third_party/memory-benchmarks"

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi
export GRAPHMEM_TOKENIZER_PATH="${GRAPHMEM_TOKENIZER_PATH:-${TOKENIZER}}"
export PYTHONHASHSEED=0
LUNA_MODEL="${SGAO_MODEL_GPT56_LUNA:-gpt-5.6-luna}"
mkdir -p "${ROOT}/graph"

event() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    | tee -a "${ROOT}/orchestrator.log"
}

wait_embedding() {
  local delay=5
  until curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null; do
    event "embedding service unavailable; waiting ${delay}s for managed restart"
    sleep "${delay}"
    if (( delay < 30 )); then delay=$((delay + 5)); fi
  done
}

required=(
  "${PY}" "${HARD64}/graph/graphmem.sqlite"
  "${HARD64}/graph/relation_embeddings.sqlite" "${HARD64}/build_report.json"
  "${CONTROL_ROOT}/graph/graphmem.sqlite" "${CONTROL_PREPARED}"
  "${CONTROL_METADATA}" "${LME}" "${LOCOMO}" "${GOLD}" "${CONFIG}"
  "${RUNTIME_BASE}" "${TOKENIZER}" "${MEMORY_BENCHMARKS}"
)
for path in "${required[@]}"; do
  [[ -e "${path}" ]] || { event "missing required path: ${path}"; exit 2; }
done

while true; do
  wait_embedding
  if "${PY}" "${REPO}/scripts/run_v5_6_full_build.py" \
      --seed-db "${HARD64}/graph/graphmem.sqlite" \
      --seed-report "${HARD64}/build_report.json" \
      --seed-embedding-db "${CONTROL_ROOT}/graph/graphmem.sqlite" \
      --seed-relation-embedding-db \
        "${HARD64}/graph/relation_embeddings.sqlite" \
      --target-db "${ROOT}/graph/graphmem.sqlite" \
      --relation-embedding-db "${ROOT}/graph/relation_embeddings.sqlite" \
      --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" \
      --config "${CONFIG}" --profile b5 --embedding \
      --embedding-request-model Qwen3-Embedding-0.6B \
      --memory-workers 16 --max-concurrency 64 \
      --llm-model "${LUNA_MODEL}" --llm-base-url "${SGAO_BASE_URL}" \
      --llm-api-key-env SGAO_API_KEY --llm-request-profile openai \
      --llm-reasoning-effort none --llm-request-timeout-seconds 300 \
      --require-complete-diagnostics \
      --report "${ROOT}/build_report.json" \
      >>"${ROOT}/build.log" 2>&1; then
    break
  fi
  if [[ -f "${ROOT}/build_report.json" ]] \
      && [[ "$(jq '.summary.token_gate_violations | length' \
        "${ROOT}/build_report.json")" != 0 ]]; then
    event "full build hit a non-recoverable token-gate violation"
    exit 1
  fi
  event "full build pass interrupted; preserving published graphs and resuming"
  sleep 20
done
event "Luna-none full graph build complete"

"${PY}" "${REPO}/scripts/precompile_dense_indexes.py" \
  --db "${ROOT}/graph/graphmem.sqlite" --config "${CONFIG}" \
  --output "${ROOT}/dense_indexes" --backend faiss_flat --workers 16 \
  >"${ROOT}/dense_precompile.log" 2>&1
event "full graph FAISS sidecars compiled"

"${PY}" - "${RUNTIME_BASE}" "${ROOT}/runtime.json" \
    "${ROOT}/dense_indexes" "${ROOT}/compiled_memory_views" \
    "${CONTROL_ROOT}/query_embedding_cache.sqlite" <<'PY'
import json
import sys
from pathlib import Path

source, target, dense, compiled, query_cache = map(Path, sys.argv[1:])
payload = json.loads(source.read_text(encoding="utf-8"))
payload["profile"] = "v5_66_luna_none_full"
retrieval = payload["retrieval"]
retrieval["dense_sidecar_dir"] = str(dense)
retrieval["compiled_cache_dir"] = str(compiled)
retrieval["query_embedding_cache_path"] = str(query_cache)
target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

while true; do
  wait_embedding
  if "${PY}" "${REPO}/scripts/run_v5_6_answer.py" \
      --source-db "${ROOT}/graph/graphmem.sqlite" \
      --relation-embedding-db "${ROOT}/graph/relation_embeddings.sqlite" \
      --output-root "${ROOT}/candidate" --run-root "${ROOT}/candidate/prepare" \
      --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" --full \
      --config "${CONFIG}" --runtime-config "${ROOT}/runtime.json" \
      --answer-policy v5_63 --embedding \
      --embedding-request-model Qwen3-Embedding-0.6B \
      --prepare-only --navigate-workers 32 --checkpoint-every 100 --resume \
      >>"${ROOT}/candidate_prepare.log" 2>&1; then
    break
  fi
  event "candidate preparation interrupted; waiting to resume"
  sleep 10
done
event "candidate 2,040-request preparation complete"

answer_arm() {
  local label="$1"
  local prepared="$2"
  local metadata="$3"
  local source_db="$4"
  local output="${ROOT}/${label}/answer"
  while ! "${PY}" "${REPO}/scripts/replay_v5_prepared_answers.py" \
      --prepared "${prepared}" --metadata-answers "${metadata}" \
      --metadata-prompt-policy ignore --source-db "${source_db}" \
      --config "${CONFIG}" --output-root "${output}" \
      --answer-model "${LUNA_MODEL}" --answer-base-url "${SGAO_BASE_URL}" \
      --answer-api-key-env SGAO_API_KEY --answer-request-profile openai \
      --answer-reasoning-effort high \
      --packing-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
      --max-output-tokens 16384 --workers 64 --checkpoint-every 50 --resume \
      >>"${ROOT}/${label}_answer.log" 2>&1; do
    event "${label} Luna-high answer pass interrupted; resuming"
    sleep 20
  done
  event "${label} 2,040/2,040 Luna-high answers complete"
}

answer_arm control "${CONTROL_PREPARED}" "${CONTROL_METADATA}" \
  "${CONTROL_ROOT}/graph/graphmem.sqlite" & control_answer_pid=$!
answer_arm candidate "${ROOT}/candidate/prepare/prepared_answers.jsonl" \
  "${CONTROL_METADATA}" "${ROOT}/graph/graphmem.sqlite" \
  & candidate_answer_pid=$!
wait "${control_answer_pid}"
wait "${candidate_answer_pid}"

judge_arm() {
  local label="$1"
  local answer="${ROOT}/${label}/answer"
  local judge="${ROOT}/${label}/judge/luna_medium"
  while ! "${PY}" "${REPO}/scripts/evaluate_mem0_judge.py" \
      --answers "${answer}/answers_longmemeval.jsonl" \
      --output-dir "${judge}/lme" --model "${LUNA_MODEL}" \
      --base-url "${SGAO_BASE_URL}" --api-key-env SGAO_API_KEY \
      --request-profile openai --reasoning-effort medium \
      --max-tokens 2048 --workers 32 --resume \
      >>"${ROOT}/${label}_judge_lme.log" 2>&1; do
    event "${label} LME judge interrupted; resuming"
    sleep 20
  done
  while ! "${PY}" \
      "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
      --data "${LOCOMO}" --answers "${answer}/answers_locomo.jsonl" \
      --output-dir "${judge}/locomo" \
      --memory-benchmarks-repo "${MEMORY_BENCHMARKS}" \
      --model "${LUNA_MODEL}" --base-url "${SGAO_BASE_URL}" \
      --api-key-env SGAO_API_KEY --request-profile openai \
      --reasoning-effort medium --max-tokens 2048 --workers 32 --resume \
      >>"${ROOT}/${label}_judge_locomo.log" 2>&1; do
    event "${label} LoCoMo judge interrupted; resuming"
    sleep 20
  done
  event "${label} 2,040/2,040 Luna-medium verdicts complete"
}

judge_arm control & control_judge_pid=$!
judge_arm candidate & candidate_judge_pid=$!
wait "${control_judge_pid}"
wait "${candidate_judge_pid}"

for label in control candidate; do
  "${PY}" "${REPO}/scripts/summarize_gpt56_full.py" \
    --root "${ROOT}/${label}" \
    --prepared "$(if [[ "${label}" == control ]]; then \
      printf '%s' "${CONTROL_PREPARED}"; else \
      printf '%s' "${ROOT}/candidate/prepare/prepared_answers.jsonl"; fi)" \
    --judges luna_medium --output "${ROOT}/${label}/full_summary.json"
done
event "Luna-none full rebuild and paired Luna-high benchmark complete"
