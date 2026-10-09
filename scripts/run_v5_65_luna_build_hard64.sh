#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO}")"
PY="${PYTHON_BIN:-${WORKSPACE}/.conda-envs/graphmem-v58/bin/python}"
ROOT="${V565_ROOT:-${WORKSPACE}/artifacts/report/v5_65/luna_build_hard64}"
SELECTION="${ROOT}/selection"
LME="${WORKSPACE}/artifacts/data/longmemeval_s_cleaned.json"
LOCOMO="${WORKSPACE}/artifacts/data/locomo10_graphmem.json"
GOLD="${REPO}/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"
CONFIG="${REPO}/configs/v5/v5_57_lossless_atomic.json"
RUNTIME_BASE="${REPO}/configs/v5/runtime_v5_59_hybrid64.json"
TOKENIZER="${WORKSPACE}/artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json"
MEMORY_BENCHMARKS="${WORKSPACE}/third_party/memory-benchmarks"
CONTROL_GRAPH="${WORKSPACE}/artifacts/report/v5_57/full/graph/graphmem.sqlite"
CONTROL_RELATION="${WORKSPACE}/artifacts/report/v5_57/full/graph/relation_embeddings.sqlite"

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi
export GRAPHMEM_TOKENIZER_PATH="${GRAPHMEM_TOKENIZER_PATH:-${TOKENIZER}}"
export PYTHONHASHSEED=0
LUNA_MODEL="${SGAO_MODEL_GPT56_LUNA:-gpt-5.6-luna}"
mkdir -p "${ROOT}"

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

build_arm() {
  local effort="$1"
  local arm="${ROOT}/luna_${effort}"
  mkdir -p "${arm}/graph"
  # Semantic extraction does not depend on the embedding service.  Warm its
  # exact response cache first, publishing only a disposable lexical graph.
  # This overlaps a managed embedding outage without changing the final graph.
  if [[ ! -f "${arm}/semantic_cache_warm.complete" ]]; then
    while ! "${PY}" "${REPO}/scripts/run_v5_6_full_build.py" \
        --target-db "${arm}/graph/graphmem.sqlite" \
        --memory-id-file "${SELECTION}/memory_ids.txt" \
        --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" \
        --config "${CONFIG}" --profile b5 \
        --skip-predicate-canonicalization \
        --memory-workers 8 --max-concurrency 32 \
        --llm-model "${LUNA_MODEL}" --llm-base-url "${SGAO_BASE_URL}" \
        --llm-api-key-env SGAO_API_KEY --llm-request-profile openai \
        --llm-reasoning-effort "${effort}" \
        --llm-request-timeout-seconds 1800 \
        --require-complete-diagnostics \
        --report "${arm}/build_report_semantic_warm.json" \
        >>"${arm}/build_semantic_warm.log" 2>&1; do
      event "luna-${effort} semantic warm pass interrupted; resuming"
      sleep 20
    done
    touch "${arm}/semantic_cache_warm.complete"
    event "luna-${effort} semantic extraction cache complete"
  fi

  if [[ ! -f "${arm}/final_materialization_started" ]]; then
    touch "${arm}/preserve_unpublished_llm_cache"
    "${PY}" - "${arm}/graph/graphmem.sqlite" <<'PY'
import sqlite3
import sys
with sqlite3.connect(sys.argv[1]) as db:
    db.execute("DELETE FROM graph_versions")
PY
    touch "${arm}/final_materialization_started"
    event "luna-${effort} disposable graph unpublished; full materialization queued"
  fi
  while true; do
    wait_embedding
    if "${PY}" "${REPO}/scripts/run_v5_6_full_build.py" \
        --target-db "${arm}/graph/graphmem.sqlite" \
        --relation-embedding-db "${arm}/graph/relation_embeddings.sqlite" \
        --memory-id-file "${SELECTION}/memory_ids.txt" \
        --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" \
        --config "${CONFIG}" --profile b5 --embedding \
        --embedding-request-model Qwen3-Embedding-0.6B \
        --memory-workers 8 --max-concurrency 32 \
        --llm-model "${LUNA_MODEL}" --llm-base-url "${SGAO_BASE_URL}" \
        --llm-api-key-env SGAO_API_KEY --llm-request-profile openai \
        --llm-reasoning-effort "${effort}" \
        --preserve-unpublished-llm-cache \
        --llm-request-timeout-seconds 1800 \
        --require-complete-diagnostics \
        --report "${arm}/build_report.json" \
        >>"${arm}/build.log" 2>&1; then
      break
    fi
    if [[ -f "${arm}/build_report.json" ]] \
        && [[ "$(jq '.summary.token_gate_violations | length' \
          "${arm}/build_report.json")" != 0 ]]; then
      event "luna-${effort} build hit a non-recoverable token-gate violation"
      return 1
    fi
    event "luna-${effort} build interrupted; preserving cache and resuming"
    sleep 20
  done
  event "luna-${effort} 42/42 memories built"
}

runtime_for_arm() {
  local arm="$1"
  local runtime="${ROOT}/${arm}/runtime.json"
  "${PY}" - "${RUNTIME_BASE}" "${runtime}" \
      "${ROOT}/${arm}/dense_indexes" \
      "${ROOT}/${arm}/compiled_memory_views" \
      "${WORKSPACE}/artifacts/report/v5_57/full/query_embedding_cache.sqlite" <<'PY'
import json
import sys
from pathlib import Path

source, target, dense, compiled, query_cache = map(Path, sys.argv[1:])
payload = json.loads(source.read_text(encoding="utf-8"))
payload["profile"] = "v5_65_luna_build_hard64"
retrieval = payload["retrieval"]
retrieval["dense_sidecar_dir"] = str(dense)
retrieval["compiled_cache_dir"] = str(compiled)
retrieval["query_embedding_cache_path"] = str(query_cache)
target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

question_args=()
while IFS= read -r question_id; do
  [[ -n "${question_id}" ]] && question_args+=(--question-id "${question_id}")
done <"${SELECTION}/question_ids.txt"

prepare_arm() {
  local arm="$1"
  local graph="$2"
  local relation="$3"
  local runtime="$4"
  local prepare="${ROOT}/${arm}/prepare"
  mkdir -p "${ROOT}/${arm}"
  while ! "${PY}" "${REPO}/scripts/run_v5_6_answer.py" \
      --source-db "${graph}" --relation-embedding-db "${relation}" \
      --output-root "${ROOT}/${arm}" --run-root "${prepare}" \
      --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" --full \
      --config "${CONFIG}" --runtime-config "${runtime}" \
      --answer-policy v5_63 --embedding \
      --embedding-request-model Qwen3-Embedding-0.6B \
      --prepare-only --navigate-workers 16 --checkpoint-every 64 --resume \
      "${question_args[@]}" \
      >>"${ROOT}/${arm}/prepare.log" 2>&1; do
    event "${arm} preparation needs embedding service; waiting to resume"
    wait_embedding
    sleep 5
  done
  event "${arm} frozen retrieval/prompt preparation complete"
}

answer_arm() {
  local arm="$1"
  local graph="$2"
  local answer="${ROOT}/${arm}/answer_luna_high"
  while ! "${PY}" "${REPO}/scripts/replay_v5_prepared_answers.py" \
      --prepared "${ROOT}/${arm}/prepare/prepared_answers.jsonl" \
      --metadata-answers "${SELECTION}/metadata_answers.jsonl" \
      --metadata-prompt-policy ignore --source-db "${graph}" \
      --config "${CONFIG}" --output-root "${answer}" \
      --answer-model "${LUNA_MODEL}" --answer-base-url "${SGAO_BASE_URL}" \
      --answer-api-key-env SGAO_API_KEY --answer-request-profile openai \
      --answer-reasoning-effort high --packing-model \
      Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
      --max-output-tokens 4096 --workers 64 --checkpoint-every 16 --resume \
      >>"${ROOT}/${arm}/answer.log" 2>&1; do
    event "${arm} Luna-high answer interrupted; resuming"
    sleep 20
  done
  event "${arm} 64/64 Luna-high answers complete"
}

judge_arm() {
  local arm="$1"
  local answer="${ROOT}/${arm}/answer_luna_high"
  local judge="${ROOT}/${arm}/judge_luna_medium"
  while ! "${PY}" "${REPO}/scripts/evaluate_mem0_judge.py" \
      --answers "${answer}/answers_longmemeval.jsonl" \
      --output-dir "${judge}/lme" --model "${LUNA_MODEL}" \
      --base-url "${SGAO_BASE_URL}" --api-key-env SGAO_API_KEY \
      --request-profile openai --reasoning-effort medium \
      --max-tokens 2048 --workers 32 --resume \
      >>"${ROOT}/${arm}/judge_lme.log" 2>&1; do
    event "${arm} LME judge interrupted; resuming"
    sleep 20
  done
  while ! "${PY}" \
      "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
      --data "${SELECTION}/locomo_hard64.json" \
      --answers "${answer}/answers_locomo.jsonl" \
      --output-dir "${judge}/locomo" \
      --memory-benchmarks-repo "${MEMORY_BENCHMARKS}" \
      --model "${LUNA_MODEL}" --base-url "${SGAO_BASE_URL}" \
      --api-key-env SGAO_API_KEY --request-profile openai \
      --reasoning-effort medium --max-tokens 2048 --workers 32 --resume \
      >>"${ROOT}/${arm}/judge_locomo.log" 2>&1; do
    event "${arm} LoCoMo judge interrupted; resuming"
    sleep 20
  done
  event "${arm} 64/64 Luna-medium verdicts complete"
}

# The complete control pipeline does not depend on either new build.  Run it
# immediately so cached query embeddings can overlap the managed embedding
# outage and all cold-build wall time.
control_pipeline() {
  prepare_arm qwen_control "${CONTROL_GRAPH}" "${CONTROL_RELATION}" \
    "${RUNTIME_BASE}"
  answer_arm qwen_control "${CONTROL_GRAPH}"
  judge_arm qwen_control
}
control_pipeline & control_pipeline_pid=$!

# Both new arms run concurrently under the remote endpoint's measured pending
# limit (32 each, 64 aggregate).
build_arm none & none_build_pid=$!
build_arm low & low_build_pid=$!
wait "${none_build_pid}"
wait "${low_build_pid}"

for effort in none low; do
  runtime_for_arm "luna_${effort}"
  "${PY}" "${REPO}/scripts/precompile_dense_indexes.py" \
    --db "${ROOT}/luna_${effort}/graph/graphmem.sqlite" \
    --config "${CONFIG}" --output "${ROOT}/luna_${effort}/dense_indexes" \
    --backend faiss_flat --workers 16 \
    >"${ROOT}/luna_${effort}/dense_precompile.log" 2>&1 &
  if [[ "${effort}" == "none" ]]; then
    none_faiss_pid=$!
  else
    low_faiss_pid=$!
  fi
done
wait "${none_faiss_pid}"
wait "${low_faiss_pid}"
event "both Luna-built graph FAISS sidecars compiled"

prepare_arm luna_none "${ROOT}/luna_none/graph/graphmem.sqlite" \
  "${ROOT}/luna_none/graph/relation_embeddings.sqlite" \
  "${ROOT}/luna_none/runtime.json" & none_prepare_pid=$!
prepare_arm luna_low "${ROOT}/luna_low/graph/graphmem.sqlite" \
  "${ROOT}/luna_low/graph/relation_embeddings.sqlite" \
  "${ROOT}/luna_low/runtime.json" & low_prepare_pid=$!
wait "${none_prepare_pid}"
wait "${low_prepare_pid}"

answer_arm luna_none "${ROOT}/luna_none/graph/graphmem.sqlite" & none_answer_pid=$!
answer_arm luna_low "${ROOT}/luna_low/graph/graphmem.sqlite" & low_answer_pid=$!
wait "${none_answer_pid}"
wait "${low_answer_pid}"

judge_arm luna_none & none_judge_pid=$!
judge_arm luna_low & low_judge_pid=$!
wait "${none_judge_pid}"
wait "${low_judge_pid}"
wait "${control_pipeline_pid}"

"${PY}" "${REPO}/scripts/summarize_luna_build_hard64.py" \
  --root "${ROOT}" --output "${ROOT}/summary.json"
event "Luna build-quality hard64 experiment complete"
