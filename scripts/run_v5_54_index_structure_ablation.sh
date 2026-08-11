#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO}")"
PY="${PYTHON_BIN:-${WORKSPACE}/.conda-envs/graphmem-v58/bin/python}"
ROOT="${V554_INDEX_ROOT:-${WORKSPACE}/artifacts/report/v5_54/index_structure_ablation}"
PHASE="${V554_INDEX_PHASE:-all}"
LME="${WORKSPACE}/artifacts/data/longmemeval_s_cleaned.json"
LOCOMO="${WORKSPACE}/artifacts/data/locomo10_graphmem.json"
GOLD="${REPO}/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"
CONFIG="${REPO}/configs/v5/v5_17_budget230.json"
GRAPH="${WORKSPACE}/artifacts/report/v5_21/full_minimal_repair/m2_safe_witness/graph/graphmem.sqlite"
TOKENIZER="${WORKSPACE}/artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json"
BASELINE="${WORKSPACE}/artifacts/report/v5_54/anonymous_temporal_layout/final"
MEMORY_BENCHMARKS="${WORKSPACE}/third_party/memory-benchmarks"
ANSWER_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
ANSWER_WORKERS="${V554_INDEX_ANSWER_WORKERS:-256}"
JUDGE_WORKERS="${V554_INDEX_JUDGE_WORKERS:-32}"

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi
export GRAPHMEM_TOKENIZER_PATH="${GRAPHMEM_TOKENIZER_PATH:-${TOKENIZER}}"
export GRAPHMEM_LOCAL_API_KEY="${GRAPHMEM_LOCAL_API_KEY:-EMPTY}"
export PYTHONHASHSEED=0

for path in "${PY}" "${LME}" "${LOCOMO}" "${GOLD}" "${CONFIG}" \
            "${GRAPH}" "${GRAPHMEM_TOKENIZER_PATH}" "${BASELINE}/answers.jsonl" \
            "${BASELINE}/answer_usage.jsonl" "${BASELINE}/paired_judge_longmemeval.jsonl" \
            "${BASELINE}/paired_judge_locomo.jsonl" "${MEMORY_BENCHMARKS}"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done
mkdir -p "${ROOT}"

arms=(seed_only hierarchy_only flat_graph full)
budgets=(32 64)
if [[ -n "${V554_INDEX_ONLY_ARM:-}" ]]; then arms=("${V554_INDEX_ONLY_ARM}"); fi
if [[ -n "${V554_INDEX_ONLY_BUDGET:-}" ]]; then budgets=("${V554_INDEX_ONLY_BUDGET}"); fi
benchmarks=(longmemeval locomo)
if [[ -n "${V554_INDEX_ONLY_BENCHMARK:-}" ]]; then
  benchmarks=("${V554_INDEX_ONLY_BENCHMARK}")
fi

runtime_config() {
  echo "${REPO}/configs/v5/runtime_v5_54_accuracy$1.json"
}

arm_flags() {
  case "$1" in
    seed_only) echo "--no-h10-traversal --no-hierarchical-routing" ;;
    hierarchy_only) echo "--no-h10-traversal" ;;
    flat_graph) echo "--no-hierarchical-routing" ;;
    full) echo "" ;;
    *) echo "unknown arm: $1" >&2; return 2 ;;
  esac
}

wait_local_models() {
  local delay=5
  until curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null \
      && curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null; do
    echo "local vLLM unavailable; waiting ${delay}s for managed restart"
    sleep "${delay}"
    if (( delay < 30 )); then delay=$((delay + 5)); fi
  done
}

prepare_arm() {
  local budget="$1" arm="$2" output="${ROOT}/turn${budget}/${arm}/prepare"
  local flags
  flags="$(arm_flags "${arm}")"
  mkdir -p "${output}"
  wait_local_models
  # shellcheck disable=SC2086
  "${PY}" "${REPO}/scripts/run_v5_6_answer.py" \
    --source-db "${GRAPH}" --output-root "${ROOT}" --run-root "${output}" \
    --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" \
    --config "${CONFIG}" --runtime-config "$(runtime_config "${budget}")" \
    --answer-policy v5_54 --max-output-tokens 2000 --full --prepare-only \
    --save-candidate-scores --navigate-workers 1 --checkpoint-every 200 \
    --label "v554_index_turn${budget}_${arm}" --resume ${flags}
}

answer_arm() {
  local budget="$1" arm="$2"
  local prepared="${ROOT}/turn${budget}/${arm}/prepare/prepared_answers.jsonl"
  local output="${ROOT}/turn${budget}/${arm}/answer"
  mkdir -p "${output}"
  while true; do
    wait_local_models
    if "${PY}" "${REPO}/scripts/replay_v5_prepared_answers.py" \
      --prepared "${prepared}" --metadata-answers "${BASELINE}/answers.jsonl" \
      --metadata-prompt-policy ignore --reuse-answers "${BASELINE}/answers.jsonl" \
      --reuse-usage "${BASELINE}/answer_usage.jsonl" \
      --source-db "${GRAPH}" --config "${CONFIG}" --output-root "${output}" \
      --answer-model "${ANSWER_MODEL}" --answer-base-url http://127.0.0.1:8002/v1 \
      --answer-api-key-env GRAPHMEM_LOCAL_API_KEY --answer-request-profile qwen \
      --packing-model "${ANSWER_MODEL}" --max-output-tokens 2000 \
      --workers "${ANSWER_WORKERS}" --checkpoint-every 256 --resume; then
      break
    fi
    echo "turn${budget}/${arm} answer interrupted; waiting and resuming"
    sleep 5
  done
}

judge_delta() {
  local budget="$1" arm="$2" benchmark="$3"
  local output="${ROOT}/turn${budget}/${arm}/answer"
  local delta="${output}/judge_${benchmark}/delta_answers.jsonl"
  local delta_manifest="${output}/judge_${benchmark}/delta_manifest.json"
  local delta_judge="${output}/judge_${benchmark}/delta"
  local merged="${output}/judge_${benchmark}/paired_verdicts.jsonl"
  local merged_manifest="${output}/judge_${benchmark}/paired_manifest.json"
  local lock_file="${output}/judge_${benchmark}/owner.lock" lock_fd
  mkdir -p "${output}/judge_${benchmark}"
  exec {lock_fd}>"${lock_file}"
  if ! flock -n "${lock_fd}"; then
    echo "turn${budget}/${arm}/${benchmark} already has a judge owner; skipping duplicate launcher"
    exec {lock_fd}>&-
    return 0
  fi
  "${PY}" "${REPO}/scripts/paired_judge_delta.py" prepare \
    --baseline-answers "${BASELINE}/answers.jsonl" \
    --candidate-answers "${output}/answers.jsonl" --benchmark "${benchmark}" \
    --output "${delta}" --manifest "${delta_manifest}"
  if [[ -s "${delta}" ]]; then
    if [[ -s "${delta_judge}/auto_eval.jsonl" ]]; then
      "${PY}" "${REPO}/scripts/deduplicate_judge_jsonl.py" \
        --input "${delta_judge}/auto_eval.jsonl" \
        --audit "${output}/judge_${benchmark}/resume_dedup_audit.json"
    fi
    if [[ "${benchmark}" == "longmemeval" ]]; then
      while ! "${PY}" "${REPO}/scripts/evaluate_mem0_judge.py" \
          --answers "${delta}" --output-dir "${delta_judge}" \
          --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
          --request-profile openai --workers "${JUDGE_WORKERS}" --resume; do
        echo "Luna unavailable for turn${budget}/${arm}/${benchmark}; waiting"
        sleep 15
      done
    else
      while ! "${PY}" "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
          --data "${LOCOMO}" --answers "${delta}" --output-dir "${delta_judge}" \
          --memory-benchmarks-repo "${MEMORY_BENCHMARKS}" \
          --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
          --request-profile openai --workers "${JUDGE_WORKERS}" --resume; do
        echo "Luna unavailable for turn${budget}/${arm}/${benchmark}; waiting"
        sleep 15
      done
    fi
  else
    mkdir -p "${delta_judge}"
    : > "${delta_judge}/auto_eval.jsonl"
  fi
  "${PY}" "${REPO}/scripts/paired_judge_delta.py" merge \
    --baseline-answers "${BASELINE}/answers.jsonl" \
    --candidate-answers "${output}/answers.jsonl" --benchmark "${benchmark}" \
    --baseline-judge "${BASELINE}/paired_judge_${benchmark}.jsonl" \
    --delta-judge "${delta_judge}/auto_eval.jsonl" \
    --output "${merged}" --manifest "${merged_manifest}"
  flock -u "${lock_fd}"
  exec {lock_fd}>&-
}

prepare_all() {
  local pids=()
  for budget in "${budgets[@]}"; do
    for arm in "${arms[@]}"; do
      prepare_arm "${budget}" "${arm}" &
      pids+=("$!")
    done
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
  "${PY}" "${REPO}/scripts/audit_v5_54_index_ablation.py" \
    prepare --root "${ROOT}" --expected 2040
}

answer_all() {
  for budget in "${budgets[@]}"; do
    for arm in "${arms[@]}"; do answer_arm "${budget}" "${arm}"; done
  done
}

judge_all() {
  for budget in "${budgets[@]}"; do
    for arm in "${arms[@]}"; do
      for benchmark in "${benchmarks[@]}"; do
        judge_delta "${budget}" "${arm}" "${benchmark}"
      done
    done
  done
}

summarize() {
  "${PY}" "${REPO}/scripts/audit_v5_54_index_ablation.py" \
    final --root "${ROOT}" --expected 2040
  "${PY}" "${REPO}/scripts/summarize_v5_54_index_ablation.py" \
    --root "${ROOT}" --output "${ROOT}/summary.json"
  "${PY}" "${REPO}/scripts/render_v5_54_index_ablation.py" \
    --summary "${ROOT}/summary.json" --report "${WORKSPACE}/GraphMem_report"
}

case "${PHASE}" in
  prepare) prepare_all ;;
  answer) answer_all ;;
  judge) judge_all ;;
  summarize) summarize ;;
  all) prepare_all; answer_all; judge_all; summarize ;;
  *) echo "unknown V554_INDEX_PHASE=${PHASE}" >&2; exit 2 ;;
esac
