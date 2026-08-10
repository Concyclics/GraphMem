#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO}")"
PY="${PYTHON_BIN:-${WORKSPACE}/.conda-envs/graphmem-v58/bin/python}"
ROOT="${V520_FULL_TYPE_ROOT:-${WORKSPACE}/artifacts/report/v5_20/full_type_graph_ablation}"
FULL64="${WORKSPACE}/artifacts/report/v5_20/full_budget_benchmark/turn64/answer"
HARD="${WORKSPACE}/artifacts/report/v5_20/graph_structure_ablation_dev200"
LME="${WORKSPACE}/artifacts/data/longmemeval_s_cleaned.json"
LOCOMO="${WORKSPACE}/artifacts/data/locomo10_graphmem.json"
GOLD="${REPO}/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"
CONFIG="${REPO}/configs/v5/v5_17_budget230.json"
TOKENIZER="${WORKSPACE}/artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json"
GRAPH_ROOT="${WORKSPACE}/artifacts/report/v5_19/full_benchmark"
GRAPH_DB="${GRAPH_ROOT}/graph/graphmem.sqlite"
DENSE="${GRAPH_ROOT}/dense_indexes"
QUERY_CACHE="${GRAPH_ROOT}/query_embeddings.sqlite"
MB="${WORKSPACE}/third_party/memory-benchmarks"

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi
export GRAPHMEM_TOKENIZER_PATH="${GRAPHMEM_TOKENIZER_PATH:-${TOKENIZER}}"
export PYTHONHASHSEED=0
mkdir -p "${ROOT}"

wait_local_models() {
  while ! curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null \
      || ! curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null; do
    echo "local vLLM unavailable; waiting for managed restart"
    sleep 10
  done
}

run_arm() {
  local arm="$1"
  shift
  local output="${ROOT}/${arm}/answer"
  mkdir -p "${output}"
  if [[ ! -f "${output}/answer_cache.sqlite" ]]; then
    cp "${HARD}/${arm}/answer/answer_cache.sqlite" "${output}/answer_cache.sqlite"
  fi
  while true; do
    wait_local_models
    if "${PY}" "${REPO}/scripts/run_v5_6_answer.py" \
      --source-db "${GRAPH_DB}" --output-root "${ROOT}/${arm}" --run-root "${output}" \
      --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" \
      --full --lme-type multi-session --lme-type temporal-reasoning \
      --locomo-category 1 --locomo-category 2 \
      --config "${CONFIG}" --profile h11 \
      --max-evidence-turns 64 --max-evidence-tokens 12000 \
      --max-answer-tokens 12000 --max-output-tokens 2000 \
      --span-pack-window 96 --obligation-aware-packing \
      --native-seed-fusion --queryir-soft-fallback \
      --source-time-normalization --precision-grounded-prompt \
      --graph-hop-decay 0.3 --expansion-beam 2 \
      --rare-lexical-relations --query-gated-rare-lexical \
      --embedding --dense-sidecar-dir "${DENSE}" --dense-backend faiss_flat \
      --query-embedding-cache "${QUERY_CACHE}" \
      --answer-workers "${V520_FULL_TYPE_ANSWER_WORKERS_PER_ARM:-64}" \
      --checkpoint-every 100 --label "v520_full_type_${arm}" --resume "$@"; then
      break
    fi
    echo "${arm} interrupted; waiting and resuming"
    sleep 10
  done
}

judge_arm() {
  local arm="$1"
  local output="${ROOT}/${arm}/answer"
  while ! "${PY}" "${REPO}/scripts/evaluate_mem0_judge.py" \
      --answers "${output}/answers_longmemeval.jsonl" \
      --output-dir "${output}/judge_lme" --model gpt-5.6-luna \
      --api-key-env SGAO_API_KEY --request-profile openai --workers 32 --resume; do
    sleep 15
  done
  while ! "${PY}" "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
      --data "${LOCOMO}" --answers "${output}/answers_locomo.jsonl" \
      --output-dir "${output}/judge_locomo" --memory-benchmarks-repo "${MB}" \
      --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
      --request-profile openai --workers 32 --resume; do
    sleep 15
  done
}

"${PY}" "${REPO}/scripts/materialize_v5_20_full_type_control.py" \
  --source "${FULL64}" --output "${ROOT}/topology_layout/answer"

run_arm seed_only --no-h10-traversal --no-hierarchical-routing --evidence-order adaptive &
p1=$!
run_arm flat_graph --no-hierarchical-routing --evidence-order adaptive &
p2=$!
run_arm hierarchical --evidence-order adaptive &
p3=$!
run_arm graph_rerank_layout --evidence-order topological_plain &
p4=$!
wait "${p1}" "${p2}" "${p3}" "${p4}"

judge_arm seed_only & j1=$!
judge_arm flat_graph & j2=$!
judge_arm hierarchical & j3=$!
judge_arm graph_rerank_layout & j4=$!
wait "${j1}" "${j2}" "${j3}" "${j4}"

"${PY}" "${REPO}/scripts/summarize_v5_20_graph_ablation.py" \
  --root "${ROOT}" --expected-questions 869 --output "${ROOT}/summary.json"
