#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO}")"
PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE}/.conda-envs/graphmem-v58/bin/python}"
ROOT="${V520_GRAPH_ABLATION_ROOT:-${WORKSPACE}/artifacts/report/v5_20/graph_structure_ablation_dev200}"
DATA="${WORKSPACE}/artifacts/development_sets/hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804"
LME="${DATA}/longmemeval_hard_multisession50_temporal50.json"
LOCOMO="${DATA}/locomo_hard_cat1_multihop50_cat2_temporal50.json"
GOLD="${REPO}/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"
CONFIG="${REPO}/configs/v5/v5_17_budget230.json"
TOKENIZER="${WORKSPACE}/artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json"
GRAPH_ROOT="${WORKSPACE}/artifacts/report/v5_19/full_benchmark"
GRAPH_DB="${GRAPH_ROOT}/graph/graphmem.sqlite"
DENSE="${GRAPH_ROOT}/dense_indexes"
QUERY_CACHE="${GRAPH_ROOT}/query_embeddings.sqlite"
MEMORY_BENCHMARKS="${WORKSPACE}/third_party/memory-benchmarks"

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi
export GRAPHMEM_TOKENIZER_PATH="${GRAPHMEM_TOKENIZER_PATH:-${TOKENIZER}}"
export PYTHONHASHSEED=0

for path in "${PYTHON_BIN}" "${LME}" "${LOCOMO}" "${GOLD}" "${CONFIG}" \
            "${GRAPHMEM_TOKENIZER_PATH}" "${GRAPH_DB}" "${DENSE}" \
            "${QUERY_CACHE}" "${MEMORY_BENCHMARKS}"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done
mkdir -p "${ROOT}"

wait_local_models() {
  local delay=5
  while true; do
    if curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null \
        && curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null; then
      return 0
    fi
    echo "local vLLM unavailable; waiting ${delay}s for the managed restart"
    sleep "${delay}"
    if (( delay < 30 )); then delay=$((delay + 5)); fi
  done
}

run_arm() {
  local arm="$1"
  shift
  local arm_root="${ROOT}/${arm}"
  mkdir -p "${arm_root}"
  while true; do
    wait_local_models
    if "${PYTHON_BIN}" "${REPO}/scripts/run_v5_6_answer.py" \
      --source-db "${GRAPH_DB}" --output-root "${arm_root}" \
      --run-root "${arm_root}/answer" \
      --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" \
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
      --answer-workers "${V520_HARD_ANSWER_WORKERS:-64}" --checkpoint-every 200 \
      --label "v520_graph_${arm}_accuracy64" --resume "$@"; then
      break
    fi
    echo "${arm} answer interrupted; waiting for managed vLLM and resuming"
    sleep 5
  done

  "${PYTHON_BIN}" "${REPO}/scripts/evaluate_mem0_judge.py" \
    --answers "${arm_root}/answer/answers_longmemeval.jsonl" \
    --output-dir "${arm_root}/answer/judge_lme" \
    --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
    --request-profile openai --workers 100 --resume
  "${PYTHON_BIN}" "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
    --data "${LOCOMO}" \
    --answers "${arm_root}/answer/answers_locomo.jsonl" \
    --output-dir "${arm_root}/answer/judge_locomo" \
    --memory-benchmarks-repo "${MEMORY_BENCHMARKS}" \
    --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
    --request-profile openai --workers 100 --resume
}

# All arms share one frozen graph, dense vectors, QueryIR, 64-turn/12K pack,
# answer model and judge.  Each row introduces exactly one structural mechanism.
run_arm seed_only --no-h10-traversal --no-hierarchical-routing \
  --evidence-order adaptive
run_arm flat_graph --no-hierarchical-routing --evidence-order adaptive
run_arm hierarchical --evidence-order adaptive
run_arm topology_layout --evidence-order topological

"${PYTHON_BIN}" "${REPO}/scripts/summarize_v5_20_graph_ablation.py" \
  --root "${ROOT}" --output "${ROOT}/summary.json"
