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
ARM="graph_rerank_layout"
ARM_ROOT="${ROOT}/${ARM}"

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi
export GRAPHMEM_TOKENIZER_PATH="${GRAPHMEM_TOKENIZER_PATH:-${TOKENIZER}}"
export PYTHONHASHSEED=0
mkdir -p "${ARM_ROOT}"

wait_local_models() {
  while ! curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null \
      || ! curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null; do
    echo "local vLLM unavailable; waiting for managed restart"
    sleep 10
  done
}

while true; do
  wait_local_models
  if "${PYTHON_BIN}" "${REPO}/scripts/run_v5_6_answer.py" \
    --source-db "${GRAPH_DB}" --output-root "${ARM_ROOT}" \
    --run-root "${ARM_ROOT}/answer" \
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
    --evidence-order topological_plain \
    --answer-workers "${V520_HARD_ANSWER_WORKERS:-64}" --checkpoint-every 200 \
    --label "v520_graph_${ARM}_accuracy64" --resume; then
    break
  fi
  echo "${ARM} answer interrupted; waiting and resuming"
  sleep 10
done

while true; do
  if "${PYTHON_BIN}" "${REPO}/scripts/evaluate_mem0_judge.py" \
    --answers "${ARM_ROOT}/answer/answers_longmemeval.jsonl" \
    --output-dir "${ARM_ROOT}/answer/judge_lme" \
    --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
    --request-profile openai --workers 32 --resume; then
    break
  fi
  echo "${ARM} LME judge unavailable; waiting and resuming"
  sleep 15
done

while true; do
  if "${PYTHON_BIN}" "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
    --data "${LOCOMO}" --answers "${ARM_ROOT}/answer/answers_locomo.jsonl" \
    --output-dir "${ARM_ROOT}/answer/judge_locomo" \
    --memory-benchmarks-repo "${MEMORY_BENCHMARKS}" \
    --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
    --request-profile openai --workers 32 --resume; then
    break
  fi
  echo "${ARM} LoCoMo judge unavailable; waiting and resuming"
  sleep 15
done

"${PYTHON_BIN}" "${REPO}/scripts/summarize_v5_20_graph_ablation.py" \
  --root "${ROOT}" --output "${ROOT}/summary.json"
