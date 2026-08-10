#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO}")"
PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE}/.conda-envs/graphmem-v58/bin/python}"
ROOT="${V519_ABLATION_ROOT:-${WORKSPACE}/artifacts/report/v5_19/attribute_ablation_dev200}"
DATA="${WORKSPACE}/artifacts/development_sets/hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804"
LME="${DATA}/longmemeval_hard_multisession50_temporal50.json"
LOCOMO="${DATA}/locomo_hard_cat1_multihop50_cat2_temporal50.json"
GOLD="${REPO}/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"
CONFIG="${REPO}/configs/v5/v5_17_budget230.json"
TOKENIZER="${WORKSPACE}/artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json"
MEMORY_BENCHMARKS="${WORKSPACE}/third_party/memory-benchmarks"
REPORT_REPO="${WORKSPACE}/GraphMem_report"

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi
export GRAPHMEM_TOKENIZER_PATH="${GRAPHMEM_TOKENIZER_PATH:-${TOKENIZER}}"
export PYTHONHASHSEED=0

for path in "${PYTHON_BIN}" "${LME}" "${LOCOMO}" "${GOLD}" "${CONFIG}" "${GRAPHMEM_TOKENIZER_PATH}"; do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 2; }
done
mkdir -p "${ROOT}"

declare -A SIGNALS=(
  [full]="scene_similar,shared_entity,state_compatible,collection_related,temporal_near,lexical_rare"
  [no_scene]="shared_entity,state_compatible,collection_related,temporal_near,lexical_rare"
  [no_entity_family]="scene_similar,temporal_near,lexical_rare"
  [no_temporal]="scene_similar,shared_entity,state_compatible,collection_related,lexical_rare"
  [no_lexical]="scene_similar,shared_entity,state_compatible,collection_related,temporal_near"
  [semantic_only]="scene_similar"
)
ARMS=(full no_scene no_entity_family no_temporal no_lexical semantic_only)

wait_local_models() {
  local delay=5
  while true; do
    if curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null \
        && curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null; then
      return 0
    fi
    echo "local vLLM unavailable; waiting ${delay}s for the managed service restart"
    sleep "${delay}"
    if (( delay < 30 )); then delay=$((delay + 5)); fi
  done
}

wait_local_models

arm_ready() {
  "${PYTHON_BIN}" -c '
import json, sqlite3, sys
from pathlib import Path
root = Path(sys.argv[1])
report = root / "build_report.json"
db = root / "graph" / "graphmem.sqlite"
if not report.exists() or not db.exists():
    print(0); raise SystemExit
payload = json.loads(report.read_text()).get("summary", {})
with sqlite3.connect(db) as connection:
    built = connection.execute(
        "SELECT COUNT(*) FROM graph_versions WHERE graph_checksum != ?", ("",)
    ).fetchone()[0]
print(int(payload.get("memories_built") == 110 and not payload.get("failures") and built == 110))
' "$1"
}

build_arm() {
  local arm="$1"
  local arm_root="${ROOT}/${arm}"
  local frozen_args=()
  if [[ "${arm}" != "full" ]]; then
    frozen_args+=(
      --frozen-semantic-cache-only
      --frozen-semantic-source-report "${ROOT}/full/build_report.json"
    )
  fi
  mkdir -p "${arm_root}/graph"
  "${PYTHON_BIN}" "${REPO}/scripts/run_v5_6_full_build.py" \
    --target-db "${arm_root}/graph/graphmem.sqlite" \
    --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" \
    --config "${CONFIG}" --profile b5 --development-set \
    --embedding \
    --relation-embedding-db "${arm_root}/graph/relation_embeddings.sqlite" \
    --memory-workers 16 --max-concurrency 256 \
    --enabled-relation-signals "${SIGNALS[${arm}]}" \
    "${frozen_args[@]}" \
    --report "${arm_root}/build_report.json"
}

# The canonical arm is a true cold build.  Do not rewrite its report when a
# later control arm is resumed: the report is also the all-510 seed ledger and
# must continue to state that 110 Memories were built cold in this run.
full_ready="$(arm_ready "${ROOT}/full")"
if [[ "${full_ready}" != "1" ]]; then
  build_arm full
else
  echo "Full cold arm already complete; preserving its build report"
fi
# Every control starts from its frozen extraction/vector caches only; no graph
# table survives the clone.
for arm in no_scene no_entity_family no_temporal no_lexical semantic_only; do
  arm_root="${ROOT}/${arm}"
  if [[ ! -e "${arm_root}/graph/graphmem.sqlite" ]]; then
    "${PYTHON_BIN}" "${REPO}/scripts/prepare_v5_19_ablation_arm.py" \
      --source-db "${ROOT}/full/graph/graphmem.sqlite" \
      --target-db "${arm_root}/graph/graphmem.sqlite" \
      --relation-source "${ROOT}/full/graph/relation_embeddings.sqlite" \
      --relation-target "${arm_root}/graph/relation_embeddings.sqlite" \
      --manifest "${arm_root}/cache_seed_manifest.json"
  fi
  if [[ "$(arm_ready "${arm_root}")" == "1" ]]; then
    echo "${arm} build already complete; preserving its build report"
  else
    build_arm "${arm}"
  fi
done

for arm in "${ARMS[@]}"; do
  arm_root="${ROOT}/${arm}"
  "${PYTHON_BIN}" "${REPO}/scripts/precompile_dense_indexes.py" \
    --db "${arm_root}/graph/graphmem.sqlite" --config "${CONFIG}" \
    --output "${arm_root}/dense_indexes" --backend faiss_flat --workers 16

  lexical_args=()
  if [[ "${SIGNALS[${arm}]}" == *lexical_rare* ]]; then
    lexical_args+=(--rare-lexical-relations)
  fi
  while true; do
    wait_local_models
    if "${PYTHON_BIN}" "${REPO}/scripts/run_v5_6_answer.py" \
      --source-db "${arm_root}/graph/graphmem.sqlite" \
      --output-root "${arm_root}" --run-root "${arm_root}/answer" \
      --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" \
      --config "${CONFIG}" --profile h11 \
      --max-evidence-turns 64 --max-evidence-tokens 12000 \
      --span-pack-window 96 --obligation-aware-packing \
      --native-seed-fusion --queryir-soft-fallback \
      --source-time-normalization --graph-hop-decay 0.3 --expansion-beam 2 \
      --embedding --dense-sidecar-dir "${arm_root}/dense_indexes" \
      --dense-backend faiss_flat \
      --query-embedding-cache "${ROOT}/query_embeddings.sqlite" \
      --answer-workers 32 --checkpoint-every 25 --label "v519_${arm}_accuracy64" \
      --resume "${lexical_args[@]}"; then
      break
    fi
    echo "${arm} answer interrupted; waiting for vLLM and resuming checkpoint"
    sleep 5
  done

  "${PYTHON_BIN}" "${REPO}/scripts/evaluate_mem0_judge.py" \
    --answers "${arm_root}/answer/answers_longmemeval.jsonl" \
    --output-dir "${arm_root}/answer/judge_lme" \
    --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
    --request-profile openai --workers 32 --resume
  "${PYTHON_BIN}" "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
    --data "${LOCOMO}" \
    --answers "${arm_root}/answer/answers_locomo.jsonl" \
    --output-dir "${arm_root}/answer/judge_locomo" \
    --memory-benchmarks-repo "${MEMORY_BENCHMARKS}" \
    --model gpt-5.6-luna --api-key-env SGAO_API_KEY \
    --request-profile openai --workers 32 --resume
done

"${PYTHON_BIN}" "${REPO}/scripts/summarize_v5_19_attribute_ablation.py" \
  --root "${ROOT}" --output "${ROOT}/ablation_summary.json"
"${PYTHON_BIN}" "${REPO}/scripts/audit_v5_19_experiment.py" ablation \
  --root "${ROOT}" --expected-questions 200
"${PYTHON_BIN}" "${REPO}/scripts/render_v5_19_report_assets.py" \
  --ablation "${ROOT}/ablation_summary.json" --report "${REPORT_REPO}"
