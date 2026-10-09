#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO}")"
PY="${PYTHON_BIN:-${WORKSPACE}/.conda-envs/graphmem-v58/bin/python}"
ROOT="${V572_ROOT:-${WORKSPACE}/artifacts/report/v5_72_local_locomo_best_of_8}"
LME="${WORKSPACE}/artifacts/data/longmemeval_s_cleaned.json"
LOCOMO="${WORKSPACE}/artifacts/data/locomo10_graphmem.json"
GOLD="${REPO}/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"
CONFIG="${REPO}/configs/v5/v5_57_lossless_atomic.json"
RUNTIME="${REPO}/configs/v5/runtime_v5_59_hybrid64.json"
TOKENIZER="${WORKSPACE}/artifacts/model_assets/qwen3_30b_a3b_instruct_2507_fp8/tokenizer.json"
MEMORY_BENCHMARKS="${WORKSPACE}/third_party/memory-benchmarks"
GRAPH_DB="${ROOT}/graph/graphmem.sqlite"
RELATION_DB="${ROOT}/graph/relation_embeddings.sqlite"
BUILD_REPORT="${ROOT}/build_report.json"
LLM_URL="http://127.0.0.1:8002/v1"
EMBED_URL="http://127.0.0.1:8003/v1"
LLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
EMBED_MODEL="BAAI/bge-m3"

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

wait_local_models() {
  local delay=5
  until curl -fsS --max-time 5 "${LLM_URL}/models" >/dev/null \
      && curl -fsS --max-time 5 "${EMBED_URL}/models" >/dev/null; do
    event "local model service unavailable; waiting ${delay}s for managed restart"
    sleep "${delay}"
    if (( delay < 30 )); then delay=$((delay + 5)); fi
  done
}

MEMORY_IDS="${ROOT}/locomo_memory_ids.json"
if [[ ! -f "${MEMORY_IDS}" ]]; then
  "${PY}" - "${LOCOMO}" "${MEMORY_IDS}" <<'PY'
import json
import sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
rows = json.loads(source.read_text(encoding="utf-8"))
memory_ids = sorted({"locomo:" + str(row["locomo_sample_id"]) for row in rows})
if len(memory_ids) != 10:
    raise RuntimeError(f"expected 10 LoCoMo memories, got {len(memory_ids)}")
target.write_text(json.dumps(memory_ids, indent=2) + "\n", encoding="utf-8")
PY
fi

run_build() {
  wait_local_models
  "${PY}" "${REPO}/scripts/run_v5_6_full_build.py" \
    --target-db "${GRAPH_DB}" --relation-embedding-db "${RELATION_DB}" \
    --memory-id-file "${MEMORY_IDS}" \
    --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" \
    --config "${CONFIG}" --profile b5 \
    --llm-model "${LLM_MODEL}" --llm-base-url "${LLM_URL}" \
    --llm-request-profile qwen --llm-request-timeout-seconds 1800 \
    --embedding --embedding-model "${EMBED_MODEL}" \
    --embedding-base-url "${EMBED_URL}" --embedding-request-model bge-m3 \
    --memory-workers 10 --max-concurrency 256 \
    --require-complete-diagnostics --report "${BUILD_REPORT}"
}

until run_build >>"${ROOT}/build.log" 2>&1; do
  event "LoCoMo build interrupted; preserving graph/cache and resuming"
  sleep 20
done
"${PY}" - "${BUILD_REPORT}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
rows = payload.get("rows", [])
summary = payload.get("summary", {})
assert len(rows) == 10, len(rows)
assert not summary.get("failures"), summary.get("failures")
assert summary.get("token_ledger_memories") == 10, summary.get("token_ledger_memories")
PY
event "10/10 LoCoMo conversations rebuilt with local Qwen and BGE-M3"

if [[ ! -f "${ROOT}/dense_indexes/manifest.json" ]]; then
  "${PY}" "${REPO}/scripts/precompile_dense_indexes.py" \
    --db "${GRAPH_DB}" --config "${CONFIG}" --model-id "${EMBED_MODEL}" \
    --output "${ROOT}/dense_indexes" --backend faiss_flat --workers 10 \
    >"${ROOT}/dense_precompile.log" 2>&1
fi
event "10 per-conversation FAISS indexes compiled"

PREPARE="${ROOT}/prepare"
until wait_local_models && "${PY}" "${REPO}/scripts/run_v5_6_answer.py" \
    --source-db "${GRAPH_DB}" --relation-embedding-db "${RELATION_DB}" \
    --output-root "${ROOT}" --run-root "${PREPARE}" \
    --lme "${LME}" --locomo "${LOCOMO}" --gold "${GOLD}" --full \
    --lme-type __locomo_only__ \
    --locomo-category 1 --locomo-category 2 \
    --locomo-category 3 --locomo-category 4 \
    --config "${CONFIG}" --runtime-config "${RUNTIME}" \
    --answer-policy v5_63 --embedding \
    --embedding-model "${EMBED_MODEL}" --embedding-base-url "${EMBED_URL}" \
    --embedding-request-model bge-m3 \
    --dense-sidecar-dir "${ROOT}/dense_indexes" \
    --compiled-cache-dir "${ROOT}/compiled_memory_views" \
    --query-embedding-cache "${ROOT}/query_embedding_cache.sqlite" \
    --prepare-only --navigate-workers 32 --checkpoint-every 50 --resume \
    >>"${ROOT}/prepare.log" 2>&1; do
  event "LoCoMo retrieval preparation interrupted; resuming"
  sleep 20
done
event "1,540 frozen 64-turn LoCoMo prompts prepared"

ANSWER="${ROOT}/answer_n8"
until wait_local_models && "${PY}" "${REPO}/scripts/replay_v5_prepared_best_of.py" \
    --prepared "${PREPARE}/prepared_answers.jsonl" \
    --locomo-data "${LOCOMO}" --output-root "${ANSWER}" \
    --model "${LLM_MODEL}" --base-url "${LLM_URL}" \
    --n 8 --temperature 0.7 --top-p 0.8 --top-k 20 --seed 7208 \
    --max-output-tokens 2000 --workers 48 --resume \
    >>"${ROOT}/answer_n8.log" 2>&1; do
  event "local n=8 answer pass interrupted; resuming"
  sleep 20
done
event "1,540 local n=8 answer requests complete"

JUDGE_ROOT="${ROOT}/judge_luna_medium"
for candidate_index in 1 2 3 4 5 6 7 8; do
  INPUT="${ROOT}/judge_inputs/candidate_${candidate_index}.jsonl"
  "${PY}" "${REPO}/scripts/materialize_locomo_best_of_judge_prefix.py" \
    --candidates "${ANSWER}/candidates.jsonl" --judge-root "${JUDGE_ROOT}" \
    --candidate-index "${candidate_index}" --output "${INPUT}" \
    >>"${ROOT}/judge_materialize.log" 2>&1
  until "${PY}" "${REPO}/scripts/evaluate_memory_benchmarks_locomo_judge.py" \
      --data "${LOCOMO}" --answers "${INPUT}" \
      --output-dir "${JUDGE_ROOT}/candidate_${candidate_index}" \
      --memory-benchmarks-repo "${MEMORY_BENCHMARKS}" \
      --model "${LUNA_MODEL}" --base-url "${SGAO_BASE_URL}" \
      --api-key-env SGAO_API_KEY --request-profile openai \
      --reasoning-effort medium --max-tokens 2048 --workers 64 --resume \
      >>"${ROOT}/judge_candidate_${candidate_index}.log" 2>&1; do
    event "candidate ${candidate_index} Luna judge interrupted; resuming"
    sleep 20
  done
  event "Best-of-${candidate_index} prefix judged"
done

"${PY}" "${REPO}/scripts/summarize_locomo_local_best_of.py" \
  --candidates "${ANSWER}/candidates.jsonl" \
  --answer-manifest "${ANSWER}/run_manifest.json" \
  --build-report "${BUILD_REPORT}" --graph-db "${GRAPH_DB}" \
  --relation-db "${RELATION_DB}" --judge-root "${JUDGE_ROOT}" \
  --output "${ROOT}/summary.json" | tee "${ROOT}/summary.txt"
event "local LoCoMo Best-of-1..8 benchmark complete"
