#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${STATE_DIR:-/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5/gate_a_services_20260804}"
VLLM_BIN="${VLLM_BIN:-/home/chenhan/miniconda3/envs/vllm/bin/vllm}"
HF_CACHE_ROOT="${HF_HOME:-/home/chenhan/.cache/huggingface}"
EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-${HF_CACHE_ROOT}/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-${HF_CACHE_ROOT}/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507-FP8/snapshots/5a5a776300a41aaa681dd7ff0106608ef2bc90db}"
EMBED_SERVED_NAME="Qwen/Qwen3-Embedding-0.6B"
LLM_SERVED_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
EMBED_GPU="${EMBED_GPU:-1}"
LLM_GPUS="${LLM_GPUS:-2,3}"
LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN:-65536}"

mkdir -p "${STATE_DIR}"

ready() {
  curl -fsS --max-time 3 "http://127.0.0.1:$1/v1/models" >/dev/null
}

start_one() {
  local session="$1" gpu="$2" logfile="$3"
  shift 3
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "${session} already exists" >&2
    return 1
  fi
  tmux new-session -d -s "${session}" \
    "cd '${REPO}' && exec env CUDA_VISIBLE_DEVICES='${gpu}' HF_HUB_OFFLINE=1 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel '${VLLM_BIN}' serve $* >'${logfile}' 2>&1"
}

run_embedding_supervisor() {
  while true; do
    printf '%s embedding supervisor starting vLLM\n' "$(date -Is)" >>"${STATE_DIR}/embedding.log"
    set +e
    env CUDA_VISIBLE_DEVICES="${EMBED_GPU}" HF_HUB_OFFLINE=1 \
      VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel \
      "${VLLM_BIN}" serve "${EMBED_MODEL_PATH}" \
      --served-model-name "${EMBED_SERVED_NAME}" --runner pooling \
      --host 127.0.0.1 --port 8001 --max-model-len 32768 \
      --gpu-memory-utilization 0.10 --trust-remote-code \
      >>"${STATE_DIR}/embedding.log" 2>&1
    local exit_code=$?
    set -e
    printf '%s embedding vLLM exited code=%s; restarting in 5s\n' \
      "$(date -Is)" "${exit_code}" >>"${STATE_DIR}/embedding.log"
    sleep 5
  done
}

run_llm_supervisor() {
  while true; do
    printf '%s llm supervisor starting vLLM\n' "$(date -Is)" >>"${STATE_DIR}/llm.log"
    set +e
    env CUDA_VISIBLE_DEVICES="${LLM_GPUS}" HF_HUB_OFFLINE=1 \
      VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
      VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel \
      "${VLLM_BIN}" serve "${LLM_MODEL_PATH}" \
      --served-model-name "${LLM_SERVED_NAME}" --host 127.0.0.1 --port 8002 \
      --tensor-parallel-size 2 --max-model-len "${LLM_MAX_MODEL_LEN}" \
      --gpu-memory-utilization 0.88 --max-num-seqs 128 \
      --max-num-batched-tokens 65536 --trust-remote-code \
      >>"${STATE_DIR}/llm.log" 2>&1
    local exit_code=$?
    set -e
    printf '%s llm vLLM exited code=%s; restarting in 5s\n' \
      "$(date -Is)" "${exit_code}" >>"${STATE_DIR}/llm.log"
    sleep 5
  done
}

wait_one() {
  local session="$1" port="$2" logfile="$3" attempt
  for attempt in $(seq 1 120); do
    if ready "${port}"; then
      echo "${session} ready on ${port}"
      return 0
    fi
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      tail -n 80 "${logfile}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "${session} did not become ready; see ${logfile}" >&2
  return 1
}

start_services() {
  [[ -x "${VLLM_BIN}" ]] || { echo "missing vllm: ${VLLM_BIN}" >&2; return 1; }
  [[ -d "${EMBED_MODEL_PATH}" ]] || { echo "missing embedding snapshot" >&2; return 1; }
  [[ -d "${LLM_MODEL_PATH}" ]] || { echo "missing LLM snapshot" >&2; return 1; }
  start_embedding
  start_heartbeat
  start_llm
  start_llm_heartbeat
}

start_embedding() {
  if ready 8001; then
    tmux has-session -t graphmem-v5-embed 2>/dev/null || {
      echo "port 8001 belongs to an unmanaged service" >&2; return 1; }
    echo "graphmem-v5-embed already ready"
    return 0
  fi
  tmux new-session -d -s graphmem-v5-embed \
    "cd '${REPO}' && exec env HF_HOME='${HF_CACHE_ROOT}' EMBED_GPU='${EMBED_GPU}' STATE_DIR='${STATE_DIR}' VLLM_BIN='${VLLM_BIN}' EMBED_MODEL_PATH='${EMBED_MODEL_PATH}' bash scripts/v5_gate_a_model_services.sh supervise-embed"
  wait_one graphmem-v5-embed 8001 "${STATE_DIR}/embedding.log"
}

start_heartbeat() {
  ready 8001 || { echo "embedding service is not ready" >&2; return 1; }
  if tmux has-session -t graphmem-v5-embed-heartbeat 2>/dev/null; then
    echo "graphmem-v5-embed-heartbeat already running"
    return 0
  fi
  tmux new-session -d -s graphmem-v5-embed-heartbeat \
    "cd '${REPO}' && exec /home/chenhan/miniconda3/envs/agent/bin/python scripts/v5_embedding_heartbeat.py --log '${STATE_DIR}/embedding_heartbeat.jsonl'"
}

start_llm() {
  [[ -d "${LLM_MODEL_PATH}" ]] || { echo "missing LLM snapshot: ${LLM_MODEL_PATH}" >&2; return 1; }
  if ready 8002; then
    tmux has-session -t graphmem-v5-llm 2>/dev/null || {
      echo "port 8002 belongs to an unmanaged service" >&2; return 1; }
    echo "graphmem-v5-llm already ready"
    return 0
  fi
  tmux new-session -d -s graphmem-v5-llm \
    "cd '${REPO}' && exec env HF_HOME='${HF_CACHE_ROOT}' LLM_GPUS='${LLM_GPUS}' STATE_DIR='${STATE_DIR}' VLLM_BIN='${VLLM_BIN}' LLM_MODEL_PATH='${LLM_MODEL_PATH}' LLM_MAX_MODEL_LEN='${LLM_MAX_MODEL_LEN}' bash scripts/v5_gate_a_model_services.sh supervise-llm"
  wait_one graphmem-v5-llm 8002 "${STATE_DIR}/llm.log"
}

start_llm_heartbeat() {
  ready 8002 || { echo "LLM service is not ready" >&2; return 1; }
  if tmux has-session -t graphmem-v5-llm-heartbeat 2>/dev/null; then
    echo "graphmem-v5-llm-heartbeat already running"
    return 0
  fi
  tmux new-session -d -s graphmem-v5-llm-heartbeat \
    "cd '${REPO}' && exec /home/chenhan/miniconda3/envs/agent/bin/python scripts/v5_llm_heartbeat.py --log '${STATE_DIR}/llm_heartbeat.jsonl'"
}

stop_services() {
  local session
  for session in graphmem-v5-embed-heartbeat graphmem-v5-llm-heartbeat graphmem-v5-embed graphmem-v5-llm; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      tmux kill-session -t "${session}"
    fi
  done
}

status_services() {
  tmux list-sessions 2>/dev/null || true
  curl -fsS --max-time 3 http://127.0.0.1:8001/v1/models || true
  curl -fsS --max-time 3 http://127.0.0.1:8002/v1/models || true
}

case "${1:-}" in
  start) start_services ;;
  start-embed) start_embedding ;;
  start-llm) start_llm ;;
  start-llm-heartbeat) start_llm_heartbeat ;;
  start-heartbeat) start_heartbeat ;;
  stop) stop_services ;;
  status) status_services ;;
  supervise-embed) run_embedding_supervisor ;;
  supervise-llm) run_llm_supervisor ;;
  *) echo "usage: $0 {start|start-embed|start-llm|start-heartbeat|start-llm-heartbeat|stop|status}" >&2; exit 2 ;;
esac
