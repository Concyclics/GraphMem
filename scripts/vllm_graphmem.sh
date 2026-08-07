#!/usr/bin/env bash
# Manual lifecycle entry point for GraphMem's local vLLM services.
#
# Defaults deliberately match the current shared deployment:
#   embedding: GPU 1, port 8001, gpu_memory_utilization=0.10
#   LLM:       GPU 2,3 TP=2, port 8002, max_num_seqs=384
# Override only when needed, for example:
#   LLM_MAX_NUM_SEQS=256 ./scripts/vllm_graphmem.sh restart

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_SCRIPT="${REPO}/scripts/v5_gate_a_model_services.sh"

STATE_DIR="${STATE_DIR:-/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5_4/services}"
HF_HOME="${HF_HOME:-/ssd1/chenhan/huggingface}"
EMBED_GPU="${EMBED_GPU:-1}"
LLM_GPUS="${LLM_GPUS:-2,3}"
LLM_TP_SIZE="${LLM_TP_SIZE:-2}"
LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN:-65536}"
LLM_MAX_NUM_SEQS="${LLM_MAX_NUM_SEQS:-384}"

usage() {
  cat <<'EOF'
Usage: scripts/vllm_graphmem.sh <command>

Commands:
  start             Start embedding, embedding heartbeat, LLM, and LLM heartbeat.
  stop              Stop all GraphMem-managed services and heartbeats.
  restart           Stop all, then start all with the configured parameters.
  status            Show tmux sessions and probe ports 8001/8002.
  start-embed       Start embedding service only.
  start-llm         Start 30B service only.
  start-heartbeat   Start embedding heartbeat only.
  start-llm-heartbeat
                    Start LLM heartbeat only.

Current defaults:
  embedding GPU=1; LLM GPUs=2,3 TP=2; max-model-len=65536;
  LLM max-num-seqs=384; LLM gpu-memory-utilization=0.88.

Environment overrides: STATE_DIR, HF_HOME, EMBED_GPU, LLM_GPUS,
LLM_TP_SIZE, LLM_MAX_MODEL_LEN, LLM_MAX_NUM_SEQS.
EOF
}

command="${1:-}"
case "${command}" in
  start|stop|status|start-embed|start-llm|start-heartbeat|start-llm-heartbeat)
    exec env \
      STATE_DIR="${STATE_DIR}" HF_HOME="${HF_HOME}" EMBED_GPU="${EMBED_GPU}" \
      LLM_GPUS="${LLM_GPUS}" LLM_TP_SIZE="${LLM_TP_SIZE}" \
      LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN}" LLM_MAX_NUM_SEQS="${LLM_MAX_NUM_SEQS}" \
      bash "${SERVICE_SCRIPT}" "${command}"
    ;;
  restart)
    env \
      STATE_DIR="${STATE_DIR}" HF_HOME="${HF_HOME}" EMBED_GPU="${EMBED_GPU}" \
      LLM_GPUS="${LLM_GPUS}" LLM_TP_SIZE="${LLM_TP_SIZE}" \
      LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN}" LLM_MAX_NUM_SEQS="${LLM_MAX_NUM_SEQS}" \
      bash "${SERVICE_SCRIPT}" stop
    exec env \
      STATE_DIR="${STATE_DIR}" HF_HOME="${HF_HOME}" EMBED_GPU="${EMBED_GPU}" \
      LLM_GPUS="${LLM_GPUS}" LLM_TP_SIZE="${LLM_TP_SIZE}" \
      LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN}" LLM_MAX_NUM_SEQS="${LLM_MAX_NUM_SEQS}" \
      bash "${SERVICE_SCRIPT}" start
    ;;
  help|-h|--help|"")
    usage
    ;;
  *)
    echo "Unknown command: ${command}" >&2
    usage >&2
    exit 2
    ;;
esac
