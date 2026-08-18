#!/usr/bin/env bash
# Manual lifecycle entry point for GraphMem's local vLLM services.
#
# Defaults deliberately match the current shared deployment:
#   embedding: GPU 1, port 8001 (or next free port), gpu_memory_utilization=0.10
#   LLM:       one automatically selected idle GPU, port 8002 (or next free), TP=1
# Override only when needed, for example:
#   LLM_MAX_NUM_SEQS=256 ./scripts/vllm_graphmem.sh restart

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_SCRIPT="${REPO}/scripts/v5_gate_a_model_services.sh"

STATE_DIR="${STATE_DIR:-/ssd3/chenhan/Spark_MemGraph_Dev/artifacts/v5_4/services}"
HF_HOME="${HF_HOME:-/ssd1/chenhan/huggingface}"
EMBED_GPU="${EMBED_GPU:-1}"
LLM_GPU="${LLM_GPU:-auto}"
EMBED_PORT="${EMBED_PORT:-8001}"
LLM_PORT="${LLM_PORT:-8002}"
PORT_SEARCH_LIMIT="${PORT_SEARCH_LIMIT:-100}"
LLM_IDLE_MAX_MEMORY_MIB="${LLM_IDLE_MAX_MEMORY_MIB:-512}"
LLM_IDLE_MAX_UTILIZATION="${LLM_IDLE_MAX_UTILIZATION:-5}"
LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN:-65536}"
LLM_MAX_NUM_SEQS="${LLM_MAX_NUM_SEQS:-384}"

usage() {
  cat <<'EOF'
Usage: scripts/vllm_graphmem.sh <command>

Commands:
  start             Start embedding, embedding heartbeat, LLM, and LLM heartbeat.
  stop              Stop all GraphMem-managed services and heartbeats.
  restart           Stop all, then start all with the configured parameters.
  status            Show tmux sessions and probe the selected ports.
  select-llm-gpu    Print the idle physical GPU that would be used by the LLM.
  start-embed       Start embedding service only.
  start-llm         Start 30B service only.
  start-heartbeat   Start embedding heartbeat only.
  start-llm-heartbeat
                    Start LLM heartbeat only.

Current defaults:
  embedding GPU=1, port>=8001; LLM GPU=auto (one idle GPU), port>=8002, TP=1;
  LLM max-num-seqs=384; LLM gpu-memory-utilization=0.88.

Environment overrides: STATE_DIR, HF_HOME, EMBED_GPU, LLM_GPU, EMBED_PORT,
LLM_PORT, PORT_SEARCH_LIMIT,
LLM_MAX_MODEL_LEN, LLM_MAX_NUM_SEQS, LLM_IDLE_MAX_MEMORY_MIB,
LLM_IDLE_MAX_UTILIZATION.
EOF
}

command="${1:-}"
case "${command}" in
  start|stop|status|select-llm-gpu|start-embed|start-llm|start-heartbeat|start-llm-heartbeat)
    exec env \
      STATE_DIR="${STATE_DIR}" HF_HOME="${HF_HOME}" EMBED_GPU="${EMBED_GPU}" \
      LLM_GPU="${LLM_GPU}" EMBED_PORT="${EMBED_PORT}" LLM_PORT="${LLM_PORT}" \
      PORT_SEARCH_LIMIT="${PORT_SEARCH_LIMIT}" \
      LLM_IDLE_MAX_MEMORY_MIB="${LLM_IDLE_MAX_MEMORY_MIB}" LLM_IDLE_MAX_UTILIZATION="${LLM_IDLE_MAX_UTILIZATION}" \
      LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN}" LLM_MAX_NUM_SEQS="${LLM_MAX_NUM_SEQS}" \
      bash "${SERVICE_SCRIPT}" "${command}"
    ;;
  restart)
    env \
      STATE_DIR="${STATE_DIR}" HF_HOME="${HF_HOME}" EMBED_GPU="${EMBED_GPU}" \
      LLM_GPU="${LLM_GPU}" EMBED_PORT="${EMBED_PORT}" LLM_PORT="${LLM_PORT}" \
      PORT_SEARCH_LIMIT="${PORT_SEARCH_LIMIT}" \
      LLM_IDLE_MAX_MEMORY_MIB="${LLM_IDLE_MAX_MEMORY_MIB}" LLM_IDLE_MAX_UTILIZATION="${LLM_IDLE_MAX_UTILIZATION}" \
      LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN}" LLM_MAX_NUM_SEQS="${LLM_MAX_NUM_SEQS}" \
      bash "${SERVICE_SCRIPT}" stop
    exec env \
      STATE_DIR="${STATE_DIR}" HF_HOME="${HF_HOME}" EMBED_GPU="${EMBED_GPU}" \
      LLM_GPU="${LLM_GPU}" EMBED_PORT="${EMBED_PORT}" LLM_PORT="${LLM_PORT}" \
      PORT_SEARCH_LIMIT="${PORT_SEARCH_LIMIT}" \
      LLM_IDLE_MAX_MEMORY_MIB="${LLM_IDLE_MAX_MEMORY_MIB}" LLM_IDLE_MAX_UTILIZATION="${LLM_IDLE_MAX_UTILIZATION}" \
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
