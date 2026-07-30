#!/usr/bin/env bash
# Full LongMemEval benchmark runner for GraphMem.
#
# 复用 smoke_longmemeval.sh 拉起 embed + llm 两个 vLLM 服务，然后对
# LongMemEval_S（清洗版，共 500 题、6 种题型）跑【全部题目 + 全部题型】，
# 两个对照变体一起跑。
#
# 用法：
#   scripts/run_full_longmemeval.sh                  # 起服务 + 跑全量，结束后停服务
#   KEEP_UP=1 scripts/run_full_longmemeval.sh        # 跑完保留服务（方便继续调试）
#   QTYPE=multi-session scripts/run_full_longmemeval.sh   # 只跑某个题型
#   OUTDIR=/path/to/out scripts/run_full_longmemeval.sh   # 自定义输出目录
#   nohup bash scripts/run_full_longmemeval.sh > run_full.out 2>&1 &
#   
# 可断点续跑：脚本带 --resume，被中断后再次执行会接着上次进度。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE="${SCRIPT_DIR}/smoke_longmemeval.sh"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 互斥锁：同一时间只允许一个 run_full 实例。否则多个实例会共用同一套
# embed/llm 服务和 pidfile，先结束的那个执行 stop 时会把别人正在用的服务杀掉
# （SIGTERM），导致另一边 Connection refused 崩溃。
LOCKFILE="${LOCKFILE:-/tmp/graphmem_run_full.lock}"
exec 9>"${LOCKFILE}"
if ! flock -n 9; then
  echo "[fatal] 已有另一个 run_full 在运行（锁文件 ${LOCKFILE}）。" >&2
  echo "        请等它结束，或先 'bash scripts/smoke_longmemeval.sh stop' 清理后再跑。" >&2
  exit 1
fi

# ============================ CONFIG (可用环境变量覆盖) ============================
RUN_ENV="${RUN_ENV:-graphmem}"
DATA="${DATA:-${REPO}/data/longmemeval_s_cleaned.json}"
OUTDIR="${OUTDIR:-${REPO}/runs/full_s}"

LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
LLM_PORT="${LLM_PORT:-8001}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
EMBED_PORT="${EMBED_PORT:-8002}"
# 超长输入交给 vLLM 截断，避免 "maximum context length" 的 400。须 <= smoke 里的 EMBED_MAXLEN。
EMBED_TRUNCATE_TOKENS="${EMBED_TRUNCATE_TOKENS:-16384}"
# embed 与 llm 抢 GPU、吞吐低，批小一点让每个请求更快返回，避免 ReadTimeout。
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-16}"

QTYPE="${QTYPE:-all}"          # all = 全部题型；也可 multi-session / temporal-reasoning / knowledge-update 等
MAX_Q="${MAX_Q:-100000}"       # 远大于 500，等同于"全部题目"
QWORKERS="${QWORKERS:-2}"      # llm 用 max_num_seqs=1，并发收益有限，按需上调
VARIANTS="${VARIANTS:-direct_session_k16_compact_no_compress direct_session_k16_compact_graphmem}"

# demo 阶段的 HF 缓存：用有写权限的目录，并允许联网下载 LLMLingua 压缩模型。
RUN_HF_HOME="${RUN_HF_HOME:-${HF_HOME:-${HOME}/.cache/huggingface}}"
LLMLINGUA_DEVICE="${LLMLINGUA_DEVICE:-cpu}"   # 压缩模型放 CPU，避免抢 GPU 显存
# 可选：构建阶段启用 LLM 辅助连边（默认关闭，方便 A/B 与回滚）
ENABLE_LLM_ROOT_EDGES="${ENABLE_LLM_ROOT_EDGES:-0}"
LLM_ROOT_EDGE_MAX_TOKENS="${LLM_ROOT_EDGE_MAX_TOKENS:-256}"
LLM_ROOT_EDGE_NEIGHBORS_PER_REL="${LLM_ROOT_EDGE_NEIGHBORS_PER_REL:-2}"
LLM_ROOT_EDGE_MIN_SHARED="${LLM_ROOT_EDGE_MIN_SHARED:-1}"
LLM_ROOT_EDGE_ANCHOR_LIMIT="${LLM_ROOT_EDGE_ANCHOR_LIMIT:-8}"
# =================================================================================

CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
_activate() {
  # conda 钩子与 set -u 不兼容，切换环境时临时关掉 nounset
  set +u
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "$1"
  set -u
}

echo "[1/2] 确保服务就绪 (embed :${EMBED_PORT}, llm :${LLM_PORT}) ..."
bash "${SMOKE}" up || { echo "[fatal] 服务未就绪，先解决服务问题再跑全量。" >&2; exit 1; }

echo "[2/2] 跑全量 benchmark: type=${QTYPE} max=${MAX_Q} variants=[${VARIANTS}] -> ${OUTDIR}"
llm_root_edge_args=()
if [[ "${ENABLE_LLM_ROOT_EDGES}" == "1" ]]; then
  llm_root_edge_args=(
    --enable-llm-root-edges
    --llm-root-edge-max-tokens "${LLM_ROOT_EDGE_MAX_TOKENS}"
    --llm-root-edge-neighbors-per-relation "${LLM_ROOT_EDGE_NEIGHBORS_PER_REL}"
    --llm-root-edge-min-shared "${LLM_ROOT_EDGE_MIN_SHARED}"
    --llm-root-edge-anchor-limit "${LLM_ROOT_EDGE_ANCHOR_LIMIT}"
  )
fi
( _activate "${RUN_ENV}"
  cd "${REPO}"
  export HF_HOME="${RUN_HF_HOME}"
  export HF_HUB_OFFLINE="0"
  mkdir -p "${RUN_HF_HOME}"
  export SGAO_API_KEY="dummy"
  export SGAO_BASE_URL="http://127.0.0.1:${LLM_PORT}/v1"
  export SGAO_MODEL="${LLM_MODEL}"
  export EMBEDDING_TRUNCATE_TOKENS="${EMBED_TRUNCATE_TOKENS}"
  export EMBEDDING_BATCH_SIZE="${EMBED_BATCH_SIZE}"
  python scripts/run_token_demo.py \
    --data "${DATA}" \
    --question-type "${QTYPE}" \
    --max-questions "${MAX_Q}" \
    --output-dir "${OUTDIR}" \
    --variants ${VARIANTS} \
    --embedding-base-url "http://127.0.0.1:${EMBED_PORT}/v1" \
    --embedding-model "${EMBED_MODEL}" \
    --question-workers "${QWORKERS}" \
    --llmlingua-device-map "${LLMLINGUA_DEVICE}" \
    "${llm_root_edge_args[@]}" \
    --resume
)
status=$?

if [[ "${KEEP_UP:-0}" != "1" ]]; then
  echo "[done] 停止服务（如需保留请设 KEEP_UP=1 重跑）"
  bash "${SMOKE}" stop
else
  echo "[done] 已保留服务运行（KEEP_UP=1）"
fi
exit ${status}
