#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-LiquidAI/LFM3-Omni-2B}"
MODEL_REVISION="${MODEL_REVISION:-21d75123f86c1f7ffb4bef1ea202710c4e7ebe67}"
VLLM_PORT="${VLLM_PORT:-8091}"
APP_PORT="${APP_PORT:-8000}"
VLLM_STARTUP_TIMEOUT_SEC="${VLLM_STARTUP_TIMEOUT_SEC:-900}"
DEFAULT_VOICE="${DEFAULT_VOICE:-naledi}"
DEFAULT_LANG="${DEFAULT_LANG:-en}"
DELEGATION_MODE="${DELEGATION_MODE:-selective_tool}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-/app/deploy/lfm3_omni_runpod.yaml}"

if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
    DEFAULT_CALLS_DIR="/workspace/calls"
else
    export HF_HOME="${HF_HOME:-/tmp/hf-cache}"
    DEFAULT_CALLS_DIR="/app/calls"
fi

CALLS_DIR="${CALLS_DIR:-${DEFAULT_CALLS_DIR}}"
mkdir -p "${HF_HOME}" "${CALLS_DIR}"

VLLM_PID=""
APP_PID=""

cleanup() {
    local rc=$?
    set +e
    if [ -n "${APP_PID}" ] && kill -0 "${APP_PID}" 2>/dev/null; then
        kill -TERM "${APP_PID}" 2>/dev/null || true
    fi
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -TERM "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    exit "${rc}"
}
trap cleanup SIGTERM SIGINT

export MODEL_ID MODEL_REVISION
echo "[start_runpod] Ensuring complete HuggingFace snapshot for ${MODEL_ID} @ ${MODEL_REVISION} (HF_HOME=${HF_HOME})..."
python3 -c '
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    revision=os.environ["MODEL_REVISION"],
)
with open("/tmp/model_local_dir.txt", "w", encoding="utf-8") as fh:
    fh.write(path)
'
MODEL_LOCAL_DIR="$(cat /tmp/model_local_dir.txt)"
echo "[start_runpod] Snapshot ready at ${MODEL_LOCAL_DIR}"
test -f "${MODEL_LOCAL_DIR}/model.safetensors"
test -f "${MODEL_LOCAL_DIR}/audio_tokenizer/model.safetensors"

echo "[start_runpod] Launching vLLM-Omni on 127.0.0.1:${VLLM_PORT}"
echo "[start_runpod] Served model: ${MODEL_ID} <- ${MODEL_LOCAL_DIR}"

vllm serve "${MODEL_LOCAL_DIR}" \
    --served-model-name "${MODEL_ID}" \
    --omni \
    --trust-remote-code \
    --deploy-config "${DEPLOY_CONFIG}" \
    --async-chunk \
    --limit-mm-per-prompt '{"audio":8}' \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" &
VLLM_PID=$!

deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SEC))
last_log=0
while true; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[start_runpod] ERROR: vllm serve exited prematurely." >&2
        wait "${VLLM_PID}" || true
        exit 1
    fi
    if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "[start_runpod] vLLM-Omni is ready at http://127.0.0.1:${VLLM_PORT}"
        break
    fi
    if [ "${SECONDS}" -ge "${deadline}" ]; then
        echo "[start_runpod] ERROR: Timed out waiting ${VLLM_STARTUP_TIMEOUT_SEC}s for vLLM-Omni." >&2
        kill -TERM "${VLLM_PID}" 2>/dev/null || true
        exit 1
    fi
    if [ $((SECONDS - last_log)) -ge 15 ]; then
        echo "[start_runpod] Waiting for vLLM-Omni startup (${SECONDS}s elapsed)..."
        last_log="${SECONDS}"
    fi
    sleep 3
done

APP_ARGS=(
    /app/telesales_agent.py
    --backend vllm
    --vllm-url "http://127.0.0.1:${VLLM_PORT}"
    --model "${MODEL_ID}"
    --host 0.0.0.0
    --port "${APP_PORT}"
    --voice "${DEFAULT_VOICE}"
    --lang "${DEFAULT_LANG}"
    --delegation-mode "${DELEGATION_MODE}"
    --calls-dir "${CALLS_DIR}"
)

if [ -n "${FLAMETREE_AGENT_ID:-}" ]; then
    APP_ARGS+=(--flametree-agent-id "${FLAMETREE_AGENT_ID}")
fi
if [ -n "${FLAMETREE_BASE_URL:-}" ]; then
    APP_ARGS+=(--flametree-base-url "${FLAMETREE_BASE_URL}")
fi

echo "[start_runpod] Starting Call Studio on 0.0.0.0:${APP_PORT} (calls_dir=${CALLS_DIR})"
python3 "${APP_ARGS[@]}" &
APP_PID=$!
wait "${APP_PID}"
