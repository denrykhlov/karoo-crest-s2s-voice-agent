#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-LiquidAI/LFM3-Omni-2B}"
MODEL_REVISION="${MODEL_REVISION:-21d75123f86c1f7ffb4bef1ea202710c4e7ebe67}"
VOICE_BACKEND="${VOICE_BACKEND:-transformers}"
APP_PORT="${APP_PORT:-8000}"
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
export VLLM_LOCAL_MODEL_DIR="${MODEL_LOCAL_DIR}"

APP_ARGS=(
    /app/telesales_agent.py
    --backend "${VOICE_BACKEND}"
    --model "${MODEL_ID}"
    --model-revision "${MODEL_REVISION}"
    --host 0.0.0.0
    --port "${APP_PORT}"
    --voice "${DEFAULT_VOICE}"
    --lang "${DEFAULT_LANG}"
    --delegation-mode "${DELEGATION_MODE}"
    --calls-dir "${CALLS_DIR}"
)

if [ "${VOICE_BACKEND}" = "vllm" ]; then
    APP_ARGS+=(--vllm-deploy-config "${DEPLOY_CONFIG}")
fi

if [ -n "${FLAMETREE_AGENT_ID:-}" ]; then
    APP_ARGS+=(--flametree-agent-id "${FLAMETREE_AGENT_ID}")
fi
if [ -n "${FLAMETREE_BASE_URL:-}" ]; then
    APP_ARGS+=(--flametree-base-url "${FLAMETREE_BASE_URL}")
fi

echo "[start_runpod] Starting Call Studio (--backend ${VOICE_BACKEND}) on 0.0.0.0:${APP_PORT} (calls_dir=${CALLS_DIR})"
exec python3 "${APP_ARGS[@]}"
