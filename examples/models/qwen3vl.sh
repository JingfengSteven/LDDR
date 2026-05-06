#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TASK_NAME="${TASK_NAME:-videomme}"
PRETRAINED_PATH="${PRETRAINED_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_ROOT/outputs/${TASK_NAME}}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12346}"
BATCH_SIZE="${BATCH_SIZE:-1}"

MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-8}"
TARGET_TOKEN_PER_FRAME="${TARGET_TOKEN_PER_FRAME:-1024}"
MIN_TOKEN_PER_FRAME="${MIN_TOKEN_PER_FRAME:-256}"
MAX_TOKEN_PER_FRAME="${MAX_TOKEN_PER_FRAME:-1024}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
USE_LONGCLIP="${USE_LONGCLIP:-True}"

export PYTHONNOUSERSITE=1
export HF_HUB_ENABLE_HF_TRANSFER=False
export USE_TF=0
export USE_JAX=0
export USE_TORCH=1
export TMPDIR="${TMPDIR:-/tmp}"

cd "$PROJECT_ROOT"

accelerate launch --num_processes="${NUM_PROCESSES}" --main_process_port="${MAIN_PROCESS_PORT}" -m lmms_eval \
    --model qwen3_vl_chat_wo_ours_v4 \
    --model_args "pretrained=${PRETRAINED_PATH},attn_implementation=${ATTN_IMPLEMENTATION},interleave_visuals=False,max_num_frames=${MAX_NUM_FRAMES},use_longclip=${USE_LONGCLIP},target_token_per_frame=${TARGET_TOKEN_PER_FRAME},min_token_per_frame=${MIN_TOKEN_PER_FRAME},max_token_per_frame=${MAX_TOKEN_PER_FRAME},task_name=${TASK_NAME}" \
    --tasks "${TASK_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --verbosity DEBUG \
    --output_path "${OUTPUT_PATH}"
