# Run longvideobench evaluation with offline mode to prevent video download
# export HF_HOME="/workspace/GX_Project/data"
# export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="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"

export PYTHONNOUSERSITE=1
export HF_HUB_ENABLE_HF_TRANSFER=False
export USE_TF=0
export USE_JAX=0
export USE_TORCH=1
export TMPDIR=/tmp
export TMPDIR=/tmp/triton_cache
source activate
conda activate lmms-eval
cd ./LDDR



export TASK_NAME=longvideobench_val_v


export MAX_NUM_FRAMES=8
export TARGET_TOKEN_PER_FRAME=1024
export MIN_TOKEN_PER_FRAME=256
export MAX_TOKEN_PER_FRAME=1024


export PRETRAINED_PATH=""
export ATTN_IMPLEMENTATION="flash_attention_2"


export BATCH_SIZE=1
export NUM_PROCESSES=1
export OUTPUT_PATH="./eval/${TASK_NAME}_${MAX_NUM_FRAMES}f"
export USE_LONG_CLIP=False


echo "========================================"
echo "Running with configuration:"
echo "  TASK_NAME: ${TASK_NAME}"
echo "  MAX_NUM_FRAMES: ${MAX_NUM_FRAMES}"
echo "  TARGET_TOKEN_PER_FRAME: ${TARGET_TOKEN_PER_FRAME}"
echo "  MIN_TOKEN_PER_FRAME: ${MIN_TOKEN_PER_FRAME}"
echo "  MAX_TOKEN_PER_FRAME: ${MAX_TOKEN_PER_FRAME}"
echo "  PRETRAINED: ${PRETRAINED_PATH}"
echo "  BATCH_SIZE: ${BATCH_SIZE}"
echo "  NUM_PROCESSES: ${NUM_PROCESSES}"
echo "  OUTPUT_PATH: ${OUTPUT_PATH}"
echo "========================================"


accelerate launch --num_processes=${NUM_PROCESSES} --main_process_port=12342 -m lmms_eval \
    --model qwen2_5_vl_chat_wo_ours_v4 \
    --model_args=pretrained=${PRETRAINED_PATH},attn_implementation=${ATTN_IMPLEMENTATION},interleave_visuals=False,max_num_frames=${MAX_NUM_FRAMES},use_longclip=${USE_LONG_CLIP},target_token_per_frame=${TARGET_TOKEN_PER_FRAME},min_token_per_frame=${MIN_TOKEN_PER_FRAME},max_token_per_frame=${MAX_TOKEN_PER_FRAME},test_cdp_min_allocation=${TEST_CDP_MIN_ALLOCATION},test_min_token_max_coverage=${TEST_MIN_TOKEN_MAX_COVERAGE},task_name=${TASK_NAME} \
    --tasks ${TASK_NAME} \
    --verbosity=DEBUG \
    --log_samples \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH}
