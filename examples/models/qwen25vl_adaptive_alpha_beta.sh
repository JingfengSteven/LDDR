#!/bin/bash
# CDPruner + DPP with Adaptive Alpha-Beta Strategy
# 基于图文相似度分布自适应调整 α 和 β 参数

# HuggingFace settings
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="hf_xxx"
export HF_HUB_ENABLE_HF_TRANSFER=False

# Optional: Specify GPU devices
# export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --num_processes=1 --main_process_port=12342 -m lmms_eval \
    --model qwen2_5_vl_chat_wo_ours_v3 \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct,attn_implementation=flash_attention_2,max_num_frames=8,target_token_per_frame=512,min_token_per_frame=256,max_token_per_frame=4096 \
    --tasks videomme \
    --verbosity=DEBUG \
    --batch_size 1 \
    --output_path ./outputs/qwen25vl_adaptive_alpha_beta \
    --log_samples
