#!/bin/bash
# AKS Strategy Evaluation Script
# Adaptive Keyframe Selection with recursive temporal segmentation

# HuggingFace settings
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="hf_xxx"
export HF_HUB_ENABLE_HF_TRANSFER=False

# Optional: Specify GPU devices
# export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --num_processes=1 --main_process_port=12342 -m lmms_eval \
    --model qwen2_5_vl_chat_w_aks \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct,attn_implementation=flash_attention_2,max_num_frames=8,target_token_per_frame=512 \
    --tasks videomme \
    --verbosity=DEBUG \
    --batch_size 1 \
    --output_path ./outputs/qwen25vl_aks \
    --log_samples
