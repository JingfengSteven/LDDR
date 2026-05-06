# install lmms_eval without building dependencies
# cd lmms_eval;
# pip install --no-deps -U -e .

# install LLaVA without building dependencies
# cd LLaVA
# pip install --no-deps -U -e .

# # install all the requirements that require for reproduce llava results
# pip install -r llava_repr_requirements.txt
export HF_TOKEN="hf_xxx"
# export NCCL_P2P_DISABLE=1

# export CUDA_VISIBLE_DEVICES=2,7
export PYTHONNOUSERSITE=1
export HF_HUB_ENABLE_HF_TRANSFER=False
export USE_TF=0
export USE_JAX=0
export USE_TORCH=1
export TMPDIR=/tmp
export TMPDIR=/tmp/triton_cache
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="hf_xxx"
export HF_HUB_ENABLE_HF_TRANSFER=False

# Run and exactly reproduce llava_v1.5 results!
# mme as an example
cd /path/to/your/workspace/EACL/lmms-eval

  /path/to/your/workspace/cache_center/envs/bin/python -m lmms_eval \
    --model llava \
    --model_args pretrained="/path/to/your/workspace/EACL/ckpt/llava1_5,attn_implementation=flash_attention_2,device_map=auto" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix reproduce \
    --verbosity=DEBUG \
    --output_path /path/to/your/workspace/EACL/llava_res/