#!/bin/bash
# ==============================================================================
# Megatron-SWIFT single-node lightweight debug training (Qwen2.5-0.5B-Instruct full-param SFT)
#
# Uses 1 GPU + a local small SFT JSONL by default. Purpose: quickly validate environment / data flow /
# Megatron-SWIFT launch chain without depending on remote dataset Hub. For 30B MoE + 128K production training,
# use multi-node:
#   - bash train_megatron_multinode.sh N   (multi-node, via torchrun)
#
# Uses mcore-bridge to load HF models online, no need to convert weights to mcore first
#
# For multi-GPU debugging, override environment variables:
#   NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 bash train_megatron.sh
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Lightweight debug defaults ----
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"   # or a locally downloaded HF path
DATASET="${DATASET:-${SCRIPT_DIR}/debug_data/sft_debug.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./megatron_output/qwen2.5-0.5b-debug-sft}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---- Default: pull model from HuggingFace; dataset defaults to local JSONL ----
export USE_HF="${USE_HF:-1}"

# ---- Required by Megatron: limit CUDA stream concurrency, otherwise NCCL may deadlock ----
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Launch training ----
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
megatron sft \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --tuner_type full \
    --save_safetensors true \
    \
    `# === Parallelism: small model defaults to pure DP / single GPU, no MoE-specific params ===` \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --sequence_parallel false \
    \
    `# === Batch / learning rate ===` \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --num_train_epochs 1 \
    --finetune true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.03 \
    --min_lr 1e-6 \
    --weight_decay 0.01 \
    \
    `# === Memory optimization ===` \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --cross_entropy_loss_fusion true \
    --gradient_accumulation_fusion false \
    --attention_backend flash \
    --packing false \
    --max_length 2048 \
    \
    `# === Save / logging ===` \
    --output_dir "${OUTPUT_DIR}" \
    --save_steps 50 \
    --logging_steps 1 \
    --no_save_optim true \
    --no_save_rng true \
    \
    `# === Data loading ===` \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4

# ---- After training, convert mcore weights back to HF (for inference / uploading) ----
# CUDA_VISIBLE_DEVICES=0 \
# swift export \
#     --mcore_model ${OUTPUT_DIR}/vx-xxx/checkpoint-xxx \
#     --to_hf true \
#     --torch_dtype bfloat16 \
#     --output_dir ${OUTPUT_DIR}/vx-xxx/checkpoint-xxx-hf
