#!/bin/bash
# ==============================================================================
# Megatron-SWIFT multi-node training (Qwen3-30B-A3B-Thinking-2507 full-param SFT, SSH approach)
# Default: 8 nodes x 8 H200 = 64 GPUs, 128K context
#
# Usage:  bash train_megatron_multinode.sh <NODE_RANK>
#   NODE_RANK: 0 ~ (NNODES-1)
#
# Prerequisites:
#   - configure.sh has been run to generate env.sh (inter-node SSH required)
#   - cached_dataset has been synced to all nodes (or using shared filesystem)
# ==============================================================================
set -e

NODE_RANK=${1:?"Usage: bash train_megatron_multinode.sh <NODE_RANK>  (0 ~ NNODES-1)"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/env.sh"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: ${ENV_FILE} not found, please run bash configure.sh first"
    exit 1
fi

source "$ENV_FILE"

if [ "$NODE_RANK" -ge "$NNODES" ]; then
    echo "Error: NODE_RANK=${NODE_RANK} out of range (NNODES=${NNODES}, valid range 0~$((NNODES-1)))"
    exit 1
fi

# ---- NCCL config (already detected by env.sh) ----
export NCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"

# ---- Force HuggingFace, skip ModelScope ----
export USE_HF=1

# ---- Required by Megatron: limit CUDA stream concurrency, otherwise NCCL may deadlock ----
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- (Optional) MODELSCOPE_CACHE / HF_HOME shared path (avoids data preprocessing inconsistencies across nodes) ----
# Megatron multi-node training is sensitive to data consistency; shared filesystem for cache is most reliable
# export HF_HOME="${SHARED_HF_HOME:-/shared/hf_cache}"

CUDA_DEVS=$(seq -s, 0 $((NPROC_PER_NODE - 1)))

# ---- Parallelism strategy (8 nodes x 8 H200 = 64 GPUs; efficient config, quality-first) ----
#   TP=4, CP=2 fits within a single node's 8 GPUs; PP=2 reduces per-layer memory pressure
#   DP = world / TP / PP / CP = 64 / 4 / 2 / 2 = 4
#   With global_batch=128: grad accumulation = 128 / (micro=1 * DP=4) = 32
# Override defaults via nodes.conf or environment variables
TP=${TP:-4}
PP=${PP:-2}
EP=${EP:-4}
CP="${CP:-2}"
TP_COMM_OVERLAP="${TP_COMM_OVERLAP:-false}"
REPORT_TO="${REPORT_TO:-tensorboard wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-megatron-swift}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-searchswarm-sft}"

WANDB_ENV_FILE="${SCRIPT_DIR}/.wandb_env"
if [ -f "${WANDB_ENV_FILE}" ]; then
    # Local secret file. Do not commit it.
    source "${WANDB_ENV_FILE}"
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT
export WANDB_EXP_NAME
if [[ " ${REPORT_TO} " == *" wandb "* && -z "${WANDB_API_KEY}" ]]; then
    echo "Error: REPORT_TO includes wandb but WANDB_API_KEY is empty. Please verify ${WANDB_ENV_FILE} exists, or export WANDB_API_KEY."
    exit 1
fi

NNODES=$NNODES \
NODE_RANK=$NODE_RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
NPROC_PER_NODE=$NPROC_PER_NODE \
CUDA_VISIBLE_DEVICES=$CUDA_DEVS \
megatron sft \
    --model ${MODEL_PATH} \
    --cached_dataset ${DATA_PATH} \
    --tuner_type full \
    --save_safetensors true \
    \
    `# === Parallelism strategy (default TP=4 PP=2 CP=2 EP=4) ===` \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --expert_model_parallel_size ${EP} \
    --context_parallel_size ${CP} \
    --sequence_parallel true \
    \
    `# === Communication / compute overlap (tp_comm_overlap is TE/PyTorch version sensitive; enable with TP_COMM_OVERLAP=true) ===` \
    --tp_comm_overlap ${TP_COMM_OVERLAP} \
    --overlap_grad_reduce true \
    --overlap_param_gather true \
    \
    `# === Recompute: selective (default; nearly zero overhead with FA) ===` \
    --recompute_granularity selective \
    \
    `# === Fusion kernels ===` \
    --apply_rope_fusion true \
    --cross_entropy_loss_fusion true \
    --attention_backend flash \
    \
    `# === MoE acceleration (no quality impact) ===` \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_permute_fusion true \
    --moe_aux_loss_coeff 1e-3 \
    \
    `# === Batch (with CP=2, PP=2: DP=4; 128 packed seq per step; grad accumulation=32) ===` \
    --micro_batch_size 1 \
    --global_batch_size 128 \
    --num_train_epochs 3 \
    --finetune true \
    --lr 5e-5 \
    --lr_warmup_fraction 0.03 \
    --min_lr 1e-6 \
    --weight_decay 0.01 \
    \
    `# === Sequence length ===` \
    --packing true \
    --max_length 131072 \
    \
    `# === Save / logging ===` \
    --output_dir ${OUTPUT_DIR} \
    --save_steps 100 \
    --eval_steps 100 \
    --logging_steps 1 \
    --report_to ${REPORT_TO} \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_exp_name "${WANDB_EXP_NAME}" \
    --no_save_optim true \
    --no_save_rng true \
    \
    `# === Data loading ===` \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8
