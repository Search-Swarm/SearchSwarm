#!/bin/bash
# ==============================================================================
# Megatron-SWIFT multi-node training (Ray approach - no inter-node SSH required)
#
# Prerequisites:
#   1. All nodes have run start_ray.sh (head / worker)
#   2. All nodes have the same cached_dataset path (rsync or shared filesystem)
#
# Megatron-SWIFT does not support --use_ray natively, so Ray is used here as a
# "passwordless SSH" launcher: ray_launch_megatron.py dispatches `megatron sft`
# as a task to each Ray GPU node.
#
# Inter-node communication params are read from .nccl_env (auto-detected by start_ray.sh).
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# ---- Edit paths below, or override via env vars MODEL_PATH / DATA_PATH / OUTPUT_DIR ----
# MODEL_PATH must be a local path that exists on every Ray node.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
DATA_PATH="${DATA_PATH:-/path/to/cached_dataset/train}"   # must be the same path on all nodes
OUTPUT_DIR="${OUTPUT_DIR:-./megatron_output/qwen3-30b-a3b-sft}"

# ---- Parallelism strategy (8 nodes x 8 H200 = 64 GPUs; efficient config, quality-first) ----
# TP=4, CP=2 fits within a single node's 8 GPUs; PP=2 reduces per-layer memory pressure
# DP = world / TP / PP / CP = 64 / 4 / 2 / 2 = 4
# With global_batch=128: grad accumulation = 128 / (micro=1 * DP=4) = 32
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TP="${TP:-4}"
PP="${PP:-2}"
EP="${EP:-4}"
CP="${CP:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
TP_COMM_OVERLAP="${TP_COMM_OVERLAP:-false}"
CLEANUP_STALE="${CLEANUP_STALE:-true}"
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

TOTAL_GPUS="${TOTAL_GPUS:-64}"
MODEL_PARALLEL_SIZE=$((TP * PP * CP))
if [ $((TOTAL_GPUS % MODEL_PARALLEL_SIZE)) -ne 0 ]; then
    echo "Error: TOTAL_GPUS=${TOTAL_GPUS} is not divisible by TP*PP*CP=${TP}*${PP}*${CP}=${MODEL_PARALLEL_SIZE}"
    exit 1
fi
DP=$((TOTAL_GPUS / MODEL_PARALLEL_SIZE))
if [ $((TP * CP)) -gt "${GPUS_PER_NODE}" ]; then
    warn "TP*CP=$((TP * CP)) > GPUS_PER_NODE=${GPUS_PER_NODE}; TP/CP high-frequency communication may cross nodes, potentially causing significant slowdown"
fi

# ---- NCCL config auto-loaded from .nccl_env detected by start_ray.sh ----
if [ -f "${SCRIPT_DIR}/.nccl_env" ]; then
    ok "NCCL config: ${SCRIPT_DIR}/.nccl_env (auto-detected by start_ray.sh)"
    cat "${SCRIPT_DIR}/.nccl_env"
else
    warn ".nccl_env not found; NCCL interface will be auto-detected by Ray (usually fine; set manually if errors occur)"
fi

# ---- Verify Ray cluster status ----
echo ""
echo "============================================"
echo "  Ray cluster status"
echo "============================================"
ray status

# ---- Launch training ----
echo ""
echo "============================================"
echo "  Launching Megatron-SWIFT training"
echo "  Model:   ${MODEL_PATH}"
echo "  Data:    ${DATA_PATH}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Parallel: TP=${TP}  PP=${PP}  EP=${EP}  CP=${CP}  DP=${DP}  (GPUs/node=${GPUS_PER_NODE})"
echo "  TP comm overlap: ${TP_COMM_OVERLAP}"
echo "  Cleanup stale processes before launch: ${CLEANUP_STALE}"
echo "  Logging: ${REPORT_TO}  (wandb_project=${WANDB_PROJECT}, wandb_exp_name=${WANDB_EXP_NAME})"
echo "  Port:    ${MASTER_PORT}  (if busy, ray_launch_megatron.py will auto-select a free port)"
echo "============================================"
echo ""

LAUNCHER_CLEANUP_ARGS=()
if [ "${CLEANUP_STALE}" = "false" ] || [ "${CLEANUP_STALE}" = "0" ]; then
    LAUNCHER_CLEANUP_ARGS+=(--no-cleanup)
fi

python "${SCRIPT_DIR}/ray_launch_megatron.py" \
    --gpus-per-node "${GPUS_PER_NODE}" \
    --master-port "${MASTER_PORT}" \
    --nccl-env "${SCRIPT_DIR}/.nccl_env" \
    "${LAUNCHER_CLEANUP_ARGS[@]}" \
    -- \
    --model "${MODEL_PATH}" \
    --cached_dataset "${DATA_PATH}" \
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
    --tp_comm_overlap "${TP_COMM_OVERLAP}" \
    --overlap_grad_reduce true \
    --overlap_param_gather true \
    \
    `# === Recompute: selective (Megatron default; nearly zero overhead with FA) ===` \
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
    --output_dir "${OUTPUT_DIR}" \
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
