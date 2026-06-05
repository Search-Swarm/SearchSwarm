#!/bin/bash
# ==============================================================================
# Megatron-SWIFT multi-node launcher for shared-filesystem clusters
# (no inter-node SSH, no Ray, no scheduler-assigned ranks).
#
# Use this when a scheduler (Kubernetes Job, cloud batch, SLURM with a shared
# filesystem, ...) starts N identical containers that:
#   - share one filesystem (RDZV_DIR is visible to every node), and
#   - do NOT tell each node its rank or who the master is.
#
# Run the SAME command on every node at the same time. The nodes self-organize
# through RDZV_DIR:
#   1. each node atomically claims a unique rank (an mkdir lock),
#   2. rank 0 publishes its IP + port (the torch.distributed master),
#   3. every node waits at a barrier until all NNODES have arrived,
#   4. each node launches `megatron sft` with its rank and the shared master.
#
# These must be IDENTICAL on every node for one launch:
#   NNODES, RDZV_DIR (on the shared FS), MODEL_PATH, DATA_PATH, OUTPUT_DIR.
# Use a FRESH RDZV_DIR per launch (e.g. include a job id) so stale rank locks
# from a previous run don't linger.
# ==============================================================================
set -eo pipefail

# ---- Cluster shape (override via env) ----
NNODES="${NNODES:?set NNODES to the total number of nodes}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"          # GPUs per node
MASTER_PORT="${MASTER_PORT:-29500}"            # torch.distributed master port

# ---- Paths (the SAME path must resolve on every node) ----
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
DATA_PATH="${DATA_PATH:?set DATA_PATH to the cached_dataset dir (same path on every node)}"
OUTPUT_DIR="${OUTPUT_DIR:-./megatron_output/searchswarm-sft}"

# Rendezvous directory on the SHARED filesystem. Same path on every node, fresh
# per launch (see header).
RDZV_DIR="${RDZV_DIR:-${OUTPUT_DIR}/rdzv}"

# ---- Parallelism (same defaults as the SSH / Ray scripts) ----
TP="${TP:-4}"
PP="${PP:-2}"
EP="${EP:-4}"
CP="${CP:-2}"
TP_COMM_OVERLAP="${TP_COMM_OVERLAP:-false}"

# This node's IP. Override NODE_IP if auto-detection picks the wrong interface
# (multi-NIC / InfiniBand hosts).
NODE_IP="${NODE_IP:-$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^127\.' | head -1 || true)}"

mkdir -p "$RDZV_DIR" "$OUTPUT_DIR"

log() { echo "[$(date '+%H:%M:%S')] [rank ${NODE_RANK:-?}] $*"; }

# --- claim a unique rank -------------------------------------------------------
# mkdir is atomic on a POSIX filesystem, so exactly one node wins each lock.
claim_node_rank() {
    local rank
    for rank in $(seq 0 $((NNODES - 1))); do
        if mkdir "${RDZV_DIR}/rank_${rank}.lock" 2>/dev/null; then
            echo "$rank"
            return 0
        fi
    done
    echo "ERROR: ranks 0..$((NNODES - 1)) are all claimed in ${RDZV_DIR}." \
         "Use a fresh RDZV_DIR or delete it before re-running." >&2
    return 1
}

# --- wait until a file exists and is non-empty ---------------------------------
wait_for_file() {
    local path="$1" timeout="${2:-900}" waited=0
    while [ ! -s "$path" ]; do
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge "$timeout" ]; then
            echo "ERROR: timed out after ${timeout}s waiting for $path" >&2
            exit 1
        fi
    done
}

# --- barrier: wait until all NNODES nodes reach the same point -----------------
barrier() {
    local name="$1"
    touch "${RDZV_DIR}/${name}.rank_${NODE_RANK}"
    while [ "$(find "$RDZV_DIR" -maxdepth 1 -name "${name}.rank_*" | wc -l)" -lt "$NNODES" ]; do
        sleep 2
    done
}

# --- NCCL / CUDA environment ---------------------------------------------------
configure_nccl() {
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

    # InfiniBand if the host has it, otherwise TCP.
    if [ -d /sys/class/infiniband ] && [ -n "$(ls /sys/class/infiniband 2>/dev/null)" ]; then
        export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    else
        export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    fi

    # Use the interface that routes to the master, unless the caller set one.
    if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
        local iface
        iface=$(ip route get "$MASTER_ADDR" 2>/dev/null \
                | awk '{for (i = 1; i <= NF; i++) if ($i == "dev") {print $(i + 1); exit}}' || true)
        if [ -n "$iface" ] && [ "$iface" != "lo" ]; then
            export NCCL_SOCKET_IFNAME="$iface"
        fi
    fi
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-}}"
}

# ================================== main ======================================

NODE_RANK=$(claim_node_rank)
export NODE_RANK
log "claimed rank ${NODE_RANK}/${NNODES} (ip=${NODE_IP}, host=$(hostname))"

# Rank 0 is the master: publish its address so the workers can find it.
if [ "$NODE_RANK" -eq 0 ]; then
    echo "$NODE_IP"     > "${RDZV_DIR}/master_addr"
    echo "$MASTER_PORT" > "${RDZV_DIR}/master_port"
    log "elected master ${NODE_IP}:${MASTER_PORT}"
else
    wait_for_file "${RDZV_DIR}/master_addr"
    wait_for_file "${RDZV_DIR}/master_port"
fi
MASTER_ADDR=$(cat "${RDZV_DIR}/master_addr")
MASTER_PORT=$(cat "${RDZV_DIR}/master_port")

configure_nccl
export MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE

# GPUs 0..NPROC_PER_NODE-1 as a comma list, e.g. "0,1,2,3,4,5,6,7".
gpu_list=""
for i in $(seq 0 $((NPROC_PER_NODE - 1))); do gpu_list="${gpu_list:+$gpu_list,}$i"; done
export CUDA_VISIBLE_DEVICES="$gpu_list"

log "distributed: NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER=${MASTER_ADDR}:${MASTER_PORT} NPROC_PER_NODE=${NPROC_PER_NODE}"
log "nccl: NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-auto} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"

# Hold every node until all of them have arrived, then launch together.
log "waiting at barrier for all ${NNODES} nodes..."
barrier ready
log "all nodes present — launching megatron sft"

export USE_HF=1
megatron sft \
    --model "${MODEL_PATH}" \
    --cached_dataset "${DATA_PATH}" \
    --tuner_type full \
    --save_safetensors true \
    \
    `# === Parallelism (TP=4 PP=2 EP=4 CP=2 per 8-GPU node) ===` \
    --tensor_model_parallel_size "${TP}" \
    --pipeline_model_parallel_size "${PP}" \
    --expert_model_parallel_size "${EP}" \
    --context_parallel_size "${CP}" \
    --sequence_parallel true \
    \
    `# === Communication / compute overlap ===` \
    --tp_comm_overlap "${TP_COMM_OVERLAP}" \
    --overlap_grad_reduce true \
    --overlap_param_gather true \
    \
    `# === Recompute + fusion kernels ===` \
    --recompute_granularity selective \
    --apply_rope_fusion true \
    --cross_entropy_loss_fusion true \
    --attention_backend flash \
    \
    `# === MoE acceleration ===` \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_permute_fusion true \
    --moe_aux_loss_coeff 1e-3 \
    \
    `# === Batch / optimization ===` \
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
    --report_to tensorboard \
    --no_save_optim true \
    --no_save_rng true \
    \
    `# === Data loading ===` \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8
