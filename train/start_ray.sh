#!/bin/bash
# ==============================================================================
# Ray cluster startup script - run on each machine
#
# Usage:
#   Head node:    bash start_ray.sh head
#   Worker node:  bash start_ray.sh worker <HEAD_IP>
#
# Automatically detects NCCL network configuration before starting
# ==============================================================================
set -e

MODE=${1:?"Usage: bash start_ray.sh head  or  bash start_ray.sh worker <HEAD_IP>"}

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

RAY_PORT=6379
DASHBOARD_PORT=8265

# ===================== Stop existing Ray =====================

if ray status &>/dev/null; then
    warn "Detected a running Ray instance, stopping it first..."
    ray stop --force 2>/dev/null || true
    sleep 2
fi

# ===================== NCCL auto-detection =====================

info "Detecting NCCL network configuration..."

# IB detection
if [ -d /sys/class/infiniband ] && [ "$(ls /sys/class/infiniband 2>/dev/null)" ]; then
    export NCCL_IB_DISABLE=0
    IB_DEVS=$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ')
    ok "InfiniBand/RoCE devices found: ${IB_DEVS}-> NCCL_IB_DISABLE=0"
else
    export NCCL_IB_DISABLE=1
    info "No InfiniBand found -> NCCL_IB_DISABLE=1 (using TCP)"
fi

# Interface name detection:
#   1. Exclude lo/docker/veth/br-/reth (reth is a bond sub-interface, usually has no IP)
#   2. Keep only interfaces with IPv4 addresses (3rd column contains x.x.x.x/)
#   3. Prefer standard Ethernet interfaces like eth/ens/eno
IFNAME=""
_pick_iface() {
    ip -br addr show \
        | grep ' UP ' \
        | grep -v -E '^(lo|docker|veth|br-|reth)' \
        | awk '$3 ~ /[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/ {print $1}'
}

_CANDIDATES=$(_pick_iface)
if [ -n "$_CANDIDATES" ]; then
    # Prefer standard interfaces like eth*/ens*/eno*
    IFNAME=$(echo "$_CANDIDATES" | grep -E '^(eth|ens|eno)' | head -1)
    # If no standard interface found, use the first one with an IP
    [ -z "$IFNAME" ] && IFNAME=$(echo "$_CANDIDATES" | head -1)
fi

if [ -n "$IFNAME" ]; then
    export NCCL_SOCKET_IFNAME="$IFNAME"
    export GLOO_SOCKET_IFNAME="$IFNAME"
    _IFADDR=$(ip -br addr show dev "$IFNAME" | awk '{print $3}' | cut -d/ -f1)
    ok "Network interface: ${IFNAME} (${_IFADDR})"
else
    warn "Unable to auto-detect network interface; NCCL will select automatically"
    warn "If training fails, set manually: export NCCL_SOCKET_IFNAME=<interface_name>"
    info "Current interface list:"
    ip -br addr show | grep ' UP ' | grep -v '^lo '
fi

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

# ===================== Start Ray =====================

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
info "GPUs on this machine: ${NUM_GPUS}"

if [ "$MODE" = "head" ]; then
    info "Starting Ray Head node..."
    RAY_memory_monitor_refresh_ms=0 ray start --head \
        --port=${RAY_PORT} \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=${DASHBOARD_PORT} \
        --num-gpus=${NUM_GPUS}

    HEAD_IP=$(hostname -I | awk '{print $1}')
    echo ""
    ok "Ray Head started!"
    echo ""
    echo "  Dashboard: http://${HEAD_IP}:${DASHBOARD_PORT}"
    echo ""
    echo "  On each worker machine, run:"
    echo "    bash start_ray.sh worker ${HEAD_IP}"
    echo ""
    echo "  After all workers have joined, run:"
    echo "    bash train_megatron_ray.sh"
    echo ""

elif [ "$MODE" = "worker" ]; then
    HEAD_IP=${2:?"Worker mode requires HEAD_IP: bash start_ray.sh worker <HEAD_IP>"}
    info "Joining Ray cluster (Head: ${HEAD_IP}:${RAY_PORT})..."
    RAY_memory_monitor_refresh_ms=0 ray start \
        --address="${HEAD_IP}:${RAY_PORT}" \
        --num-gpus=${NUM_GPUS}

    echo ""
    ok "Joined Ray cluster!"
    echo ""

else
    echo "Error: unknown mode '${MODE}'"
    echo "Usage: bash start_ray.sh head  or  bash start_ray.sh worker <HEAD_IP>"
    exit 1
fi

# Save NCCL config for training scripts
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cat > "${SCRIPT_DIR}/.nccl_env" << EOF
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"
EOF
ok "NCCL config saved to .nccl_env"

echo ""
info "Current NCCL configuration:"
echo "  NCCL_SOCKET_IFNAME = ${NCCL_SOCKET_IFNAME:-<not set>}"
echo "  NCCL_IB_DISABLE    = ${NCCL_IB_DISABLE}"
echo "  GLOO_SOCKET_IFNAME = ${GLOO_SOCKET_IFNAME:-<not set>}"
echo "  NCCL_DEBUG         = ${NCCL_DEBUG:-INFO}"
