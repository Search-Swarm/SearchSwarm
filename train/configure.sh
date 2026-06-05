#!/bin/bash
# ==============================================================================
# Megatron-SWIFT auto-configuration script - run on the head node (node0)
# Reads nodes.conf -> detects network -> configures SSH -> generates env.sh
#
# Also writes a few Megatron-specific env vars (CUDA_DEVICE_MAX_CONNECTIONS) into env.sh.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF_FILE="${SCRIPT_DIR}/nodes.conf"
ENV_FILE="${SCRIPT_DIR}/env.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

# ===================== Load configuration =====================

if [ ! -f "$CONF_FILE" ]; then
    fail "${CONF_FILE} not found"
    echo "Please edit nodes.conf first and fill in machine IPs and data paths"
    exit 1
fi

source "$CONF_FILE"

NNODES=${#NODES[@]}

echo ""
echo "============================================"
echo "  Megatron-SWIFT multi-node training auto-configuration (${NNODES} nodes)"
echo "============================================"
echo ""

if [ "$NNODES" -lt 1 ]; then
    fail "NODES array is empty; please add at least 1 node IP in nodes.conf"
    exit 1
fi

VALIDATION_OK=true

for i in "${!NODES[@]}"; do
    ip="${NODES[$i]}"
    if [[ -z "$ip" || "$ip" == *"xxx"* ]]; then
        fail "NODES[$i] not properly configured (current value: ${ip})"
        VALIDATION_OK=false
    fi
done

for var in SSH_USER DATA_PATH MODEL_PATH; do
    val="${!var}"
    if [[ -z "$val" || "$val" == "/path/to"* ]]; then
        fail "${var} not properly configured (current value: ${val})"
        VALIDATION_OK=false
    fi
done

if [ "$VALIDATION_OK" != "true" ]; then
    echo ""
    fail "Please edit nodes.conf with correct values and try again"
    exit 1
fi

ok "nodes.conf validation passed"
echo "  Nodes:         ${NNODES}"
echo "  Node IPs:      ${NODES[*]}"
echo "  GPUs per node: ${NPROC_PER_NODE:-8}"
echo "  Data:          ${DATA_PATH}"
echo "  Model:         ${MODEL_PATH}"

# ===================== 1. Check passwordless SSH =====================

echo ""
echo "--------------------------------------------"
info "[1/4] Checking passwordless SSH connectivity..."
echo "--------------------------------------------"

SSH_FAILED=()
for i in "${!NODES[@]}"; do
    ip=${NODES[$i]}
    if ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
           "${SSH_USER}@${ip}" "echo ok" &>/dev/null; then
        ok "node${i} (${ip})"
    else
        fail "node${i} (${ip}) - passwordless SSH login failed"
        SSH_FAILED+=($i)
    fi
done

if [ ${#SSH_FAILED[@]} -gt 0 ]; then
    echo ""
    warn "The following nodes need passwordless SSH configured: ${SSH_FAILED[*]}"

    if [ ! -f ~/.ssh/id_rsa ]; then
        info "Generating SSH key pair..."
        ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa -q
        ok "Key pair generated"
    fi

    for i in "${SSH_FAILED[@]}"; do
        ip=${NODES[$i]}
        echo ""
        info "Configuring node${i} (${ip}) - please enter password:"
        ssh-copy-id -o StrictHostKeyChecking=no "${SSH_USER}@${ip}"
    done

    echo ""
    info "Re-verifying..."
    for i in "${SSH_FAILED[@]}"; do
        ip=${NODES[$i]}
        if ssh -o BatchMode=yes -o ConnectTimeout=10 "${SSH_USER}@${ip}" "echo ok" &>/dev/null; then
            ok "node${i} (${ip})"
        else
            fail "node${i} (${ip}) still unable to connect!"
            echo "Please check the network and SSH configuration manually and try again"
            exit 1
        fi
    done
fi

ok "All nodes SSH connected"

# ===================== 2. Detect network interface =====================

echo ""
echo "--------------------------------------------"
info "[2/4] Detecting network interface (NCCL_SOCKET_IFNAME)..."
echo "--------------------------------------------"

NODE0_IP="${NODES[0]}"
IFNAME=""

LOCAL_IPS=$(hostname -I 2>/dev/null || ip addr show | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)

if echo "$LOCAL_IPS" | grep -qw "$NODE0_IP"; then
    info "Current machine is node0, detecting interface locally..."
    IFNAME=$(ip -br addr | grep "$NODE0_IP" | awk '{print $1}' | head -1)
else
    info "Current machine is not node0, detecting interface via SSH to node0..."
    IFNAME=$(ssh -o BatchMode=yes "${SSH_USER}@${NODE0_IP}" \
        "ip -br addr | grep '${NODE0_IP}'" 2>/dev/null | awk '{print $1}' | head -1)
fi

if [ -z "$IFNAME" ]; then
    warn "Unable to resolve interface name from IP, trying smart detection..."
    _DETECT_CMD='ip -br addr show | grep " UP " | grep -v -E "^(lo|docker|veth|br-|reth)" | awk "\$3 ~ /[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+/ {print \$1}"'
    if echo "$LOCAL_IPS" | grep -qw "$NODE0_IP"; then
        _CANDIDATES=$(eval "$_DETECT_CMD")
    else
        _CANDIDATES=$(ssh -o BatchMode=yes "${SSH_USER}@${NODE0_IP}" "$_DETECT_CMD" 2>/dev/null)
    fi
    IFNAME=$(echo "$_CANDIDATES" | grep -E '^(eth|ens|eno)' | head -1)
    [ -z "$IFNAME" ] && IFNAME=$(echo "$_CANDIDATES" | head -1)
fi

if [ -z "$IFNAME" ]; then
    warn "Unable to auto-detect interface name"
    echo ""
    info "Network interfaces on node0 (only those with IP addresses):"
    if echo "$LOCAL_IPS" | grep -qw "$NODE0_IP"; then
        ip -br addr show | grep ' UP ' | awk '$3 ~ /[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/'
    else
        ssh -o BatchMode=yes "${SSH_USER}@${NODE0_IP}" \
            "ip -br addr show | grep ' UP ' | awk '\$3 ~ /[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/'" 2>/dev/null
    fi
    echo ""
    read -p "  Enter the inter-node communication interface name (e.g. eth0, ib0, bond0): " IFNAME
fi

ok "Network interface: ${IFNAME}"

# ===================== 3. Detect InfiniBand =====================

echo ""
echo "--------------------------------------------"
info "[3/4] Detecting InfiniBand / RoCE..."
echo "--------------------------------------------"

IB_DISABLE=1

IB_CHECK_CMD='
if [ -d /sys/class/infiniband ] && [ "$(ls /sys/class/infiniband 2>/dev/null)" ]; then
    echo "IB_FOUND $(ls /sys/class/infiniband)"
elif command -v ibstat &>/dev/null; then
    echo "IB_FOUND ibstat"
else
    echo "IB_NOT_FOUND"
fi
'

IB_RESULT=$(ssh -o BatchMode=yes "${SSH_USER}@${NODE0_IP}" "$IB_CHECK_CMD" 2>/dev/null \
    || eval "$IB_CHECK_CMD")

if echo "$IB_RESULT" | grep -q "IB_FOUND"; then
    IB_DISABLE=0
    IB_DETAIL=$(echo "$IB_RESULT" | sed 's/IB_FOUND //')
    ok "IB device found (${IB_DETAIL}) -> NCCL_IB_DISABLE=0"
else
    IB_DISABLE=1
    warn "No IB device detected -> NCCL_IB_DISABLE=1 (using TCP/IP)"
fi

# ===================== 4. Detect port =====================

echo ""
echo "--------------------------------------------"
info "[4/4] Detecting communication port..."
echo "--------------------------------------------"

MASTER_PORT=29500

PORT_CHECK_CMD="ss -tlnp 2>/dev/null | grep -c ':${MASTER_PORT} ' || true"
PORT_USED=$(ssh -o BatchMode=yes "${SSH_USER}@${NODE0_IP}" "$PORT_CHECK_CMD" 2>/dev/null \
    || eval "$PORT_CHECK_CMD")

while [ "${PORT_USED:-0}" -gt 0 ]; do
    warn "Port ${MASTER_PORT} is in use, trying next..."
    MASTER_PORT=$((MASTER_PORT + 1))
    PORT_CHECK_CMD="ss -tlnp 2>/dev/null | grep -c ':${MASTER_PORT} ' || true"
    PORT_USED=$(ssh -o BatchMode=yes "${SSH_USER}@${NODE0_IP}" "$PORT_CHECK_CMD" 2>/dev/null \
        || eval "$PORT_CHECK_CMD")
done

ok "Communication port: ${MASTER_PORT}"

# ===================== Generate env.sh =====================

echo ""
echo "--------------------------------------------"
info "Generating configuration file ${ENV_FILE}..."
echo "--------------------------------------------"

NPROC=${NPROC_PER_NODE:-8}
TOTAL_GPUS=$((NNODES * NPROC))

NODES_ARRAY_STR=""
for i in "${!NODES[@]}"; do
    NODES_ARRAY_STR+="    \"${NODES[$i]}\"    # node${i}"
    if [ $i -eq 0 ]; then
        NODES_ARRAY_STR+=" (head node)"
    fi
    NODES_ARRAY_STR+=$'\n'
done

cat > "${ENV_FILE}" << EOF
#!/bin/bash
# ============================================================
# Megatron-SWIFT cluster configuration - auto-generated by configure.sh
# Generated at: $(date '+%Y-%m-%d %H:%M:%S')
# Regenerate: bash configure.sh
# ============================================================

SSH_USER="${SSH_USER}"

CLUSTER_NODES=(
${NODES_ARRAY_STR})

NNODES=${NNODES}
NPROC_PER_NODE=${NPROC}
TOTAL_GPUS=${TOTAL_GPUS}

MASTER_ADDR="${NODE0_IP}"
MASTER_PORT=${MASTER_PORT}

NCCL_SOCKET_IFNAME="${IFNAME}"
NCCL_IB_DISABLE=${IB_DISABLE}
GLOO_SOCKET_IFNAME="${IFNAME}"

# Training data / model / output
DATA_PATH="${DATA_PATH}"
MODEL_PATH="${MODEL_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:-./megatron_output/qwen3-30b-a3b-thinking-sft}"

# Parallelism strategy (blank = train_megatron_multinode.sh uses defaults TP=4 PP=2 EP=4)
TP="${TP:-}"
PP="${PP:-}"
EP="${EP:-}"
EOF

ok "Configuration written to ${ENV_FILE}"

echo ""
echo "============================================"
echo -e "  ${GREEN}Configuration complete!${NC}"
echo "============================================"
echo ""
echo "  Nodes:           ${NNODES}"
echo "  GPUs per node:   ${NPROC}"
echo "  Total GPUs:      ${TOTAL_GPUS}"
echo "  Head node (node0): ${NODE0_IP}"
echo "  Comm port:       ${MASTER_PORT}"
echo "  Network iface:   ${IFNAME}"
echo "  IB status:       $([ $IB_DISABLE -eq 0 ] && echo 'enabled (high-speed)' || echo 'disabled (TCP)')"
echo "  Data path:       ${DATA_PATH}"
echo "  Model:           ${MODEL_PATH}"
echo "  Output dir:      ${OUTPUT_DIR:-./megatron_output/qwen3-30b-a3b-thinking-sft}"
echo ""
echo "  Next steps:"
echo "    1. bash setup_env.sh                  # Install Megatron-SWIFT on this machine"
echo "    2. bash train_megatron_multinode.sh 0  # Launch training on each node"
echo "============================================"
