#!/bin/bash
# ==============================================================================
# Start the local vLLM servers used when MODEL_MODE=local (or SUB_AGENT_MODE=local).
#
# Brings up one vLLM server per GPU on ports 6001-6008, then blocks until every
# port answers /v1/models. run_multi_react.py round-robins planning work across
# these eight ports. Skip this script entirely if both the main agent and the
# sub-agents run in API mode.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env not found at $ENV_FILE. Edit .env with your model path and API keys first."
    exit 1
fi

echo "Loading environment from .env ..."
set -a            # export everything sourced below
source "$ENV_FILE"
set +a

if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH is not set in .env"
    exit 1
fi

# VLLM_MODEL is what the local ports actually serve; defaults to MODEL_PATH.
# Set it separately when the main agent talks to an API model (MODEL_MODE=api)
# but local vLLM should still serve a model for local sub-agents
# (SUB_AGENT_MODE=local) -- point SUB_AGENT_MODEL at the same value.
VLLM_MODEL="${VLLM_MODEL:-$MODEL_PATH}"
# Extra per-server flags, e.g. VLLM_EXTRA_ARGS="--max-model-len 131072".
# For tensor parallelism across GPUs, edit the launch loop below instead.
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

echo "Starting 8 vLLM servers (model=$VLLM_MODEL) on ports 6001-6008 ..."
for i in $(seq 0 7); do
    port=$((6001 + i))
    CUDA_VISIBLE_DEVICES=$i vllm serve "$VLLM_MODEL" --host 0.0.0.0 --port "$port" $VLLM_EXTRA_ARGS &
done

# ---- Wait for every port to report ready ----
ports=(6001 6002 6003 6004 6005 6006 6007 6008)
timeout=6000
start_time=$(date +%s)
echo "Waiting for servers to come up (timeout ${timeout}s) ..."
while true; do
    all_ready=true
    for port in "${ports[@]}"; do
        if ! curl -s -f "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            all_ready=false
            break
        fi
    done
    if [ "$all_ready" = "true" ]; then
        echo "All 8 vLLM servers are ready."
        break
    fi
    if [ $(( $(date +%s) - start_time )) -gt "$timeout" ]; then
        echo "Error: vLLM servers did not become ready within ${timeout}s."
        for port in "${ports[@]}"; do
            curl -s -f "http://localhost:$port/v1/models" >/dev/null 2>&1 || echo "  port $port not ready"
        done
        exit 1
    fi
    sleep 10
done
