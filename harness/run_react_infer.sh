#!/bin/bash
# ==============================================================================
# Run inference. Reads every setting from .env, then launches run_multi_react.py.
# If MODEL_MODE=local (or SUB_AGENT_MODE=local), start the vLLM servers first
# with `bash deploy_model.sh`.
#
# Override the env file with `ENV_FILE=/path/to/other.env bash run_react_infer.sh`.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: env file not found at $ENV_FILE. Edit .env first."
    exit 1
fi

echo "Loading environment from $ENV_FILE ..."
set -a            # export everything sourced below
source "$ENV_FILE"
set +a

cd "$SCRIPT_DIR"

python -u run_multi_react.py \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output "$OUTPUT_PATH" \
    --max_workers "$MAX_WORKERS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --presence_penalty "$PRESENCE_PENALTY" \
    --roll_out_count "$ROLLOUT_COUNT" \
    --total_splits "${WORLD_SIZE:-1}" \
    --worker_split "$(( ${RANK:-0} + 1 ))"
