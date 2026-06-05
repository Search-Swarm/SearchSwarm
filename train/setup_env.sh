#!/bin/bash
# ==============================================================================
# Megatron-SWIFT environment setup - run on each machine
#
# Prerequisites: Python>=3.10, CUDA>=12.4, PyTorch>=2.6, ms-swift (swift sft environment)
# Recommended: Python 3.12, CUDA 12.8, PyTorch 2.8.0
#
# Compared to swift sft (HF/DeepSpeed path), Megatron-SWIFT additionally requires:
#   - transformer-engine  (NV optimized TE kernels)
#   - apex                (NV fused-kernels, optional)
#   - mcore-bridge        (HF <-> mcore online bridge, no need to pre-convert mcore weights)
#   - megatron-core       (NV Megatron-LM core)
#   - flash-attn 2.8.3    (TE required version range)
# ==============================================================================
set -e
INSTALL_APEX="${INSTALL_APEX:-true}"
INSTALL_WANDB="${INSTALL_WANDB:-true}"

# ms-swift itself (skip upgrade if already installed)
pip install ms-swift -U
pip install --upgrade datasets
if [ "$INSTALL_WANDB" = "true" ]; then
    pip install wandb -U
fi

# transformer-engine
# If installation fails, see: https://github.com/modelscope/ms-swift/issues/3793
pip install --no-build-isolation 'transformer-engine[pytorch]' --no-cache-dir

# mcore-bridge (HF <-> mcore online bridge; pass HF model id to --model at training time, auto-converts internally)
pip install mcore-bridge -U

# megatron-core (Megatron-SWIFT requires version range 0.12 ~ 0.16)
pip install 'megatron-core==0.16.*' -U

# flash-attn 2.8.3 (upper bound required by TE 2.10)
# If installation fails, find a matching cuda/torch wheel at https://github.com/Dao-AILab/flash-attention/releases
MAX_JOBS=8 pip install 'flash-attn==2.8.3' --no-build-isolation || \
    echo "[warn] flash-attn compilation failed; please find a matching prebuilt wheel from the releases page"

# apex (optional; if installation fails, add --gradient_accumulation_fusion false to training)
if [ "$INSTALL_APEX" = "true" ]; then
    if [ ! -d apex ]; then
        git clone https://github.com/NVIDIA/apex
    fi
    (cd apex && \
        pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
            --config-settings "--build-option=--cpp_ext" \
            --config-settings "--build-option=--cuda_ext" ./) || \
        echo "[warn] apex installation failed (common). Training scripts already default to --gradient_accumulation_fusion false as fallback"
else
    echo "[info] INSTALL_APEX=false, skip apex install"
fi

# Force HuggingFace
export USE_HF=1

python - <<'PY'
import torch, swift
print(f'PyTorch:         {torch.__version__}')
print(f'CUDA avail:      {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
print(f'ms-swift:        {swift.__version__}')
try:
    import megatron.core
    print(f'megatron-core:   {megatron.core.__version__}')
except Exception as e:
    print(f'megatron-core:   IMPORT FAIL ({e})')
try:
    import mcore_bridge
    v = getattr(mcore_bridge, '__version__', '?')
    print(f'mcore-bridge:    {v}')
except Exception as e:
    print(f'mcore-bridge:    IMPORT FAIL ({e})')
try:
    import transformer_engine
    print(f'transformer-engine: {transformer_engine.__version__}')
except Exception as e:
    print(f'transformer-engine: IMPORT FAIL ({e})')
try:
    import flash_attn
    print(f'flash-attn:      {flash_attn.__version__}')
except Exception as e:
    print(f'flash-attn:      IMPORT FAIL ({e})')
print('Done.')
PY
