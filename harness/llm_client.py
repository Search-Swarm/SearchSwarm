"""API model configuration and tokenizer management.

API mode talks to any OpenAI-compatible endpoint, and every API model is treated
as a standard OpenAI-compatible model. The maps below are optional extension
points — leave them empty for the standard behaviour:

- TOKENIZER_HF_MAP: API model name -> HuggingFace tokenizer repo, used for local
  token counting (TOKEN_COUNTER=local). If a model isn't listed, token counting
  falls back to the server-reported usage.
- MODEL_PARAM_CONFIG: per-model request-parameter overrides for endpoints that
  need them. If a model isn't listed, the standard defaults below are used.
- TokenizerRegistry: global tokenizer cache, each tokenizer loaded only once.
"""

import os
from transformers import AutoTokenizer

# Optional: API model name -> HuggingFace tokenizer repo (for local token counting).
TOKENIZER_HF_MAP = {}

# Optional: per-model request-parameter overrides. Each value may set:
#   "drop_params":       set of request params to omit (e.g. {"logprobs"})
#   "force_temperature": a temperature to force, or None for the configured value
#   "extra_body":        a dict merged into the request body
MODEL_PARAM_CONFIG = {}

_DEFAULT_PARAM_CONFIG = {
    "drop_params": set(),
    "force_temperature": None,
}


# ====================================================================
# Tokenizer cache
# ====================================================================

class TokenizerRegistry:
    _cache = {}

    @classmethod
    def get(cls, model_name):
        if model_name in cls._cache:
            return cls._cache[model_name]
        hf_name = TOKENIZER_HF_MAP.get(model_name)
        if not hf_name:
            print(f"[TokenizerRegistry] No tokenizer for {model_name}, using API usage for token counting")
            cls._cache[model_name] = None
            return None
        print(f"[TokenizerRegistry] Loading tokenizer: {hf_name} ...")
        tok = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        print(f"[TokenizerRegistry] Loaded: {type(tok).__name__}, vocab_size={tok.vocab_size}")
        cls._cache[model_name] = tok
        return tok


def get_model_config(model_name):
    return MODEL_PARAM_CONFIG.get(model_name, _DEFAULT_PARAM_CONFIG)
