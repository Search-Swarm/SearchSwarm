"""
Sub-agent tool: delegates research sub-tasks to independent agents
with search and visit capabilities. Supports parallel dispatch of
multiple sub-tasks via ThreadPoolExecutor.

Supports four mode combinations via environment variables:
  Main API  + Sub API   : MODEL_MODE=api,   SUB_AGENT_MODE=api
  Main API  + Sub Local : MODEL_MODE=api,   SUB_AGENT_MODE=local
  Main Local + Sub Local: MODEL_MODE=local,  SUB_AGENT_MODE=local
  Main Local + Sub API  : MODEL_MODE=local,  SUB_AGENT_MODE=api
"""

import json
import json5
import os
import time
import random
import datetime
import threading
import itertools
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Optional, Union

from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    RateLimitError,
)
from qwen_agent.tools.base import BaseTool, register_tool

from hermes_w_py_parser import parse_tool_call_blocks
from prompt import (
    SUB_AGENT_OPENAI_TOOLS,
    render_sub_agent_system_prompt,
)

# =============================================================================
# Environment variables
# =============================================================================

SUB_AGENT_MODE = os.getenv('SUB_AGENT_MODE', os.getenv('MODEL_MODE', 'local'))
SUB_AGENT_MODEL = os.getenv('SUB_AGENT_MODEL', '')
SUB_AGENT_MAX_CONTEXT_TOKENS = int(os.getenv('SUB_AGENT_MAX_CONTEXT_TOKENS', 32768))
SUB_AGENT_MAX_GENERATION_TOKENS = int(os.getenv('SUB_AGENT_MAX_GENERATION_TOKENS', 8192))
SUB_AGENT_MAX_LLM_CALLS = int(os.getenv('SUB_AGENT_MAX_LLM_CALLS', 20))
SUB_AGENT_TIMEOUT_MINUTES = int(os.getenv('SUB_AGENT_TIMEOUT_MINUTES', 30))
SUB_AGENT_TEMPERATURE = float(os.getenv('SUB_AGENT_TEMPERATURE', '0.85'))
SUB_AGENT_TOP_P = float(os.getenv('SUB_AGENT_TOP_P', '0.95'))
SUB_AGENT_PRESENCE_PENALTY = float(os.getenv('SUB_AGENT_PRESENCE_PENALTY', '1.1'))
TEMPLATE = os.getenv('TEMPLATE', 'qwen3')

def _parse_endpoints(spec: str):
    out = []
    for raw in (spec or "").split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"SUB_AGENT_ENDPOINTS item missing port: {item!r}")
        host, port = item.rsplit(":", 1)
        out.append((host.strip(), int(port)))
    return out


_DEFAULT_ENDPOINTS = [("127.0.0.1", p) for p in (6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008)]
_ENDPOINTS = _parse_endpoints(os.getenv("SUB_AGENT_ENDPOINTS", "")) or _DEFAULT_ENDPOINTS
_endpoint_cycle = itertools.cycle(_ENDPOINTS)
_endpoint_lock = threading.Lock()

SUB_AGENT_MODEL_HF = os.getenv(
    'SUB_AGENT_MODEL_HF', 'Qwen/Qwen3-30B-A3B')
SUB_AGENT_TOKENIZE_RETRIES = int(os.getenv('SUB_AGENT_TOKENIZE_RETRIES', 10))
SUB_AGENT_TOKENIZE_TIMEOUT = float(os.getenv('SUB_AGENT_TOKENIZE_TIMEOUT', '5'))

_HF_TOKENIZER = None
_HF_TOKENIZER_LOCK = threading.Lock()
_MAX_MODEL_LEN_CACHE = None


def _get_hf_tokenizer():
    global _HF_TOKENIZER
    if _HF_TOKENIZER is None:
        with _HF_TOKENIZER_LOCK:
            if _HF_TOKENIZER is None:
                from transformers import AutoTokenizer
                print(f"[SubAgent] Loading HF tokenizer for fallback: {SUB_AGENT_MODEL_HF}")
                _HF_TOKENIZER = AutoTokenizer.from_pretrained(
                    SUB_AGENT_MODEL_HF, trust_remote_code=True)
    return _HF_TOKENIZER


def _count_via_local_tokenizer(messages, tools=None):
    tok = _get_hf_tokenizer()
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools is not None:
        kwargs["tools"] = tools
    prompt_str = tok.apply_chat_template(messages, **kwargs)
    return len(tok(prompt_str, add_special_tokens=False).input_ids)


def count_messages_tokens(messages, tools=None):
    """Count tokens via vLLM /tokenize; fall back to the local HF tokenizer."""
    global _MAX_MODEL_LEN_CACHE
    payload = {
        "messages": messages,
        "add_generation_prompt": True,
        "add_special_tokens": False,
    }
    if tools is not None:
        payload["tools"] = tools
    last_err = None
    for attempt in range(SUB_AGENT_TOKENIZE_RETRIES):
        with _endpoint_lock:
            host, port = next(_endpoint_cycle)
        try:
            r = requests.post(
                f"http://{host}:{port}/tokenize",
                json=payload,
                timeout=SUB_AGENT_TOKENIZE_TIMEOUT,
            )
            r.raise_for_status()
            body = r.json()
            max_model_len = int(body["max_model_len"])
            _MAX_MODEL_LEN_CACHE = max_model_len
            return int(body["count"]), max_model_len
        except Exception as e:
            last_err = e
            if attempt < SUB_AGENT_TOKENIZE_RETRIES - 1:
                time.sleep(min(0.2 * (2 ** attempt), 2.0))
    print(f"[SubAgent] /tokenize failed after {SUB_AGENT_TOKENIZE_RETRIES} "
          f"attempts (last: {type(last_err).__name__}: {last_err}); "
          f"falling back to local HF tokenizer")
    return _count_via_local_tokenizer(messages, tools=tools), (
        _MAX_MODEL_LEN_CACHE or SUB_AGENT_MAX_CONTEXT_TOKENS)

FORCE_ANSWER_PROMPT = (
    "You have reached the limit for this sub-task. Stop making tool calls "
    "and emit your final delivery turn now: a single <report>...</report> "
    "block, formatted exactly as the system instructions require (with inline "
    "citations [n] and a References section at the end of the <report>). Be "
    "honest about uncertainty; prefer to say less than to include incorrect "
    "claims."
)

# Sentinel returned by _call_llm_* when the LLM retry loop is fully exhausted.
# Downstream (_run_structured / _run_xml) must detect this and mark the
# sub-agent's status as 'error' rather than 'completed'.
_LLM_FAILURE_SENTINEL = "LLM call failed after all retries."
_NO_REPORT_SENTINEL = "(Sub-agent returned no usable content.)"


def _extract_report_or_sentinel(text):
    if text and '<report>' in text and '</report>' in text:
        return text.split('<report>', 1)[1].split('</report>', 1)[0].strip()
    return _NO_REPORT_SENTINEL


def _append_user_prompt(messages, prompt):
    """Append prompt as a user turn, merging with the prior user if present."""
    if messages and messages[-1].get("role") == "user":
        content = messages[-1].get("content") or ""
        messages[-1]["content"] = (content.rstrip() + "\n" + prompt).strip()
        return
    messages.append({"role": "user", "content": prompt})

# =============================================================================
# Trajectory logging — single append-only jsonl; one line per sub-agent run
# =============================================================================

_TRAJECTORY_LOCK = threading.Lock()


def _trajectory_log_path():
    """Return the path of the single append-only sub-agent trajectory log."""
    output_base = os.environ.get('OUTPUT_PATH', './results')
    model_path = os.environ.get('MODEL_PATH', 'model')
    model_name = os.path.basename(model_path.rstrip('/')) or 'model'
    experiment = os.environ.get('EXPERIMENT_NAME', '')
    dirname = f"{model_name}_{experiment}" if experiment else model_name
    return os.path.join(output_base, dirname, 'subagent_trajectories.jsonl')


def _write_trajectory(record):
    """Append one sub-agent run record as a single jsonl line.

    Safe under both intra-process concurrency (threading.Lock) and
    cross-process concurrency (fcntl.flock). Cross-process matters because
    run_multi_react.py supports multi-process via WORLD_SIZE/RANK and
    total_splits/worker_split; records are often tens to hundreds of KB
    (much larger than PIPE_BUF), so naive O_APPEND is NOT atomic across
    processes and would interleave into corrupted jsonl lines.
    """
    try:
        line = json.dumps(record, ensure_ascii=False)
    except Exception as e:
        print(f"[subagent trajectory] serialize failed: {e}")
        return
    path = _trajectory_log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception as e:
        print(f"[subagent trajectory] mkdir failed: {e}")
        return
    with _TRAJECTORY_LOCK:
        try:
            with open(path, 'a', encoding='utf-8') as f:
                try:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        f.write(line + '\n')
                        f.flush()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    # fcntl unavailable (e.g., Windows) or filesystem does
                    # not support advisory locking (some network FS).
                    # Fall back to plain append; single-process runs are
                    # still safe via _TRAJECTORY_LOCK above.
                    f.write(line + '\n')
        except Exception as e:
            print(f"[subagent trajectory] write failed: {e}")


# =============================================================================
# SubAgent: lightweight agent loop with search + visit
# =============================================================================

class SubAgent:

    _local_clients = {}

    def __init__(self, tool_map=None):
        self.mode = SUB_AGENT_MODE
        self._tool_map = tool_map or {}
        self._model_uses_reasoning = False
        self._searched_queries = []
        self._llm_calls_used = 0

        if self.mode == 'api':
            self._setup_api()

        print(f"[SubAgent] Initialized: mode={self.mode}, template={TEMPLATE}, "
              f"context={SUB_AGENT_MAX_CONTEXT_TOKENS}, max_calls={SUB_AGENT_MAX_LLM_CALLS}")

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------

    def _setup_api(self):
        from llm_client import TokenizerRegistry, get_model_config
        self._model_config = get_model_config(SUB_AGENT_MODEL)
        api_base = os.environ.get("API_BASE_URL", "http://localhost:8000")
        api_key = os.environ.get("API_KEY", "")
        self._api_client = OpenAI(
            base_url=api_base.rstrip("/") + "/v1",
            api_key=api_key,
            timeout=600.0,
        )
        print(f"[SubAgent] API mode: model={SUB_AGENT_MODEL}")

    @classmethod
    def _get_local_client(cls, host, port):
        key = (host, port)
        if key not in cls._local_clients:
            cls._local_clients[key] = OpenAI(
                api_key="EMPTY",
                base_url=f"http://{host}:{port}/v1",
                timeout=600.0,
            )
        return cls._local_clients[key]

    @staticmethod
    def _next_endpoint():
        with _endpoint_lock:
            return next(_endpoint_cycle)

    # -----------------------------------------------------------------
    # Tool execution (with search query tracking)
    # -----------------------------------------------------------------

    def _execute_tool(self, tool_name, tool_args):
        # Hard-reject recursive sub-agent dispatch. The defensive _tool_map copy
        # in CallSubAgent.__init__ should already prevent this at lookup time,
        # but keep an explicit guard in case future wiring reintroduces it.
        if tool_name == 'call_sub_agent':
            return "Error: sub-agents cannot dispatch sub-agents (call_sub_agent is not available here)."

        # Track search queries for structured return
        if tool_name == 'search':
            query = tool_args.get('query', [])
            if isinstance(query, str):
                self._searched_queries.append(query)
            elif isinstance(query, list):
                self._searched_queries.extend(query)

        if tool_name == 'PythonInterpreter' and tool_name in self._tool_map:
            code = tool_args.get('code', '')
            return self._tool_map[tool_name].call(code)

        if tool_name in self._tool_map:
            args_copy = dict(tool_args)
            args_copy["params"] = args_copy
            return self._tool_map[tool_name].call(args_copy)
        return f"Error: Tool '{tool_name}' not available. Available: {list(self._tool_map.keys())}"

    # -----------------------------------------------------------------
    # Message helpers
    # -----------------------------------------------------------------

    def _make_assistant_msg(self, content, tool_calls=None, reasoning=None):
        msg = {"role": "assistant", "content": content or ""}
        if self._model_uses_reasoning:
            if reasoning:
                msg["reasoning_content"] = reasoning
            elif not tool_calls:
                msg["reasoning_content"] = "."
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    # -----------------------------------------------------------------
    # Result parsing (returns structured dict)
    # -----------------------------------------------------------------

    def _parse_result(self, text, messages=None, status='completed'):
        content = text.strip() if text else ""
        if not content:
            content = "(Sub-agent returned no output.)"
        return {
            "content": content,
            "messages": list(messages) if messages is not None else [],
            "queries": list(self._searched_queries),
            "llm_calls": self._llm_calls_used,
            "status": status,
        }

    # -----------------------------------------------------------------
    # LLM calling: structured (API + local hermes_w_py)
    # -----------------------------------------------------------------

    def _call_llm_structured(self, client, model, messages, tools=None,
                             use_tools=True, max_tokens=None, is_api=False,
                             max_tries=50):
        """Returns (content, tool_calls, usage, reasoning, finish_reason)."""
        base_sleep = 1
        for attempt in range(max_tries):
            try:
                params = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens or SUB_AGENT_MAX_GENERATION_TOKENS,
                }

                if use_tools and tools:
                    params["tools"] = tools

                if is_api:
                    cfg = self._model_config
                    drop = cfg.get("drop_params", set())
                    if "temperature" not in drop:
                        params["temperature"] = cfg.get("force_temperature") or SUB_AGENT_TEMPERATURE
                    if "presence_penalty" not in drop:
                        params["presence_penalty"] = SUB_AGENT_PRESENCE_PENALTY
                    if "logprobs" not in drop:
                        params["logprobs"] = True
                    extra_body = cfg.get("extra_body")
                    if extra_body:
                        params["extra_body"] = extra_body
                else:
                    params["temperature"] = SUB_AGENT_TEMPERATURE
                    params["top_p"] = SUB_AGENT_TOP_P
                    params["presence_penalty"] = SUB_AGENT_PRESENCE_PENALTY
                    params["logprobs"] = True
                    params["reasoning_effort"] = "high"

                print(f"[SubAgent] LLM structured call attempt {attempt + 1}/{max_tries}")
                response = client.chat.completions.create(**params)

                msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                content = msg.content or ""

                tool_calls = None
                if msg.tool_calls:
                    tool_calls = [{
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    } for tc in msg.tool_calls]

                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }

                reasoning = getattr(msg, 'reasoning_content', None)
                if reasoning is not None:
                    self._model_uses_reasoning = True

                if content or tool_calls or reasoning:
                    print(f"[SubAgent] LLM OK (pt={usage.get('prompt_tokens', '?')}, "
                          f"ct={usage.get('completion_tokens', '?')}, finish={finish_reason})")
                    return content, tool_calls, usage, reasoning, finish_reason
                else:
                    print(f"[SubAgent] Empty response attempt {attempt + 1}/{max_tries}")

            except RateLimitError as e:
                sleep = min(base_sleep * (2 ** min(attempt, 6)) + random.uniform(0, 1), 60)
                print(f"[SubAgent] RateLimit: {e}, sleeping {sleep:.1f}s")
                time.sleep(sleep)
                continue
            except BadRequestError as e:
                print(f"[SubAgent] 400 from server (non-retryable): "
                      f"{str(e)[:300]}")
                return "", None, {}, None, "exhausted"
            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"[SubAgent] API error attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"[SubAgent] Unexpected error attempt {attempt + 1}: {e}")

            if attempt < max_tries - 1:
                sleep = min(base_sleep * (2 ** attempt) + random.uniform(0, 1), 30)
                time.sleep(sleep)

        # Exhausted all retries. Signal via finish_reason="exhausted" so
        # _run_structured marks status='error' rather than 'completed'.
        print(f"[SubAgent] All {max_tries} LLM attempts exhausted.")
        return "", None, {}, None, "exhausted"

    # -----------------------------------------------------------------
    # LLM calling: XML (local non-hermes_w_py, i.e., TEMPLATE=qwen3)
    # -----------------------------------------------------------------

    def _call_llm_xml(self, client, model, messages, max_tokens=None, max_tries=50):
        """Returns (content, prompt_tokens)."""
        base_sleep = 1
        for attempt in range(max_tries):
            try:
                params = {
                    "model": model,
                    "messages": messages,
                    "stop": ["\n<tool_response>", "<tool_response>"],
                    "temperature": SUB_AGENT_TEMPERATURE,
                    "top_p": SUB_AGENT_TOP_P,
                    "logprobs": True,
                    "max_tokens": max_tokens or SUB_AGENT_MAX_GENERATION_TOKENS,
                    "presence_penalty": SUB_AGENT_PRESENCE_PENALTY,
                    "reasoning_effort": "high",
                }

                print(f"[SubAgent] LLM XML call attempt {attempt + 1}/{max_tries}")
                response = client.chat.completions.create(**params)
                content = response.choices[0].message.content

                prompt_tokens = 0
                if response.usage:
                    prompt_tokens = response.usage.prompt_tokens

                if content and content.strip():
                    print(f"[SubAgent] LLM XML OK (pt={prompt_tokens})")
                    return content.strip(), prompt_tokens
                else:
                    print(f"[SubAgent] Empty XML response attempt {attempt + 1}")

            except BadRequestError as e:
                print(f"[SubAgent] 400 from vLLM (non-retryable): "
                      f"{str(e)[:300]}")
                return _LLM_FAILURE_SENTINEL, 0
            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"[SubAgent] API error attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"[SubAgent] Unexpected error attempt {attempt + 1}: {e}")

            if attempt < max_tries - 1:
                sleep = min(base_sleep * (2 ** attempt) + random.uniform(0, 1), 30)
                time.sleep(sleep)

        return _LLM_FAILURE_SENTINEL, 0

    # =================================================================
    # Main entry point
    # =================================================================

    def run(self, prompt, main_model=None):
        self._searched_queries = []
        self._llm_calls_used = 0
        start_time = time.time()

        preview = prompt.replace('\n', ' ')[:200]
        print(f"[SubAgent] Starting task ({self.mode}): {preview}...")

        if self.mode == 'local':
            host, port = self._next_endpoint()
            # Prefer SUB_AGENT_MODEL if explicitly set — lets the sub-agent use
            # a different model name than the main agent (e.g. an API main agent
            # with a local sub-agent). Falls back to main_model when main and
            # sub share one local vLLM deployment.
            model = SUB_AGENT_MODEL or main_model
            if not model:
                result = self._error_result("sub-agent local mode requires SUB_AGENT_MODEL or main_model.")
            else:
                client = self._get_local_client(host, port)
                print(f"[SubAgent] Local mode: endpoint={host}:{port}, model={model}, "
                      f"template={TEMPLATE}")
                # qwen3 XML parsing, upgraded to execute every <tool_call>
                # block in a single assistant turn.
                result = self._run_xml(
                    client, model, prompt,
                    multi_tool=True)
        else:
            if not SUB_AGENT_MODEL:
                result = self._error_result(
                    "SUB_AGENT_MODEL must be set for API mode sub-agent.")
            else:
                print(f"[SubAgent] API mode: model={SUB_AGENT_MODEL}")
                result = self._run_structured(self._api_client, SUB_AGENT_MODEL,
                                              prompt, is_api=True)

        result['duration_ms'] = int((time.time() - start_time) * 1000)
        return result

    def _error_result(self, msg):
        return {
            "content": f"Error: {msg}",
            "messages": [],
            "queries": [],
            "llm_calls": 0,
            "status": "error",
        }

    # =================================================================
    # Structured tool-calling loop (API + local hermes_w_py)
    # =================================================================

    def _run_structured(self, client, model, prompt, is_api=False):
        loop_start = time.time()
        cur_date = datetime.date.today().strftime("%Y-%m-%d")

        system_prompt = render_sub_agent_system_prompt(cur_date, include_tools=False)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        tools = SUB_AGENT_OPENAI_TOOLS
        prompt_token_limit = SUB_AGENT_MAX_CONTEXT_TOKENS - SUB_AGENT_MAX_GENERATION_TOKENS
        num_calls = SUB_AGENT_MAX_LLM_CALLS
        round_num = 0

        while num_calls > 0:
            if time.time() - loop_start > SUB_AGENT_TIMEOUT_MINUTES * 60:
                print(f"[SubAgent] Timeout after {SUB_AGENT_TIMEOUT_MINUTES} min")
                return self._force_answer_structured(
                    messages, client, model, is_api, status='timeout')

            round_num += 1
            num_calls -= 1
            messages_before = deepcopy(messages)

            content, tool_calls, usage, reasoning, finish_reason = \
                self._call_llm_structured(
                    client, model, messages, tools=tools, is_api=is_api)
            self._llm_calls_used += 1

            tc_info = f" [+{len(tool_calls)} tool_calls]" if tool_calls else ""
            print(f"[SubAgent] Round {round_num}: "
                  f"{(content or '')[:200]}"
                  f"{'...' if content and len(content) > 200 else ''}"
                  f"{tc_info}")

            pt = usage.get("prompt_tokens", 0)
            if pt >= prompt_token_limit or finish_reason == "length":
                print(f"[SubAgent] Token limit hit (pt={pt}/{prompt_token_limit}, "
                      f"finish={finish_reason}). Rolling back.")
                return self._force_answer_structured(
                    messages_before, client, model, is_api, status='token_limit')

            messages.append(self._make_assistant_msg(
                content, tool_calls=tool_calls, reasoning=reasoning))

            if not tool_calls:
                final = content or reasoning or ""
                if finish_reason == "exhausted":
                    return self._parse_result(final, messages=messages, status='error')
                report = _extract_report_or_sentinel(final)
                if report == _NO_REPORT_SENTINEL:
                    return self._parse_result(
                        _NO_REPORT_SENTINEL, messages=messages, status='error')
                return self._parse_result(
                    report, messages=messages, status='completed')

            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                try:
                    tool_args = json.loads(tc["function"]["arguments"])
                except Exception:
                    tool_args = {}
                try:
                    result = self._execute_tool(tool_name, tool_args)
                except Exception as e:
                    result = (f"[Tool Error] {tool_name}: {type(e).__name__}: "
                              f"{e!r}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            if not is_api:
                try:
                    next_input_count, max_model_len = count_messages_tokens(
                        messages, tools=tools)
                    next_limit = max_model_len - SUB_AGENT_MAX_GENERATION_TOKENS
                    if next_input_count > next_limit:
                        print(f"[SubAgent] Pre-call overflow: pt={next_input_count} "
                              f"> y-x={next_limit}. Rolling back.")
                        return self._force_answer_structured(
                            messages_before, client, model, is_api,
                            status='token_limit')
                except Exception as e:
                    print(f"[SubAgent] count_messages_tokens unrecoverable: "
                          f"{type(e).__name__}: {e}. Rolling back.")
                    return self._force_answer_structured(
                        messages_before, client, model, is_api,
                        status='token_limit')

            if num_calls <= 1:
                print("[SubAgent] Call limit approaching, forcing answer.")
                return self._force_answer_structured(
                    messages, client, model, is_api, status='max_calls')

        last = messages[-1].get("content", "") if messages else ""
        return self._parse_result(last, messages=messages, status='max_calls')

    def _force_answer_structured(self, messages, client, model, is_api, status='force_answer'):
        _append_user_prompt(messages, FORCE_ANSWER_PROMPT)

        max_tokens = SUB_AGENT_MAX_GENERATION_TOKENS
        if not is_api:
            try:
                final_count, max_model_len = count_messages_tokens(messages)
                if final_count + max_tokens > max_model_len:
                    max_tokens = max(1, max_model_len - final_count - 1)
                    print(f"[SubAgent] Force-answer clamp: pt={final_count}, "
                          f"adjusted_max_tokens={max_tokens}")
            except Exception as e:
                print(f"[SubAgent] Force-answer count_messages_tokens failed: "
                      f"{type(e).__name__}: {e}. Using default max_tokens.")

        content, _, _, reasoning, _ = self._call_llm_structured(
            client, model, messages, use_tools=False, is_api=is_api,
            max_tokens=max_tokens)
        self._llm_calls_used += 1
        messages.append(self._make_assistant_msg(content, reasoning=reasoning))
        final = content or reasoning or ""
        report = _extract_report_or_sentinel(final)
        if report == _NO_REPORT_SENTINEL:
            return self._parse_result(
                _NO_REPORT_SENTINEL, messages=messages, status='error')
        return self._parse_result(report, messages=messages, status=status)

    # =================================================================
    # XML tool-calling loop (local qwen3-style XML). One assistant turn may
    # contain multiple <tool_call> blocks; execute all and return one combined
    # tool_response message.
    # =================================================================

    def _run_xml(self, client, model, prompt, multi_tool=True):
        loop_start = time.time()
        cur_date = datetime.date.today().strftime("%Y-%m-%d")

        system_prompt = render_sub_agent_system_prompt(cur_date, include_tools=True)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        prompt_token_limit = SUB_AGENT_MAX_CONTEXT_TOKENS - SUB_AGENT_MAX_GENERATION_TOKENS
        num_calls = SUB_AGENT_MAX_LLM_CALLS
        round_num = 0

        while num_calls > 0:
            if time.time() - loop_start > SUB_AGENT_TIMEOUT_MINUTES * 60:
                print(f"[SubAgent] Timeout after {SUB_AGENT_TIMEOUT_MINUTES} min")
                return self._force_answer_xml(messages, client, model, status='timeout')

            round_num += 1
            num_calls -= 1
            messages_before = [m.copy() for m in messages]

            content, prompt_tokens = self._call_llm_xml(client, model, messages)
            self._llm_calls_used += 1

            print(f"[SubAgent] Round {round_num}: "
                  f"{content[:200]}"
                  f"{'...' if len(content) > 200 else ''}")

            if prompt_tokens >= prompt_token_limit:
                print(f"[SubAgent] Token limit hit (pt={prompt_tokens}/{prompt_token_limit}). "
                      f"Rolling back.")
                return self._force_answer_xml(messages_before, client, model, status='token_limit')

            if '<tool_response>' in content:
                content = content[:content.find('<tool_response>')]

            messages.append({"role": "assistant", "content": content.strip()})

            has_tool_call = '<tool_call>' in content and '</tool_call>' in content
            if has_tool_call:
                if multi_tool:
                    parsed_blocks = parse_tool_call_blocks(content)
                    results = []
                    for blk in parsed_blocks:
                        if blk['kind'] == 'python':
                            try:
                                r = self._execute_tool('PythonInterpreter', {'code': blk['code']})
                            except Exception as e:
                                r = f"[Python Interpreter Error]: {e}"
                        elif blk['kind'] == 'json':
                            try:
                                r = self._execute_tool(blk['name'], blk['arguments'])
                            except Exception as e:
                                r = f"Error executing tool call: {e}"
                        else:  # 'bad_json'
                            r = f"Error executing tool call: {blk['error']}"
                        results.append(r)
                    combined = "\n".join(
                        f"<tool_response>\n{r}\n</tool_response>" for r in results)
                    messages.append({"role": "user", "content": combined})
                else:
                    tool_call_str = content.split('<tool_call>')[1].split('</tool_call>')[0]
                    try:
                        tc = json5.loads(tool_call_str)
                        tool_name = tc.get('name', '')
                        tool_args = tc.get('arguments', {})
                        result = self._execute_tool(tool_name, tool_args)
                    except Exception as e:
                        result = f"Error executing tool call: {e}"

                    messages.append({
                        "role": "user",
                        "content": f"<tool_response>\n{result}\n</tool_response>",
                    })
            else:
                if content == _LLM_FAILURE_SENTINEL:
                    return self._parse_result(
                        content, messages=messages, status='error')
                report = _extract_report_or_sentinel(content)
                if report == _NO_REPORT_SENTINEL:
                    return self._parse_result(
                        _NO_REPORT_SENTINEL, messages=messages, status='error')
                return self._parse_result(
                    report, messages=messages, status='completed')

            try:
                next_input_count, max_model_len = count_messages_tokens(messages)
                next_limit = max_model_len - SUB_AGENT_MAX_GENERATION_TOKENS
                if next_input_count > next_limit:
                    print(f"[SubAgent] Pre-call overflow: pt={next_input_count} "
                          f"> y-x={next_limit}. Rolling back.")
                    return self._force_answer_xml(
                        messages_before, client, model, status='token_limit')
            except Exception as e:
                print(f"[SubAgent] count_messages_tokens unrecoverable: "
                      f"{type(e).__name__}: {e}. Rolling back.")
                return self._force_answer_xml(
                    messages_before, client, model, status='token_limit')

            if num_calls <= 1:
                print("[SubAgent] Call limit approaching, forcing answer.")
                return self._force_answer_xml(messages, client, model, status='max_calls')

        last = messages[-1].get("content", "") if messages else ""
        return self._parse_result(last, messages=messages, status='max_calls')

    def _force_answer_xml(self, messages, client, model, status='force_answer'):
        _append_user_prompt(messages, FORCE_ANSWER_PROMPT)

        max_tokens = SUB_AGENT_MAX_GENERATION_TOKENS
        try:
            final_count, max_model_len = count_messages_tokens(messages)
            if final_count + max_tokens > max_model_len:
                max_tokens = max(1, max_model_len - final_count - 1)
                print(f"[SubAgent] Force-answer clamp: pt={final_count}, "
                      f"adjusted_max_tokens={max_tokens}")
        except Exception as e:
            print(f"[SubAgent] Force-answer count_messages_tokens failed: "
                  f"{type(e).__name__}: {e}. Using default max_tokens.")

        content, _ = self._call_llm_xml(client, model, messages, max_tokens=max_tokens)
        self._llm_calls_used += 1
        messages.append({"role": "assistant", "content": content.strip() if content else ""})
        if content == _LLM_FAILURE_SENTINEL:
            return self._parse_result(content, messages=messages, status='error')
        report = _extract_report_or_sentinel(content or "")
        if report == _NO_REPORT_SENTINEL:
            return self._parse_result(
                _NO_REPORT_SENTINEL, messages=messages, status='error')
        return self._parse_result(report, messages=messages, status=status)


# =============================================================================
# Tool registration
# =============================================================================

@register_tool('call_sub_agent', allow_overwrite=True)
class CallSubAgent(BaseTool):
    name = "call_sub_agent"
    description = (
        "Dispatch research sub-tasks to independent agents running in parallel. "
        "Each agent can search the web and visit webpages. "
        "Coordinate each sub-agent as a new research collaborator joining the investigation for the first time. "
        "Make the division of labor explicit: what to investigate or verify, what evidence would be useful, "
        "and what result you need back. Then give the background needed to avoid wasted effort or the wrong target: "
        "why this sub-task matters, what is already established, what remains uncertain, which leads have been tried "
        "or ruled out, and where the weak points or contradictions are. Keep hypotheses, confirmed facts, and open gaps clearly separated. "
        "IMPORTANT: the sub-agent sees only the `prompt` field; the `goal` field is used only to label the sub-agent's response when it comes back to you."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": (
                                "A concrete research assignment for one sub-agent. "
                                "State the task, expected output, useful evidence, context for the larger question, "
                                "relevant constraints, and evidence state. Every sub-agent reads only its own brief \u2014 "
                                "do NOT reduce detail in later briefs just because the first one was thorough."
                            ),
                        },
                        "goal": {
                            "type": "string",
                            "description": (
                                "A short one-line objective for this sub-task, used only to label the sub-agent's response when it returns. "
                                "The sub-agent itself does not see this field."
                            ),
                        },
                    },
                    "required": ["prompt", "goal"],
                },
                "minItems": 1,
                "description": "A list of {prompt, goal} objects. Each object spawns one independent sub-agent; they run in parallel.",
            }
        },
        "required": ["prompts"],
    }

    def __init__(self, cfg: Optional[dict] = None, tool_map: Optional[dict] = None):
        super().__init__(cfg)
        # Snapshot (do NOT store the reference): the caller (react_agent.py)
        # passes the global TOOL_MAP and then mutates it to add 'call_sub_agent'
        # itself AFTER constructing us. Keeping a reference would leak that
        # mutation into sub-agents' view, enabling recursive dispatch.
        self._tool_map = dict(tool_map) if tool_map else {}

    def call(self, params: Union[str, dict], **kwargs) -> str:
        from tool_search import ToolCallFormatError

        if not isinstance(params, dict):
            raise ToolCallFormatError(
                f"[call_sub_agent] Invalid arguments: expected an object, got {type(params).__name__}"
            )

        if "prompts" not in params:
            raise ToolCallFormatError(
                "[call_sub_agent] Invalid arguments: must provide 'prompts' array. "
                f"Received keys: {list(params.keys())}"
            )

        entries = params["prompts"]
        if not isinstance(entries, list):
            raise ToolCallFormatError(
                f"[call_sub_agent] 'prompts' must be an array, got {type(entries).__name__}"
            )
        if not entries:
            raise ToolCallFormatError(
                "[call_sub_agent] 'prompts' array is empty."
            )

        for i, e in enumerate(entries):
            if not isinstance(e, dict):
                raise ToolCallFormatError(
                    f"[call_sub_agent] Invalid item at index {i}: "
                    f"must be an object with 'prompt' and 'goal' fields. Got: {type(e).__name__}"
                )
            if "prompt" not in e or not isinstance(e["prompt"], str) or not e["prompt"].strip():
                raise ToolCallFormatError(
                    f"[call_sub_agent] Item at index {i} is missing a non-empty 'prompt' string."
                )
            if "goal" not in e or not isinstance(e["goal"], str) or not e["goal"].strip():
                raise ToolCallFormatError(
                    f"[call_sub_agent] Item at index {i} is missing a non-empty 'goal' string."
                )

        main_model = kwargs.get('model')
        question = kwargs.get('question', '')

        print(f"[call_sub_agent] Dispatching {len(entries)} sub-agent(s) "
              f"{'in parallel' if len(entries) > 1 else ''}...")

        if len(entries) == 1:
            agent = SubAgent(tool_map=self._tool_map)
            try:
                result = agent.run(entries[0]["prompt"], main_model=main_model)
            except Exception as e:
                print(f"[call_sub_agent] Sub-agent failed: {e}")
                result = {
                    "content": f"Sub-agent error: {type(e).__name__}: {e}",
                    "messages": [],
                    "queries": [],
                    "llm_calls": 0,
                    "status": "error",
                    "duration_ms": 0,
                }
            self._log_trajectory(question, entries[0], result)
            return self._format_results(entries, [result])

        # Parallel dispatch for multiple sub-agents.
        # Log each trajectory inside the as_completed loop so completed work
        # is persisted incrementally — if the process dies mid-dispatch we
        # don't lose every sub-agent's trajectory.
        results = [None] * len(entries)
        with ThreadPoolExecutor(max_workers=len(entries)) as pool:
            futures = {}
            for i, e in enumerate(entries):
                agent = SubAgent(tool_map=self._tool_map)
                future = pool.submit(agent.run, e["prompt"], main_model)
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"[call_sub_agent] Sub-agent {idx + 1} failed: {e}")
                    result = {
                        "content": f"Sub-agent error: {type(e).__name__}: {e}",
                        "messages": [],
                        "queries": [],
                        "llm_calls": 0,
                        "status": "error",
                        "duration_ms": 0,
                    }
                results[idx] = result
                self._log_trajectory(question, entries[idx], result)

        return self._format_results(entries, results)

    @staticmethod
    def _log_trajectory(question, entry, result):
        record = {
            "timestamp_ms": int(time.time() * 1000),
            "question": question,
            "goal": entry.get("goal", ""),
            "prompt": entry.get("prompt", ""),
            "messages": result.get("messages", []),
            "queries": result.get("queries", []),
            "llm_calls": result.get("llm_calls", 0),
            "status": result.get("status", "unknown"),
            "duration_ms": result.get("duration_ms", 0),
            "content": result.get("content", ""),
        }
        _write_trajectory(record)

    @staticmethod
    def _format_results(entries, results):
        """Render sub-agent reports into the string returned to the main
        agent as a tool_response.

        Format (same template for 1 and N sub-agents; blocks separated by
        a horizontal rule):

            A sub-agent dispatched for goal "{goal}" returned the
            following report:

            {content}

            ---

            A sub-agent dispatched for goal "{goal2}" returned the
            following report:

            {content2}
            ...

        Design notes:
          - Goal quoted with double-quotes so goals that contain spaces,
            commas, or colons don't blur into the surrounding prose.
          - Empty goal falls back to "(unspecified)".
          - Empty content falls back to "(Sub-agent returned no output.)"
            so callers can still distinguish a silent sub-agent from
            e.g. an error message.
          - Errors ride through the same template — `content` is set to
            "Sub-agent error: ..." upstream, which reads fine under
            "returned the following report:".
          - Same template for single and multi dispatch so the trainee
            model sees one consistent pattern rather than two.
        """
        parts = []
        for entry, result in zip(entries, results):
            goal = entry.get("goal", "").strip() or "(unspecified)"
            content = result.get("content", "").strip() or "(Sub-agent returned no output.)"
            parts.append(
                f'A sub-agent dispatched for goal "{goal}" returned the following report:\n\n{content}'
            )
        return "\n\n---\n\n".join(parts)
