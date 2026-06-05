import json
import os
import random
import time
import datetime
import threading
from copy import deepcopy
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

from qwen_agent.llm.schema import Message
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, BadRequestError, RateLimitError
from transformers import AutoTokenizer
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from prompt import *

from hermes_w_py_parser import parse_tool_call_blocks
import main_checkpoint as _main_ckpt
from tool_search import *
from tool_visit import *

TOOL_TYPE = os.getenv('TOOL_TYPE', 'four')
if TOOL_TYPE == 'four':
    from tool_scholar import *
    from tool_python import *

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MODEL_MODE = os.getenv('MODEL_MODE', 'local')
TEMPLATE = os.getenv('TEMPLATE', 'qwen3')
MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))
USE_CONTEXT_MANAGEMENT = os.getenv('USE_CONTEXT_MANAGEMENT', 'False').lower() in ('true', '1', 'yes')
RESTORE_TOOL_OBSERVATION = int(os.getenv('RESTORE_TOOL_OBSERVATION', 5))
RUN_TIMEOUT_MINUTES = int(os.getenv('RUN_TIMEOUT_MINUTES', 150))
MAX_CONTEXT_TOKENS = int(os.getenv('MAX_CONTEXT_TOKENS', 128 * 1024))
MAX_GENERATION_TOKENS = int(os.getenv('MAX_GENERATION_TOKENS', 8192))
TOKEN_COUNTER = os.getenv('TOKEN_COUNTER', 'local')

_empty_resp_lock = threading.Lock()
_empty_resp_count = 0
_EMPTY_RESP_LOG = "empty_response_debug.jsonl"
_EMPTY_RESP_MAX_LOG = 100


def _log_empty_response(planning_port, msgs, response_msg, usage, finish_reason):
    global _empty_resp_count
    with _empty_resp_lock:
        _empty_resp_count += 1
        seq = _empty_resp_count
        if seq > _EMPTY_RESP_MAX_LOG:
            return
    try:
        raw_msg = response_msg.model_dump(mode="json") if hasattr(response_msg, "model_dump") else str(response_msg)
    except Exception:
        raw_msg = str(response_msg)
    entry = {
        "seq": seq,
        "timestamp": datetime.datetime.now().isoformat(),
        "port": planning_port,
        "finish_reason": finish_reason,
        "usage": usage,
        "content_repr": repr(response_msg.content),
        "has_tool_calls": response_msg.tool_calls is not None,
        "has_reasoning_content": hasattr(response_msg, "reasoning_content"),
        "reasoning_content_value": repr(getattr(response_msg, "reasoning_content", "<MISSING>")),
        "raw_message": raw_msg,
        "input_messages_count": len(msgs),
        "input_last_role": msgs[-1].get("role") if msgs else None,
        "input_last_content_preview": (msgs[-1].get("content") or "")[:500] if msgs else None,
    }
    try:
        with open(_EMPTY_RESP_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        print(f"[empty_resp_debug] #{seq}/{_EMPTY_RESP_MAX_LOG} logged to {_EMPTY_RESP_LOG}")
    except Exception as e:
        print(f"[empty_resp_debug] Failed to write log: {e}")


ENABLE_SUB_AGENT = os.getenv('ENABLE_SUB_AGENT', '0') == '1'

TOOL_CLASS = [
    Visit(),
    Search(),
]
if TOOL_TYPE == 'four':
    TOOL_CLASS.extend([Scholar(), PythonInterpreter()])
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

if ENABLE_SUB_AGENT:
    from tool_sub_agent import CallSubAgent
    _sub_agent_tool = CallSubAgent(tool_map=TOOL_MAP)
    TOOL_CLASS.append(_sub_agent_tool)
    TOOL_MAP[_sub_agent_tool.name] = _sub_agent_tool


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


def main_force_answer_prompt():
    if ENABLE_SUB_AGENT:
        return (
            "You have reached the limit for this task. Stop making tool calls "
            "and emit your final-delivery turn now: an "
            "<explanation>...</explanation> block followed by an "
            "<answer>...</answer> block, formatted exactly as the system "
            "instructions require. Put only the final answer itself inside "
            "<answer>. Be honest about uncertainty; prefer to say less than "
            "to include incorrect claims."
        )
    return (
        "You have now reached the maximum context length you can handle. You "
        "should stop making tool calls and, based on all the information above, "
        "think again and provide what you consider the most likely answer in "
        "the following format:<think>your final thinking</think>\n"
        "<answer>your answer</answer>"
    )


def filter_messages(messages, keep_last_n_users=5):
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not user_indices:
        return messages

    keep_user_indices = set()
    keep_user_indices.add(user_indices[0])
    keep_user_indices.update(user_indices[-keep_last_n_users:])

    filtered = []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role in ("assistant", "system", "tool"):
            filtered.append(m)
        elif role == "user" and i in keep_user_indices:
            filtered.append(m)
    return filtered


class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self.model_mode = MODEL_MODE

        if self.model_mode == 'api':
            from llm_client import TokenizerRegistry, get_model_config
            self._model_config = get_model_config(self.llm_local_path)
            self._tokenizer = TokenizerRegistry.get(self.llm_local_path)
            api_base = os.environ.get("API_BASE_URL", "http://localhost:8000")
            api_key = os.environ.get("API_KEY", "")
            self._api_client = OpenAI(
                base_url=api_base.rstrip("/") + "/v1",
                api_key=api_key,
                timeout=600.0,
            )
            self._openai_tools = OPENAI_TOOLS
            self._last_usage_prompt_tokens = 0
            self._model_uses_reasoning = False
            print(f"[Agent] API mode: model={self.llm_local_path}, tokenizer={'yes' if self._tokenizer else 'no (using API usage)'}, token_counter={TOKEN_COUNTER}")
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
            print(f"[Agent] Local mode (template={TEMPLATE}): model={self.llm_local_path}")

    # ==================================================================
    # Sanity check (local mode only)
    # ==================================================================

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content

    # ==================================================================
    # Local vLLM call (unchanged)
    # ==================================================================

    def call_server(self, msgs, planning_port, max_tries=128, max_tokens=None):
        client = self._get_local_client(planning_port)

        base_sleep_time = 1
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    logprobs=True,
                    max_tokens=max_tokens if max_tokens is not None else MAX_GENERATION_TOKENS,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1),
                    reasoning_effort="high"
                )
                content = chat_response.choices[0].message.content

                if content and content.strip():
                    print("--- Service call successful, received a valid response ---")
                    return content.strip()
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")

            except BadRequestError as e:
                print(f"Error: non-retryable 400 from vLLM: {str(e)[:300]}")
                return "vllm server error!!!"
            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")

        return "vllm server error!!!"

    # ==================================================================
    # Local vLLM call with structured tool calling (hermes_w_py template)
    # ==================================================================

    _local_clients = {}

    def _get_local_client(self, planning_port):
        if planning_port not in self._local_clients:
            self._local_clients[planning_port] = OpenAI(
                api_key="EMPTY",
                base_url=f"http://127.0.0.1:{planning_port}/v1",
                timeout=600.0,
            )
        return self._local_clients[planning_port]

    def call_server_tools(self, msgs, planning_port, tools=None, max_tries=128,
                          max_tokens=None, use_tools=True):
        """Call local vLLM with OpenAI-style tool calling. Returns same tuple as call_api."""
        client = self._get_local_client(planning_port)

        base_sleep_time = 1
        for attempt in range(max_tries):
            try:
                params = {
                    "model": self.model,
                    "messages": msgs,
                    "max_tokens": max_tokens if max_tokens is not None else MAX_GENERATION_TOKENS,
                    "temperature": self.llm_generate_cfg.get('temperature', 0.6),
                    "top_p": self.llm_generate_cfg.get('top_p', 0.95),
                    "logprobs": True,
                    "presence_penalty": self.llm_generate_cfg.get('presence_penalty', 1.1),
                    "reasoning_effort": "high",
                }

                if use_tools and tools:
                    params["tools"] = tools

                print(f"--- call_server_tools attempt {attempt + 1}/{max_tries} (port={planning_port}) ---")
                response = client.chat.completions.create(**params)

                msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                content = msg.content or ""

                tool_calls = None
                if msg.tool_calls:
                    tool_calls = [{
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    } for tc in msg.tool_calls]

                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                self._last_usage_prompt_tokens = usage.get("prompt_tokens", 0)

                reasoning = getattr(msg, 'reasoning_content', None)
                if reasoning is not None:
                    self._model_uses_reasoning = True

                if content or tool_calls or reasoning:
                    print(f"--- call_server_tools success (pt={usage.get('prompt_tokens','?')}, "
                          f"ct={usage.get('completion_tokens','?')}, finish={finish_reason}) ---")
                    return content, tool_calls, usage, reasoning, finish_reason
                else:
                    print(f"Warning: Attempt {attempt + 1} received empty response. "
                          f"finish={finish_reason}, ct={usage.get('completion_tokens','?')}, "
                          f"has_reasoning={hasattr(msg, 'reasoning_content')}, "
                          f"reasoning_repr={repr(getattr(msg, 'reasoning_content', '<MISSING>'))[:100]}")
                    _log_empty_response(planning_port, msgs, msg, usage, finish_reason)

            except BadRequestError as e:
                print(f"Error: non-retryable 400 from vLLM structured call: {str(e)[:300]}")
                return "", None, {}, None, "exhausted"
            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with API/network error: {e}")
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted.")

        return "vllm server error!!!", None, {}, None, None

    # ==================================================================
    # API call with retry
    # ==================================================================

    def call_api(self, msgs, max_tries=10, max_tokens=None, use_tools=True):
        """
        Call remote API model. Returns (content, tool_calls, usage_dict, reasoning_content, finish_reason).
        - content: str (may be empty if only tool_calls)
        - tool_calls: list of dicts or None
        - usage_dict: {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
        - reasoning_content: str or None
        - finish_reason: str or None ("stop", "length", "tool_calls", etc.)
        """
        cfg = self._model_config

        base_sleep_time = 1
        normal_attempts = 0

        while True:
            try:
                params = {
                    "model": self.model,
                    "messages": msgs,
                    "max_tokens": max_tokens if max_tokens is not None else MAX_GENERATION_TOKENS,
                }

                if use_tools and self._openai_tools:
                    params["tools"] = self._openai_tools

                drop = cfg.get("drop_params", set())
                if "temperature" not in drop:
                    temp = cfg.get("force_temperature") or self.llm_generate_cfg.get('temperature', 0.6)
                    params["temperature"] = temp
                if "presence_penalty" not in drop:
                    params["presence_penalty"] = self.llm_generate_cfg.get('presence_penalty', 0)
                if "logprobs" not in drop:
                    params["logprobs"] = True

                extra_body = cfg.get("extra_body")
                if extra_body:
                    params["extra_body"] = extra_body

                print(f"--- API call attempt {normal_attempts+1}/{max_tries} ---")
                response = self._api_client.chat.completions.create(**params)

                msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                content = msg.content or ""
                tool_calls = None
                if msg.tool_calls:
                    tool_calls = [{
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    } for tc in msg.tool_calls]

                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                self._last_usage_prompt_tokens = usage.get("prompt_tokens", 0)

                reasoning = getattr(msg, 'reasoning_content', None)
                if reasoning is not None:
                    self._model_uses_reasoning = True

                if content or tool_calls or reasoning:
                    print(f"--- API call successful (pt={usage.get('prompt_tokens','?')}, ct={usage.get('completion_tokens','?')}, finish={finish_reason}) ---")
                    return content, tool_calls, usage, reasoning, finish_reason
                else:
                    print(f"Warning: API returned empty content and no tool_calls. "
                          f"finish={finish_reason}, reasoning_len={len(reasoning or '')}, usage={usage} "
                          f"(attempt {normal_attempts + 1}/{max_tries})")
                    normal_attempts += 1

            except BadRequestError as e:
                print(f"Error: BadRequestError: {str(e)[:200]}")
                normal_attempts += 1

            except RateLimitError as e:
                sleep_time = min(base_sleep_time * (2 ** min(normal_attempts, 6)) + random.uniform(0, 1), 60)
                print(f"[RateLimit] {e} - sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                continue

            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: API/network error: {e}")
                normal_attempts += 1

            except Exception as e:
                print(f"Error: Unexpected: {e}")
                normal_attempts += 1

            if normal_attempts >= max_tries:
                print(f"Warning: All {max_tries} API retry attempts exhausted. Returning exhausted signal.")
                return "", None, {}, None, "exhausted"

            sleep_time = base_sleep_time * (2 ** min(normal_attempts - 1, 5)) + random.uniform(0, 1)
            sleep_time = min(sleep_time, 30)
            print(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

    # ==================================================================
    # Token counting
    # ==================================================================

    def count_tokens(self, messages, tools=None):
        if self.model_mode == 'local':
            ct_kwargs = {'add_generation_prompt': True}
            if tools is not None:
                ct_kwargs['tools'] = tools
                messages = self._prepare_messages_for_template(messages)
            try:
                token_ids = self._tokenizer.apply_chat_template(
                    messages, tokenize=True, **ct_kwargs)
                return len(token_ids)
            except Exception:
                pass
            try:
                full_prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                total = sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "")
                            for m in messages)
                return total // 3
            return len(self._tokenizer.encode(full_prompt, add_special_tokens=False))

        if self._tokenizer is None:
            return self._last_usage_prompt_tokens

        try:
            token_ids = self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)
            return len(token_ids)
        except Exception:
            pass

        is_tiktoken = type(self._tokenizer).__name__ == 'TikTokenTokenizer'
        total = 0
        for msg in messages:
            content = msg.get("content") or ""
            if content:
                if is_tiktoken:
                    total += len(self._tokenizer.encode(content))
                else:
                    total += len(self._tokenizer.encode(content, add_special_tokens=False))
            reasoning = msg.get("reasoning_content") or ""
            if reasoning:
                if is_tiktoken:
                    total += len(self._tokenizer.encode(reasoning))
                else:
                    total += len(self._tokenizer.encode(reasoning, add_special_tokens=False))
            tc_list = msg.get("tool_calls")
            if tc_list:
                for tc in tc_list:
                    fn = tc.get("function", {})
                    tc_text = (fn.get("name", "") + " " + fn.get("arguments", ""))
                    if is_tiktoken:
                        total += len(self._tokenizer.encode(tc_text))
                    else:
                        total += len(self._tokenizer.encode(tc_text, add_special_tokens=False))
        total += len(messages) * 4 + 20
        return total

    # ==================================================================
    # Message preparation for Jinja chat template (hermes_w_py)
    # ==================================================================

    def _prepare_messages_for_template(self, messages):
        """Convert tool_calls arguments from JSON strings to dicts for Jinja template rendering."""
        prepared = []
        for msg in messages:
            if msg.get("tool_calls"):
                msg = msg.copy()
                new_tcs = []
                for tc in msg["tool_calls"]:
                    tc = tc.copy()
                    fn = tc.get("function", {}).copy()
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"])
                        except Exception:
                            fn["arguments"] = {}
                    tc["function"] = fn
                    new_tcs.append(tc)
                msg["tool_calls"] = new_tcs
            prepared.append(msg)
        return prepared

    # ==================================================================
    # Assistant message builder
    # ==================================================================

    def _make_assistant_msg(self, content, tool_calls=None, reasoning=None):
        """Build an assistant message dict, conditionally including reasoning_content."""
        msg = {"role": "assistant", "content": content or ""}
        if self._model_uses_reasoning:
            if reasoning:
                msg["reasoning_content"] = reasoning
            elif not tool_calls:
                msg["reasoning_content"] = "."
                self._log_reasoning_fallback(content, tool_calls, reasoning)
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    def _log_reasoning_fallback(self, content, tool_calls, reasoning):
        try:
            with open("backstop.log", "a", encoding="utf-8") as f:
                f.write(f"[{datetime.datetime.now().isoformat()}] "
                        f"reasoning={reasoning!r}, tool_calls={tool_calls!r}, "
                        f"content={content!r}\n")
        except Exception:
            pass

    # ==================================================================
    # Dispatch
    # ==================================================================

    def _main_checkpoint_hooks(self, data, question, answer, messages,
                               num_llm_calls_available, round_num,
                               format_retry_count, start_time):
        rollout_idx = data.get("rollout_idx")
        item_index = data.get("item_index")
        checkpoint_path = data.get("checkpoint_path")
        checkpoint_enabled = bool(checkpoint_path) and _main_ckpt.checkpoints_enabled()

        def _safe_int(value, default):
            try:
                return int(value)
            except Exception:
                return default

        def _safe_float(value, default):
            try:
                return float(value)
            except Exception:
                return default

        completed_result = None
        if checkpoint_enabled:
            checkpoint = _main_ckpt.load_checkpoint(
                checkpoint_path,
                question=question,
                rollout_idx=rollout_idx,
                item_index=item_index,
            )
            if checkpoint and checkpoint.get("status") == "completed":
                print(f"[main_checkpoint] completed checkpoint found: "
                      f"{checkpoint_path}", flush=True)
                completed_result = checkpoint["result"]
            elif checkpoint and checkpoint.get("status") == "running":
                elapsed = max(
                    0.0,
                    _safe_float(checkpoint.get("elapsed_runtime_seconds"), 0.0),
                )
                start_time = time.time() - elapsed
                messages = checkpoint["messages"]
                num_llm_calls_available = _safe_int(
                    checkpoint.get("num_llm_calls_available"),
                    num_llm_calls_available,
                )
                round_num = _safe_int(checkpoint.get("round_num"), round_num)
                format_retry_count = _safe_int(
                    checkpoint.get("format_retry_count"), format_retry_count)
                print(
                    f"[main_checkpoint] resumed {checkpoint_path} "
                    f"(stage={checkpoint.get('stage')}, round={round_num}, "
                    f"remaining_calls={num_llm_calls_available})",
                    flush=True,
                )

        def save_running(stage, cur_messages, cur_round_num,
                         cur_num_llm_calls_available,
                         cur_format_retry_count, cur_start_time):
            if checkpoint_enabled:
                _main_ckpt.write_running(
                    checkpoint_path,
                    question=question,
                    answer=answer,
                    rollout_idx=rollout_idx,
                    item_index=item_index,
                    messages=cur_messages,
                    round_num=cur_round_num,
                    num_llm_calls_available=cur_num_llm_calls_available,
                    format_retry_count=cur_format_retry_count,
                    start_time=cur_start_time,
                    stage=stage,
                )

        def finish(prediction, termination, stage, cur_messages,
                   cur_round_num, cur_num_llm_calls_available,
                   cur_format_retry_count, cur_start_time):
            result = {
                "question": question,
                "answer": answer,
                "messages": cur_messages,
                "prediction": prediction,
                "termination": termination,
            }
            if checkpoint_enabled:
                _main_ckpt.write_completed(
                    checkpoint_path,
                    result=result,
                    rollout_idx=rollout_idx,
                    item_index=item_index,
                    round_num=cur_round_num,
                    num_llm_calls_available=cur_num_llm_calls_available,
                    format_retry_count=cur_format_retry_count,
                    start_time=cur_start_time,
                    stage=stage,
                )
            return result

        return (
            completed_result,
            messages,
            num_llm_calls_available,
            round_num,
            format_retry_count,
            start_time,
            save_running,
            finish,
        )

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        if self.model_mode == 'api':
            return self._run_api(data, model, **kwargs)
        if TEMPLATE == 'hermes_w_py':
            return self._run_local_hermes_w_py(data, model, **kwargs)
        return self._run_local(data, model, **kwargs)

    # ==================================================================
    # _run_local: original XML tool-call flow (unchanged logic)
    # ==================================================================

    def _run_local(self, data, model, **kwargs):
        self.model = model
        item = data['item']
        question = item.get('question') or item.get('task_question') or ''
        if not question:
            try:
                raw_msg = item['messages'][1]["content"]
                question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg
            except Exception:
                question = ''
        answer = item.get('answer') or item.get('ground_truth') or ''

        start_time = time.time()
        planning_port = data['planning_port']
        system_prompt = render_main_system_prompt(today_date(), include_tools=True)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round_num = 0
        force_answer_prompt = main_force_answer_prompt()
        prompt_token_limit = MAX_CONTEXT_TOKENS - MAX_GENERATION_TOKENS
        format_retry_count = 0
        (completed_result, messages, num_llm_calls_available, round_num,
         format_retry_count, start_time, save_running, finish) = \
            self._main_checkpoint_hooks(
                data, question, answer, messages, num_llm_calls_available,
                round_num, format_retry_count, start_time)
        if completed_result is not None:
            return completed_result
        save_running(
            "start_or_resume", messages, round_num,
            num_llm_calls_available, format_retry_count, start_time)
        while num_llm_calls_available > 0:
            if time.time() - start_time > RUN_TIMEOUT_MINUTES * 60:
                prediction = f'No answer found after {RUN_TIMEOUT_MINUTES} minutes'
                termination = f'No answer found after {RUN_TIMEOUT_MINUTES} minutes'
                return finish(
                    prediction, termination, "timeout", messages, round_num,
                    num_llm_calls_available, format_retry_count, start_time)
            round_num += 1
            num_llm_calls_available -= 1
            if USE_CONTEXT_MANAGEMENT:
                messages = filter_messages(messages, RESTORE_TOOL_OBSERVATION)
            messages_before_round = [msg.copy() for msg in messages]

            content = self.call_server(messages, planning_port)
            print(f'Round {round_num}: {content}')
            if '<tool_response>' in content:
                content = content[:content.find('<tool_response>')]
            messages.append({"role": "assistant", "content": content.strip()})
            parsed_blocks = parse_tool_call_blocks(content)
            if parsed_blocks:
                results = []
                format_error_rollback = False
                try:
                    for blk in parsed_blocks:
                        if blk['kind'] == 'python':
                            try:
                                result = TOOL_MAP['PythonInterpreter'].call(blk['code'])
                            except Exception:
                                result = "[Python Interpreter Error]: Formatting error."
                        elif blk['kind'] == 'json':
                            try:
                                result = self.custom_call_tool(
                                    blk['name'], blk['arguments'],
                                    planning_port=planning_port,
                                    model=self.model,
                                    question=question,
                                )
                            except ToolCallFormatError:
                                raise
                            except Exception:
                                result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                        else:
                            result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                        results.append(result)
                except ToolCallFormatError as e:
                    format_retry_count += 1
                    if format_retry_count < 128:
                        sleep_time = min(1 * (2 ** (format_retry_count - 1)) + random.uniform(0, 1), 30)
                        print(f"[Format Retry] Tool call format error: {e}. "
                              f"Retry {format_retry_count}/128, sleeping {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                        messages = messages_before_round
                        round_num -= 1
                        num_llm_calls_available += 1
                        save_running(
                            "format_retry_rollback", messages, round_num,
                            num_llm_calls_available, format_retry_count,
                            start_time)
                        format_error_rollback = True
                    else:
                        print(f"[Format Retry] All 128 retries exhausted. Using error message.")
                        results.append(str(e))
                        format_retry_count = 0

                if format_error_rollback:
                    continue

                combined = "\n".join(
                    "<tool_response>\n" + result + "\n</tool_response>"
                    for result in results
                )
                messages.append({"role": "user", "content": combined})
                format_retry_count = 0
            else:
                format_retry_count = 0
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break

            token_count = self.count_tokens(messages)
            print(f"round: {round_num}, post-check token count: {token_count}")

            need_force_answer = False
            rolled_back = False

            if token_count > prompt_token_limit:
                print(f"Post-check: token count ({token_count}) exceeds safe limit ({prompt_token_limit}). Rolling back to previous round.")
                messages = messages_before_round
                need_force_answer = True
                rolled_back = True
            elif num_llm_calls_available <= 1:
                print(f"LLM call limit approaching: {num_llm_calls_available} calls remaining. Forcing final answer.")
                need_force_answer = True

            if need_force_answer:
                if messages[-1]["role"] == "user":
                    messages.append({"role": "assistant", "content": "<think>Let me provide my final answer based on all information gathered.</think>"})
                messages.append({"role": "user", "content": force_answer_prompt})

                force_token_count = self.count_tokens(messages)
                adjusted_max_tokens = max(1, MAX_CONTEXT_TOKENS - force_token_count - 1)
                print(f"Forcing answer with adjusted max_tokens: {adjusted_max_tokens}")

                content = self.call_server(messages, planning_port, max_tokens=adjusted_max_tokens)
                messages.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content and '</answer>' in content:
                    prediction = content.split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached (rolled back)' if rolled_back else 'generate an answer as llm call limit reached'
                else:
                    prediction = content
                    termination = 'format error: forced answer (token limit rolled back)' if rolled_back else 'format error: forced answer (llm call limit)'
                stage = 'forced_token_limit' if rolled_back else 'forced_llm_call_limit'
                return finish(
                    prediction, termination, stage, messages, round_num,
                    num_llm_calls_available, format_retry_count, start_time)

            save_running(
                "after_round", messages, round_num,
                num_llm_calls_available, format_retry_count, start_time)

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        return finish(
            prediction, termination, termination, messages, round_num,
            num_llm_calls_available, format_retry_count, start_time)

    # ==================================================================
    # _run_local_hermes_w_py: XML path + multi-<tool_call> parsing.
    # ==================================================================

    def _run_local_hermes_w_py(self, data, model, **kwargs):
        self.model = model
        item = data['item']
        question = item.get('question') or item.get('task_question') or ''
        if not question:
            try:
                raw_msg = item['messages'][1]["content"]
                question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg
            except Exception:
                question = ''
        answer = item.get('answer') or item.get('ground_truth') or ''

        start_time = time.time()
        planning_port = data['planning_port']
        system_prompt = render_main_system_prompt(today_date(), include_tools=True)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round_num = 0
        force_answer_prompt = main_force_answer_prompt()
        prompt_token_limit = MAX_CONTEXT_TOKENS - MAX_GENERATION_TOKENS
        format_retry_count = 0
        (completed_result, messages, num_llm_calls_available, round_num,
         format_retry_count, start_time, save_running, finish) = \
            self._main_checkpoint_hooks(
                data, question, answer, messages, num_llm_calls_available,
                round_num, format_retry_count, start_time)
        if completed_result is not None:
            return completed_result
        save_running(
            "start_or_resume", messages, round_num,
            num_llm_calls_available, format_retry_count, start_time)
        while num_llm_calls_available > 0:
            if time.time() - start_time > RUN_TIMEOUT_MINUTES * 60:
                prediction = f'No answer found after {RUN_TIMEOUT_MINUTES} minutes'
                termination = f'No answer found after {RUN_TIMEOUT_MINUTES} minutes'
                return finish(
                    prediction, termination, "timeout", messages, round_num,
                    num_llm_calls_available, format_retry_count, start_time)
            round_num += 1
            num_llm_calls_available -= 1
            if USE_CONTEXT_MANAGEMENT:
                messages = filter_messages(messages, RESTORE_TOOL_OBSERVATION)
            messages_before_round = [msg.copy() for msg in messages]

            content = self.call_server(messages, planning_port)
            print(f'Round {round_num}: {content}')
            if '<tool_response>' in content:
                content = content[:content.find('<tool_response>')]
            messages.append({"role": "assistant", "content": content.strip()})

            parsed_blocks = parse_tool_call_blocks(content)
            if parsed_blocks:
                results = []
                format_error_rollback = False
                try:
                    for blk in parsed_blocks:
                        if blk['kind'] == 'python':
                            try:
                                one = TOOL_MAP['PythonInterpreter'].call(blk['code'])
                            except Exception:
                                one = "[Python Interpreter Error]: Formatting error."
                        elif blk['kind'] == 'json':
                            try:
                                one = self.custom_call_tool(blk['name'], blk['arguments'],
                                                            planning_port=planning_port, model=self.model,
                                                            question=question)
                            except ToolCallFormatError:
                                raise
                            except Exception:
                                one = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                        else:  # 'bad_json'
                            one = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                        results.append(one)
                except ToolCallFormatError as e:
                    format_retry_count += 1
                    if format_retry_count < 128:
                        sleep_time = min(1 * (2 ** (format_retry_count - 1)) + random.uniform(0, 1), 30)
                        print(f"[Format Retry] Tool call format error: {e}. "
                              f"Retry {format_retry_count}/128, sleeping {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                        messages = messages_before_round
                        round_num -= 1
                        num_llm_calls_available += 1
                        save_running(
                            "format_retry_rollback", messages, round_num,
                            num_llm_calls_available, format_retry_count,
                            start_time)
                        format_error_rollback = True
                    else:
                        print(f"[Format Retry] All 128 retries exhausted. Using error message.")
                        results.append(str(e))
                        format_retry_count = 0

                if format_error_rollback:
                    continue

                combined = "\n".join(
                    "<tool_response>\n" + r + "\n</tool_response>" for r in results)
                messages.append({"role": "user", "content": combined})
                format_retry_count = 0
            else:
                format_retry_count = 0
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break

            token_count = self.count_tokens(messages)
            print(f"round: {round_num}, post-check token count: {token_count}")

            need_force_answer = False
            rolled_back = False

            if token_count > prompt_token_limit:
                print(f"Post-check: token count ({token_count}) exceeds safe limit ({prompt_token_limit}). Rolling back to previous round.")
                messages = messages_before_round
                need_force_answer = True
                rolled_back = True
            elif num_llm_calls_available <= 1:
                print(f"LLM call limit approaching: {num_llm_calls_available} calls remaining. Forcing final answer.")
                need_force_answer = True

            if need_force_answer:
                if messages[-1]["role"] == "user":
                    messages.append({"role": "assistant", "content": "<think>Let me provide my final answer based on all information gathered.</think>"})
                messages.append({"role": "user", "content": force_answer_prompt})

                force_token_count = self.count_tokens(messages)
                adjusted_max_tokens = max(1, MAX_CONTEXT_TOKENS - force_token_count - 1)
                print(f"Forcing answer with adjusted max_tokens: {adjusted_max_tokens}")

                content = self.call_server(messages, planning_port, max_tokens=adjusted_max_tokens)
                messages.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content and '</answer>' in content:
                    prediction = content.split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached (rolled back)' if rolled_back else 'generate an answer as llm call limit reached'
                else:
                    prediction = content
                    termination = 'format error: forced answer (token limit rolled back)' if rolled_back else 'format error: forced answer (llm call limit)'
                stage = 'forced_token_limit' if rolled_back else 'forced_llm_call_limit'
                return finish(
                    prediction, termination, stage, messages, round_num,
                    num_llm_calls_available, format_retry_count, start_time)

            save_running(
                "after_round", messages, round_num,
                num_llm_calls_available, format_retry_count, start_time)

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        return finish(
            prediction, termination, termination, messages, round_num,
            num_llm_calls_available, format_retry_count, start_time)

    # ==================================================================
    # _run_api: OpenAI function-calling flow
    # ==================================================================

    def _run_api(self, data, model, **kwargs):
        self.model = model
        item = data['item']
        question = item.get('question') or item.get('task_question') or ''
        if not question:
            try:
                raw_msg = item['messages'][1]["content"]
                question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg
            except Exception:
                question = ''
        answer = item.get('answer') or item.get('ground_truth') or ''

        start_time = time.time()
        system_prompt = render_main_system_prompt(today_date(), include_tools=False)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round_num = 0
        force_answer_prompt = main_force_answer_prompt()
        prompt_token_limit = MAX_CONTEXT_TOKENS - MAX_GENERATION_TOKENS
        use_api_token_counter = (TOKEN_COUNTER == 'api')
        prev_messages_snapshot = None
        format_retry_count = 0
        (completed_result, messages, num_llm_calls_available, round_num,
         format_retry_count, start_time, save_running, finish) = \
            self._main_checkpoint_hooks(
                data, question, answer, messages, num_llm_calls_available,
                round_num, format_retry_count, start_time)
        if completed_result is not None:
            return completed_result
        save_running(
            "start_or_resume", messages, round_num,
            num_llm_calls_available, format_retry_count, start_time)

        while num_llm_calls_available > 0:
            if time.time() - start_time > RUN_TIMEOUT_MINUTES * 60:
                prediction = f'No answer found after {RUN_TIMEOUT_MINUTES} minutes'
                termination = f'No answer found after {RUN_TIMEOUT_MINUTES} minutes'
                return finish(
                    prediction, termination, "timeout", messages, round_num,
                    num_llm_calls_available, format_retry_count, start_time)

            round_num += 1
            num_llm_calls_available -= 1
            if USE_CONTEXT_MANAGEMENT:
                messages = filter_messages(messages, RESTORE_TOOL_OBSERVATION)
            messages_before_round = deepcopy(messages)

            content, tool_calls, usage, reasoning, finish_reason = self.call_api(messages, use_tools=True)
            print(f'Round {round_num}: {(content or "")[:200]}{"..." if content and len(content) > 200 else ""}'
                  f'{" [+tool_calls]" if tool_calls else ""}')

            if finish_reason == "exhausted":
                print(f"[exhausted] Rolling back round {round_num} and forcing answer from existing context.")
                messages = messages_before_round
                if messages[-1]["role"] in ("user", "tool"):
                    messages.append(self._make_assistant_msg(
                        "Let me provide my final answer based on all information gathered.",
                        reasoning="LLM retries exhausted, consolidating information to provide final answer.",
                    ))
                messages.append({"role": "user", "content": force_answer_prompt})
                fc, _, _, fr, _ = self.call_api(messages, use_tools=False)
                messages.append(self._make_assistant_msg((fc or "").strip(), reasoning=fr))
                final_text = fc or fr or ""
                if '<answer>' in final_text and '</answer>' in final_text:
                    prediction = final_text.split('<answer>')[1].split('</answer>')[0]
                    termination = 'answer (forced after retries exhausted)'
                else:
                    prediction = final_text
                    termination = 'forced answer, no answer tag (retries exhausted)'
                return finish(
                    prediction, termination, "forced_after_retries_exhausted",
                    messages, round_num, num_llm_calls_available,
                    format_retry_count, start_time)

            if use_api_token_counter:
                prompt_tokens = usage.get("prompt_tokens", 0)
                print(f"round: {round_num}, api-token-counter (pt={prompt_tokens}, finish={finish_reason}, limit={prompt_token_limit})")
                if prompt_tokens >= prompt_token_limit or finish_reason == "length":
                    print(f"[api-token-counter] Force answer triggered: "
                          f"pt={prompt_tokens}>={prompt_token_limit}={prompt_tokens >= prompt_token_limit}, "
                          f"finish_reason=={finish_reason}. Rolling back to messages_before_round.")
                    messages = messages_before_round
                    force_tc_id = f"force_summary_{round_num}"
                    messages.append(self._make_assistant_msg(
                        "I have reached the context limit and need to immediately think, summarize, and return the final answer.",
                        tool_calls=[{
                            "id": force_tc_id,
                            "type": "function",
                            "function": {"name": "I_should_summarize_now", "arguments": "{}"}
                        }],
                        reasoning=force_answer_prompt,
                    ))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": force_tc_id,
                        "content": force_answer_prompt
                    })
                    content, _, _, reasoning, _ = self.call_api(messages, max_tokens=MAX_GENERATION_TOKENS, use_tools=False)
                    messages.append(self._make_assistant_msg((content or "").strip(), reasoning=reasoning))
                    final_text = content or reasoning or ""
                    if '<answer>' in final_text and '</answer>' in final_text:
                        prediction = final_text.split('<answer>')[1].split('</answer>')[0]
                        termination = 'generate an answer as token limit reached (api token counter)'
                    else:
                        prediction = final_text
                        termination = 'forced answer, no answer tag (api token counter)'
                    return finish(
                        prediction, termination, "forced_token_limit",
                        messages, round_num, num_llm_calls_available,
                        format_retry_count, start_time)

            elif self._tokenizer is None:
                prompt_tokens = usage.get("prompt_tokens", 0)
                if prompt_tokens > MAX_CONTEXT_TOKENS:
                    print(f"[No-tokenizer] prompt_tokens ({prompt_tokens}) > MAX_CONTEXT_TOKENS ({MAX_CONTEXT_TOKENS}). "
                          f"This request is invalid. Rolling back to previous round's input.")
                    messages = prev_messages_snapshot if prev_messages_snapshot is not None else messages_before_round
                    if messages[-1]["role"] in ("user", "tool"):
                        messages.append(self._make_assistant_msg(
                            "Let me provide my final answer based on all information gathered.",
                            reasoning="Context limit exceeded, consolidating information to provide final answer.",
                        ))
                    messages.append({"role": "user", "content": force_answer_prompt})
                    content, _, usage, reasoning, _ = self.call_api(messages, use_tools=False)
                    messages.append(self._make_assistant_msg((content or "").strip(), reasoning=reasoning))
                    if content and '<answer>' in content and '</answer>' in content:
                        prediction = content.split('<answer>')[1].split('</answer>')[0]
                        termination = 'generate an answer as token limit reached (rolled back)'
                    else:
                        prediction = content or ""
                        termination = 'format error: forced answer (token limit rolled back)'
                    return finish(
                        prediction, termination, "forced_token_limit",
                        messages, round_num, num_llm_calls_available,
                        format_retry_count, start_time)

            messages.append(self._make_assistant_msg(content, tool_calls=tool_calls, reasoning=reasoning))

            if not tool_calls:
                final_text = content or reasoning or ""
                if '<answer>' in final_text and '</answer>' in final_text:
                    prediction = final_text.split('<answer>')[1].split('</answer>')[0]
                    termination = 'answer'
                elif content:
                    prediction = content
                    termination = 'answer (no tool calls, no answer tag)'
                elif reasoning:
                    prediction = reasoning
                    termination = 'answer (no tool calls, answer in reasoning only)'
                else:
                    prediction = ""
                    termination = 'answer (no tool calls, empty response)'
                return finish(
                    prediction, termination, "final_no_tool_calls",
                    messages, round_num, num_llm_calls_available,
                    format_retry_count, start_time)

            format_error_in_round = False
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                try:
                    tool_args = json.loads(tc["function"]["arguments"])
                except Exception:
                    tool_args = {}

                if tool_name == "PythonInterpreter":
                    code = tool_args.get("code", "")
                    try:
                        result = TOOL_MAP['PythonInterpreter'].call(code)
                    except Exception:
                        result = "[Python Interpreter Error]: Execution failed."
                else:
                    try:
                        result = self.custom_call_tool(tool_name, tool_args,
                                                       planning_port=None, model=self.model,
                                                       question=question)
                    except ToolCallFormatError as e:
                        format_retry_count += 1
                        if format_retry_count < 128:
                            sleep_time = min(1 * (2 ** (format_retry_count - 1)) + random.uniform(0, 1), 30)
                            print(f"[Format Retry] Tool call format error for '{tool_name}': {e}. "
                                  f"Retry {format_retry_count}/128, sleeping {sleep_time:.2f}s...")
                            time.sleep(sleep_time)
                            messages = messages_before_round
                            round_num -= 1
                            num_llm_calls_available += 1
                            save_running(
                                "format_retry_rollback", messages, round_num,
                                num_llm_calls_available, format_retry_count,
                                start_time)
                            format_error_in_round = True
                            break
                        else:
                            print(f"[Format Retry] All 128 retries exhausted for '{tool_name}'. Using error message.")
                            result = str(e)
                            format_retry_count = 0
                    except Exception as e:
                        result = f"[Tool Error] {tool_name}: {type(e).__name__}: {e!r}. Arguments received: {json.dumps(tool_args, ensure_ascii=False, default=str)[:500]}"

                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

            if format_error_in_round:
                continue

            format_retry_count = 0

            if not use_api_token_counter:
                need_force_answer = False
                rolled_back = False

                if self._tokenizer is not None:
                    token_count = self.count_tokens(messages)
                    print(f"round: {round_num}, post-check token count: {token_count} (usage.pt={usage.get('prompt_tokens','?')})")

                    if token_count > prompt_token_limit:
                        print(f"Post-check: token count ({token_count}) exceeds limit ({prompt_token_limit}). Rolling back.")
                        messages = messages_before_round
                        need_force_answer = True
                        rolled_back = True
                else:
                    print(f"round: {round_num}, no-tokenizer mode (usage.pt={usage.get('prompt_tokens','?')})")

                if not need_force_answer and num_llm_calls_available <= 1:
                    print(f"LLM call limit approaching: {num_llm_calls_available} calls remaining. Forcing final answer.")
                    need_force_answer = True

                if need_force_answer:
                    if messages[-1]["role"] in ("user", "tool"):
                        messages.append(self._make_assistant_msg(
                            "Let me provide my final answer based on all information gathered.",
                            reasoning="Context limit approaching, consolidating information to provide final answer.",
                        ))
                    messages.append({"role": "user", "content": force_answer_prompt})

                    if self._tokenizer is not None:
                        force_token_count = self.count_tokens(messages)
                        adjusted_max_tokens = max(1, MAX_CONTEXT_TOKENS - force_token_count - 1)
                    else:
                        adjusted_max_tokens = None
                    print(f"Forcing answer with adjusted max_tokens: {adjusted_max_tokens}")

                    content, _, usage, reasoning, _ = self.call_api(messages, max_tokens=adjusted_max_tokens, use_tools=False)
                    messages.append(self._make_assistant_msg((content or "").strip(), reasoning=reasoning))
                    if content and '<answer>' in content and '</answer>' in content:
                        prediction = content.split('<answer>')[1].split('</answer>')[0]
                        termination = 'generate an answer as token limit reached (rolled back)' if rolled_back else 'generate an answer as llm call limit reached'
                    else:
                        prediction = content or ""
                        termination = 'format error: forced answer (token limit rolled back)' if rolled_back else 'format error: forced answer (llm call limit)'
                    stage = 'forced_token_limit' if rolled_back else 'forced_llm_call_limit'
                    return finish(
                        prediction, termination, stage, messages, round_num,
                        num_llm_calls_available, format_retry_count,
                        start_time)

                if self._tokenizer is None:
                    prev_messages_snapshot = messages_before_round

            save_running(
                "after_round", messages, round_num,
                num_llm_calls_available, format_retry_count, start_time)

        last_content = messages[-1].get("content", "") if messages else ""
        if '<answer>' in last_content and '</answer>' in last_content:
            prediction = last_content.split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        return finish(
            prediction, termination, termination, messages, round_num,
            num_llm_calls_available, format_retry_count, start_time)

    # ==================================================================
    # Tool execution helper
    # ==================================================================

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in TOOL_MAP:
            tool_args["params"] = tool_args
            raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
            return raw_result
        else:
            return f"Error: Tool {tool_name} not found"
