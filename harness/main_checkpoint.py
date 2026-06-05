"""Per-main-agent checkpoint helpers.

Each main rollout writes one JSON checkpoint with atomic replace. A restarted
run can resume from a safe turn boundary without sharing a hot JSONL append
lock across workers.
"""

import datetime
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


CHECKPOINT_VERSION = 1

_XML_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_XML_TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)


def _truthy(value: Any) -> bool:
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def checkpoints_enabled() -> bool:
    return not _truthy(os.environ.get("MAIN_AGENT_CHECKPOINT_DISABLE", "0"))


def checkpoint_fsync_enabled() -> bool:
    return _truthy(os.environ.get("MAIN_AGENT_CHECKPOINT_FSYNC", "0"))


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except Exception:
        return
    try:
        os.fsync(fd)
    except Exception:
        pass
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


def question_hash(question: str) -> str:
    return hashlib.sha256((question or "").encode("utf-8")).hexdigest()[:16]


def safe_slug(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "")).strip("_")
    return (slug or "question")[:max_len]


def checkpoint_root(model_dir: str) -> str:
    base = os.environ.get("MAIN_AGENT_CHECKPOINT_DIR")
    if not base:
        base = os.path.join(model_dir, "_checkpoints", "main_agent")
    return base


def build_checkpoint_path(model_dir: str, rollout_idx: int, question: str,
                          worker_split: int = 1, total_splits: int = 1,
                          item_index: Optional[int] = None) -> str:
    base = checkpoint_root(model_dir)
    split = f"split_{worker_split}_of_{total_splits}"
    item_prefix = f"item_{item_index:08d}_" if item_index is not None else ""
    name = f"{item_prefix}{question_hash(question)}_{safe_slug(question, 64)}.json"
    return os.path.join(base, split, f"rollout_{rollout_idx}", name)


def _tool_call_ids(tool_calls: Any) -> Optional[list]:
    if not isinstance(tool_calls, list):
        return None
    ids = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            return None
        call_id = tc.get("id")
        if not call_id:
            return None
        ids.append(call_id)
    if len(set(ids)) != len(ids):
        return None
    return ids


def _xml_tool_call_count(content: str) -> int:
    return len(_XML_TOOL_CALL_RE.findall(content or ""))


def _xml_tool_response_count(content: str) -> int:
    return len(_XML_TOOL_RESPONSE_RE.findall(content or ""))


def messages_safe_for_resume(messages: Any) -> bool:
    """Return True when messages can be sent as the next LLM input.

    Supports both OpenAI-style assistant tool_calls/tool messages and local
    XML-style assistant <tool_call> followed by user <tool_response> blocks.
    """
    if not isinstance(messages, list):
        return False

    pending_openai = set()
    pending_xml_count = 0

    for msg in messages:
        if not isinstance(msg, dict):
            return False
        role = msg.get("role")
        content = msg.get("content") or ""

        if pending_openai:
            if role != "tool":
                return False
            call_id = msg.get("tool_call_id")
            if call_id not in pending_openai:
                return False
            pending_openai.remove(call_id)
            continue

        if pending_xml_count:
            if role != "user":
                return False
            if _xml_tool_response_count(content) != pending_xml_count:
                return False
            pending_xml_count = 0
            continue

        if role == "tool":
            return False
        if role == "user" and _xml_tool_response_count(content):
            return False

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                return False
            if tool_calls:
                tool_call_ids = _tool_call_ids(tool_calls)
                if tool_call_ids is None:
                    return False
                pending_openai = set(tool_call_ids)

            xml_count = _xml_tool_call_count(content)
            if xml_count:
                pending_xml_count = xml_count

    return not pending_openai and not pending_xml_count


def _checkpoint_payload(status: str, question: str, answer: str,
                        rollout_idx: Optional[int], item_index: Optional[int],
                        messages: Any, round_num: int,
                        num_llm_calls_available: int,
                        format_retry_count: int,
                        elapsed_runtime_seconds: float, stage: str,
                        result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    now = time.time()
    payload = {
        "version": CHECKPOINT_VERSION,
        "status": status,
        "stage": stage,
        "updated_at": now,
        "updated_at_iso": datetime.datetime.fromtimestamp(now).isoformat(),
        "question": question,
        "question_hash": question_hash(question),
        "answer": answer,
        "rollout_idx": rollout_idx,
        "item_index": item_index,
        "messages": messages,
        "round_num": round_num,
        "num_llm_calls_available": num_llm_calls_available,
        "format_retry_count": format_retry_count,
        "elapsed_runtime_seconds": elapsed_runtime_seconds,
    }
    if result is not None:
        payload["result"] = result
    return payload


def write_checkpoint(path: Optional[str], payload: Dict[str, Any]) -> bool:
    if not path or not checkpoints_enabled():
        return False
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=f".{os.getpid()}.{threading.get_ident()}.tmp",
            dir=str(target.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
                f.write("\n")
                f.flush()
                if checkpoint_fsync_enabled():
                    os.fsync(f.fileno())
            os.replace(tmp_path, target)
            if checkpoint_fsync_enabled():
                _fsync_dir(target.parent)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return True
    except Exception as e:
        print(f"[main_checkpoint] WARNING: failed to write {path}: "
              f"{type(e).__name__}: {e}", flush=True)
        return False


def write_running(path: Optional[str], question: str, answer: str,
                  rollout_idx: Optional[int], item_index: Optional[int],
                  messages: Any, round_num: int,
                  num_llm_calls_available: int, format_retry_count: int,
                  start_time: float, stage: str) -> bool:
    if not messages_safe_for_resume(messages):
        print(f"[main_checkpoint] WARNING: refused unsafe running checkpoint "
              f"at stage={stage}", flush=True)
        return False
    payload = _checkpoint_payload(
        status="running",
        question=question,
        answer=answer,
        rollout_idx=rollout_idx,
        item_index=item_index,
        messages=messages,
        round_num=round_num,
        num_llm_calls_available=num_llm_calls_available,
        format_retry_count=format_retry_count,
        elapsed_runtime_seconds=max(0.0, time.time() - start_time),
        stage=stage,
    )
    return write_checkpoint(path, payload)


def write_completed(path: Optional[str], result: Dict[str, Any],
                    rollout_idx: Optional[int], item_index: Optional[int],
                    round_num: int, num_llm_calls_available: int,
                    format_retry_count: int, start_time: float,
                    stage: str) -> bool:
    payload = _checkpoint_payload(
        status="completed",
        question=result.get("question", ""),
        answer=result.get("answer", ""),
        rollout_idx=rollout_idx,
        item_index=item_index,
        messages=result.get("messages", []),
        round_num=round_num,
        num_llm_calls_available=num_llm_calls_available,
        format_retry_count=format_retry_count,
        elapsed_runtime_seconds=max(0.0, time.time() - start_time),
        stage=stage,
        result=result,
    )
    return write_checkpoint(path, payload)


def load_checkpoint(path: Optional[str], question: str = "",
                    rollout_idx: Optional[int] = None,
                    item_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
    if not path or not checkpoints_enabled() or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[main_checkpoint] WARNING: failed to read {path}: "
              f"{type(e).__name__}: {e}", flush=True)
        return None

    if not isinstance(data, dict):
        print(f"[main_checkpoint] WARNING: ignoring non-object checkpoint "
              f"{path}", flush=True)
        return None
    if data.get("version") != CHECKPOINT_VERSION:
        print(f"[main_checkpoint] WARNING: ignoring unsupported checkpoint "
              f"version in {path}", flush=True)
        return None
    if question and data.get("question") != question:
        print(f"[main_checkpoint] WARNING: ignoring checkpoint question "
              f"mismatch: {path}", flush=True)
        return None
    if rollout_idx is not None and data.get("rollout_idx") not in (None, rollout_idx):
        print(f"[main_checkpoint] WARNING: ignoring checkpoint rollout "
              f"mismatch: {path}", flush=True)
        return None
    if item_index is not None and data.get("item_index") not in (None, item_index):
        print(f"[main_checkpoint] WARNING: ignoring checkpoint item_index "
              f"mismatch: {path}", flush=True)
        return None

    status = data.get("status")
    if status == "completed":
        result = data.get("result")
        if isinstance(result, dict):
            return data
        print(f"[main_checkpoint] WARNING: completed checkpoint has no "
              f"result: {path}", flush=True)
        return None
    if status == "running":
        if messages_safe_for_resume(data.get("messages")):
            return data
        print(f"[main_checkpoint] WARNING: ignoring unsafe running "
              f"checkpoint {path}", flush=True)
        return None

    print(f"[main_checkpoint] WARNING: ignoring checkpoint with "
          f"status={status!r}: {path}", flush=True)
    return None


def remove_checkpoint(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
        if checkpoint_fsync_enabled():
            _fsync_dir(Path(path).parent)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[main_checkpoint] WARNING: failed to remove {path}: "
              f"{type(e).__name__}: {e}", flush=True)
