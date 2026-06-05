import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool

PYTHON_TIMEOUT = int(os.getenv("PYTHON_INTERPRETER_TIMEOUT", "30"))


@register_tool('PythonInterpreter', allow_overwrite=True)
class PythonInterpreter(BaseTool):
    name = "PythonInterpreter"
    description = "Execute Python code in a sandboxed environment."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            }
        },
        "required": ["code"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            code = params
        else:
            code = params.get("code", "")

        if not code.strip():
            return "stderr:\n[PythonInterpreter Error]: Empty code."

        try:
            with tempfile.TemporaryDirectory(prefix="python_interpreter-") as tmp_dir:
                file_path = Path(tmp_dir) / "main.py"
                file_path.write_text(code, encoding="utf-8")

                result = subprocess.run(
                    [sys.executable, str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=PYTHON_TIMEOUT,
                    cwd=tmp_dir,
                )

                stdout = result.stdout.rstrip("\n")
                stderr = result.stderr.rstrip("\n")

                parts = []
                if stdout:
                    parts.append(f"stdout:\n{stdout}")
                if stderr:
                    parts.append(f"stderr:\n{stderr}")

                if not parts:
                    return "stdout:\n"

                return "\n\n".join(parts)

        except subprocess.TimeoutExpired:
            return f"stderr:\nExecution timed out after {PYTHON_TIMEOUT} seconds."
        except Exception as e:
            return f"stderr:\n{str(e)}"
