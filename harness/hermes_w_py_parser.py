"""Client-side parsing for TEMPLATE=hermes_w_py.

The model emits, per assistant turn, zero or more
blocks of:

    <tool_call>
    {"name": "<tool>", "arguments": <args-json>}
    </tool_call>

with ONE exception: for the PythonInterpreter tool, the model wraps the
Python source inside a `<code>...</code>` tag placed INSIDE the
`<tool_call>` but OUTSIDE the JSON body:

    <tool_call>
    {"name": "PythonInterpreter", "arguments": {}}
    <code>
    import math
    print(math.e)
    </code>
    </tool_call>

This format is the model's native output in both TEMPLATE=qwen3 and
TEMPLATE=hermes_w_py. Both local paths use `parse_tool_call_blocks` below
to execute every tool block in one assistant turn and keep the raw stream
verbatim for replay.

Kept in its own module so tests can exercise this logic without pulling
in the full react_agent import chain (tokenizers, vLLM clients, etc.).
"""

import re

import json5


TOOL_CALL_PATTERN = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)


def parse_tool_call_blocks(content):
    """Extract and classify every <tool_call>...</tool_call> block in content.

    Returns a list of dicts; each dict has a 'kind' key:
      {'kind': 'python', 'code': str}
          PythonInterpreter path: <code>...</code> found; `code` is the
          stripped body between <code> and </code>.
      {'kind': 'json', 'name': str, 'arguments': dict}
          Non-Python path: JSON body parsed successfully.
      {'kind': 'bad_json', 'raw': str, 'error': Exception}
          JSON body failed to parse. Caller decides the error policy
          (e.g., inline an 'invalid JSON' tool_response message).

    The classifier matches the original TEMPLATE=qwen3 single-tool
    dispatch EXACTLY: "python" (case-insensitive) appears in the block
    AND the block contains BOTH <code> and </code>. Everything else goes
    through JSON. This preserves bug-for-bug compatibility: a rogue
    tool_call with "python" in its name string but no <code> wrapper
    falls through to JSON parsing, same as before.
    """
    out = []
    for block in TOOL_CALL_PATTERN.findall(content):
        if "python" in block.lower() and '<code>' in block and '</code>' in block:
            code = block.split('<code>')[1].split('</code>')[0].strip()
            out.append({'kind': 'python', 'code': code})
        else:
            try:
                parsed = json5.loads(block)
                if not isinstance(parsed, dict):
                    out.append({'kind': 'bad_json', 'raw': block,
                                'error': ValueError(
                                    f"tool_call body must be a JSON object, "
                                    f"got {type(parsed).__name__}")})
                    continue
                out.append({
                    'kind': 'json',
                    'name': parsed.get('name', ''),
                    'arguments': parsed.get('arguments', {}),
                })
            except Exception as e:
                out.append({'kind': 'bad_json', 'raw': block, 'error': e})
    return out
