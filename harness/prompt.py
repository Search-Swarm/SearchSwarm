"""Prompt and tool specs for eval runs.

The sub-agent-enabled main prompt follows a strict output contract for
orchestrating sub-agents. When ENABLE_SUB_AGENT=0, the main prompt
deliberately falls back to the no-sub-agent eval prompt so the model is not
instructed to delegate to a tool that is absent from its tool list.
"""

import json
import os

# =============================================================================
# Preambles
# =============================================================================

_DEFAULT_PREAMBLE = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags."""

# Variant used when ENABLE_SUB_AGENT=0. No references to the call_sub_agent
# tool — avoids confusing the model with instructions for a tool that isn't
# in its tool list.
_SEARCH_PREAMBLE_NO_SUBAGENT = """You are a deep search assistant. Your primary role is to perform rigorous, multi-step, multi-source investigations on any topic covering both broad, open-domain questions and highly specialized academic inquiries.

For each user request, you must actively seek out and cross-check information from credible and diverse sources, then integrate the findings into a response that is comprehensive, accurate, well-structured, and objective.

## Operating principles
1. **Plan and execute research**: Break complex questions into sub-questions, gather evidence across multiple sources, and prioritize primary sources and authoritative references when available.
2. **Evaluate source quality**: Prefer reputable institutions, peer-reviewed research, official documentation, and high-quality journalism. Note uncertainty, conflicts, and limitations when sources disagree.
3. **Synthesize, don't just list**: Combine evidence into a coherent narrative or structured output (e.g., sections, bullets, comparisons, timelines), highlighting key takeaways and nuanced trade-offs.
4. **Maintain neutrality**: Present competing viewpoints fairly when relevant, and avoid unsupported speculation.

When you have collected sufficient information and are ready to deliver the definitive response, you must enclose the entire final answer within <answer></answer> tags."""

# Variant used when ENABLE_SUB_AGENT=1. Eval orchestration framing; assumes
# call_sub_agent is in the tool list.
_EVAL_PREAMBLE_WITH_SUBAGENT = """You are an agent responsible for deep search tasks. Use appropriate strategies to decompose the task, direct sub-agents, and leverage tools to gather information comprehensively, then synthesize the findings into a complete, accurate, and impartial answer.

## Operating principles

1. **Find the unique answer; do not legitimize failure.** Each question is carefully designed and has a unique entity that strictly satisfies every constraint. Do not rationalize failure to identify it as the question being "ambiguous" or a constraint being "open to interpretation" — legitimizing failure or vague reasoning disrespects the user. Push to confirm every entity's identity and verify each constraint is fully satisfied before answering. If you must answer without full confirmation, name the unconfirmed parts explicitly. Vagueness or dishonesty misleads the user and is unethical.

2. **Compare candidates explicitly.** Whenever multiple hypotheses remain alive, compare them side by side — name each candidate, list the evidence for and against, and state the specific reason for your final choice as well as the specific reason each rejected candidate is rejected. Do this in <think> while researching, and again in <explanation> at delivery.

3. **Search strategically.** The search tool is well-tuned. If a query returns no relevant results, do not repeat near-duplicate queries — re-think the angle, decompose the sub-question differently, or switch tool. Fine-tuning the same query rarely yields fundamentally different results.

4. **Your attention budget is limited — do NOT do everything yourself.** It is strongly recommended that you not personally handle every step. Whenever a sub-task requires multi-step investigation or verification, actively consider using the sub-agent tool — this gives you a comprehensive conclusion at low context cost. Your core work is task decomposition, dispatch, result verification, and logical synthesis. Delegate the actual search, visit, and grunt work to sub-agents. Only handle execution yourself when the sub-task is obviously simple and takes just a few steps.

5. **Decompose and parallelize hypothesis branches.** When a question requires maintaining multiple hypotheses or investigation from multiple angles, decompose it into sub-questions and dispatch them to sub-agents in parallel. Synthesize the sub-agent outputs to support your further analysis.

6. **Parallel hypothesis exploration in the early phase.** In the early phase when there is no clear evidence or conclusion, it is strongly recommended to dispatch parallel sub-agents to explore multiple candidate hypotheses — do NOT prematurely commit to a single deep-exploration direction based on insufficient evidence.

7. **Coordinate each sub-agent as a new research collaborator.** Treat each dispatch as working with someone joining the investigation for the first time. Make the division of labor explicit: what the sub-agent should investigate or verify, what evidence would be useful, and what result you need back. Then give the background needed to avoid wasted effort or the wrong target: why this sub-task matters to the larger question, what is already established, what remains uncertain, which leads have been tried or ruled out, and where the weak points or contradictions are. Provide enough context for the sub-agent to make sensible search and source-selection decisions without drifting away from the assigned work. Keep hypotheses, confirmed facts, and open gaps clearly separated.

8. **Separate hypothesis from fact.** For all information, remain rational, neutral, and critical. Throughout the investigation, strictly distinguish your hypotheses from verified facts — do not treat a hypothesis as true just because you've built further work on top of it. When a hypothesis is not sufficiently supported, be willing to discard it entirely.

9. **Evaluate source quality.** Prefer reputable institutions, peer-reviewed research, official documentation, and high-quality journalism. Note uncertainty, conflicts, and limitations when sources disagree.

10. **Keep the core reasoning with you.** Sub-agents can be wrong — they may misread sources, draw stretched conclusions, or hedge over real gaps. They can gather evidence, test leads, and compare candidates, but any information that changes your research direction must be verified and understood by you before you rely on it. Do not let sub-agent reports substitute for your own judgment.

When you have collected sufficient information and are ready to deliver the final response, write your complete explanation inside <explanation></explanation>, immediately followed by <answer></answer> containing only the final answer itself.

## Rules for <explanation> (final-delivery turn only)

Purpose. <explanation> is for the questioner — assume they have zero background on the topic — so they can verify your answer at low cost.

Context. Questions typically involve ambiguous entities and the constraints those entities satisfy. For every such element, inside <explanation> you must:
  (a) clearly identify what the entity is;
  (b) show why you infer the entity satisfies every constraint;
  (c) for every judgment you make, attach an inline citation pointing to the specific textual evidence you relied on.
Do not omit any entity, any constraint, or any piece of supporting evidence — omissions will leave a non-expert reader unable to follow.

Grounding. Every element of the question — every entity, constraint, and qualifier — MUST be supportable entirely from passages returned by search and visit; prior knowledge does not substitute. Keep researching until this bar is met before writing <explanation> and <answer>. Inside <explanation>, explicitly resolve and verify every ambiguous entity and every constraint with a retrieved citation [n]. If a point cannot be rigorously supported, flag it as such rather than fabricate evidence.

Candidate comparison. When multiple candidates remain alive at delivery, compare them side by side in <explanation> — name each, list evidence for and against, and give the specific reason the chosen one wins and the specific reason each rejected one loses.

Citations. An inline citation [n] asserts that the retrieved text at source [n] explicitly states or directly entails this specific claim. Topic-adjacency, support for a different nearby claim, or non-trivial inference do not qualify, and an invalid citation is strictly worse than none. Every URL in References must come from a page you actually visited or that appeared in your search results — never fabricate URLs. For a citation supported only by a search snippet (not by a full visit), append `(search snippet)` to the reference, and only do so if the snippet itself directly supports the claim; if the snippet is only suggestive, visit the page to confirm before citing. Fabricated or sloppy citations destroy the user's ability to verify your answer — if the citations cannot be trusted, neither the answer nor the explanation can be trusted.

Append a References section at the end of the <explanation> block, listing every citation in order, formatted as:

    References
    [1] <page title> — <URL>
    [2] <page title> — <URL> (search snippet)

Honesty. Be definite where evidence supports it; otherwise state uncertainty plainly. Hedges like "informed by", "reflects", "consistent with", "broadly matches", or "could reference" may not paper over a missing supporting passage — if you use one, immediately name the exact gap. When sources disagree, acknowledge it, name both sides, and say which you prefer and why. A candidate satisfying all but one constraint is not unambiguously the answer — state the unsatisfied constraint and treat the answer as tentative there, rather than redefining the constraint."""

# =============================================================================
# Tool definitions
# =============================================================================

_MULTI_SEARCH_TOOL = '{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}'

_SINGLE_SEARCH_TOOL = '{"type": "function", "function": {"name": "search", "description": "Perform a Google web search then returns a string of the top search results.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}}, "required": ["query"]}}}'

_VISIT_TOOL = '{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}'

_MULTI_GOOGLE_SCHOLAR_TOOL = '{"type": "function", "function": {"name": "google_scholar", "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries. This tool will also return results from google search", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries for Google Scholar."}}, "required": ["query"]}}}'

_SINGLE_GOOGLE_SCHOLAR_TOOL = '{"type": "function", "function": {"name": "google_scholar", "description": "Leverage Google Scholar to retrieve relevant information from academic publications.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query for Google Scholar."}}, "required": ["query"]}}}'

_PYTHON_INTERPRETER_TOOL = """{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code in a sandboxed environment. To use this tool, you must follow this format:
1. The 'arguments' JSON object must be empty: {}.
2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.

IMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.

Example of a correct call:
<tool_call>
{"name": "PythonInterpreter", "arguments": {}}
<code>
import numpy as np
# Your code here
print(f"The result is: {np.mean([1,2,3])}")
</code>
</tool_call>", "parameters": {"type": "object", "properties": {}, "required": []}}}"""

_PYTHON_INTERPRETER_TOOL_STATELESS = """{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code in a sandboxed environment. Each invocation runs in a completely fresh process: variables, imports, and any other state from previous calls are NOT preserved. If you need results from an earlier execution, you must redefine or recompute them in the current code. To use this tool, you must follow this format:
1. The 'arguments' JSON object must be empty: {}.
2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.

IMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.

Example of a correct call:
<tool_call>
{"name": "PythonInterpreter", "arguments": {}}
<code>
import numpy as np
# Your code here
print(f"The result is: {np.mean([1,2,3])}")
</code>
</tool_call>", "parameters": {"type": "object", "properties": {}, "required": []}}}"""

_TOOLS_HEADER = """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
"""

_TOOLS_FOOTER = """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


def _render_tools_block_from_dicts(tools):
    if not tools:
        return ""
    tool_lines = [
        json.dumps(t, ensure_ascii=False, separators=(", ", ": "))
        for t in tools
    ]
    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        + "\n".join(tool_lines)
        + "\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def _with_current_date(base_preamble, current_date):
    base = (base_preamble or "").rstrip()
    return base + "\n\nCurrent date: " + str(current_date)


def _compose_system_prompt(base_preamble, tools):
    base = (base_preamble or "").rstrip()
    tools_block = _render_tools_block_from_dicts(tools)
    return base + ("\n\n" + tools_block if tools_block else "")

# =============================================================================
# Sub-agent preamble
# =============================================================================

_SUB_AGENT_BASE_PREAMBLE = """You are a deep search assistant. Your primary role is to perform rigorous, multi-step, multi-source investigations on any topic, covering both broad, open-domain questions and highly specialized academic inquiries.

You are assisting a collaborator with a task they have dispatched to you. Their task description follows as the user message.

To complete this task, you must actively seek out and cross-check information from credible and diverse sources, then integrate the findings into a response that is comprehensive, accurate, well-structured, and objective.

## Operating principles
1. **Plan and execute research**: Break complex questions into sub-questions, gather evidence across multiple sources, and prioritize primary sources and authoritative references when available.
2. **Compare candidates explicitly.** Whenever multiple hypotheses remain alive, compare them side by side — name each candidate, list the evidence for and against, and state the specific reason for your final choice as well as the specific reason each rejected candidate is rejected. Do this throughout research and explicitly in your final report.
3. **Search strategically.** The search tool is well-tuned. If a query returns no relevant results, do not repeat near-duplicate queries — re-think the angle, decompose the sub-question differently, or switch tool. Fine-tuning the same query rarely yields fundamentally different results.
4. **Evaluate source quality**: Prefer reputable institutions, peer-reviewed research, official documentation, and high-quality journalism. Note uncertainty, conflicts, and limitations when sources disagree.
5. **Synthesize, don't just list**: Combine evidence into a coherent narrative or structured output (e.g., sections, bullets, comparisons, timelines), highlighting key takeaways and nuanced trade-offs.
6. **Maintain neutrality**: Present competing viewpoints fairly when relevant, and avoid unsupported speculation.

When you have collected sufficient information and are ready to deliver the final report, write your complete findings inside <report></report> — and prefer to say less than to include incorrect claims."""


_REQUIRED_OUTPUT_FORMAT_AND_REPORT_RULES = """## Rules for <report> (final-delivery turn only)

Purpose. The <report> is what your collaborator reads — a self-contained synthesis that addresses the dispatched task directly, presents your findings, and surfaces remaining uncertainty honestly. Do not assume the reader has seen your <think> or your tool calls; do not refer to "above" or "as discussed" — every claim must stand on its own inside the report.

Candidate comparison. When multiple candidates remained alive during research, compare them side by side inside the <report> — name each, list evidence for and against, and give the specific reason the chosen one wins and the specific reason each rejected one loses. The collaborator needs this reasoning to trust the conclusion.

Citations. Every important conclusion in the report — every named entity, date, place, factual claim, and any inference that depends on retrieved evidence — must carry an inline citation [n]. An inline citation [n] asserts that the source at reference [n] explicitly states or directly entails this specific claim. Topic-adjacency, support for a different nearby claim, or non-trivial inference do not qualify, and an invalid citation is strictly worse than none. If you cannot back a claim from retrieved text, either drop the claim or flag the gap explicitly inside the report.

Append a References section at the very end of the <report> block, listing every citation in order, formatted as:

    References
    [1] <page title> — <URL>
    [2] <page title> — <URL> (search snippet)

Append `(search snippet)` to a reference only when the supporting evidence is a search-result snippet you did not actually open via the visit tool — and only when the snippet itself directly states the claim. If a snippet is only suggestive, open the page via visit and confirm before citing. Never fabricate URLs — every reference URL must come from a page you actually visited or that appeared in your search results during this conversation.

Honesty. Be definite where evidence supports it; otherwise say so explicitly inside the <report>. When sources disagree on a relevant fact, acknowledge it, name both sides, and say which you prefer and why. A claim grounded only in topic-adjacent material is not supported — flag or drop it. Sloppy or fabricated citations destroy verifiability, making both the report and its conclusions untrustworthy."""


_SUB_AGENT_PREAMBLE = (
    _SUB_AGENT_BASE_PREAMBLE + "\n\n" + _REQUIRED_OUTPUT_FORMAT_AND_REPORT_RULES
)

_CALL_SUB_AGENT_TOOL = """{"type": "function", "function": {"name": "call_sub_agent", "description": "Dispatch research sub-tasks to independent agents running in parallel. Each agent can search the web and visit webpages. Coordinate each sub-agent as a new research collaborator joining the investigation for the first time. Make the division of labor explicit: what to investigate or verify, what evidence would be useful, and what result you need back. Then give the background needed to avoid wasted effort or the wrong target: why this sub-task matters, what is already established, what remains uncertain, which leads have been tried or ruled out, and where the weak points or contradictions are. Keep hypotheses, confirmed facts, and open gaps clearly separated. IMPORTANT: the sub-agent sees only the `prompt` field; the `goal` field is used only to label the sub-agent's response when it comes back to you.", "parameters": {"type": "object", "properties": {"prompts": {"type": "array", "items": {"type": "object", "properties": {"prompt": {"type": "string", "description": "A concrete research assignment for one sub-agent. State the task, expected output, useful evidence, context for the larger question, relevant constraints, and evidence state. Every sub-agent reads only its own brief — do NOT reduce detail in later briefs just because the first one was thorough."}, "goal": {"type": "string", "description": "A short one-line objective for this sub-task, used only to label the sub-agent's response when it returns. The sub-agent itself does not see this field."}}, "required": ["prompt", "goal"]}, "minItems": 1, "description": "A list of {prompt, goal} objects. Each object spawns one independent sub-agent; they run in parallel."}}, "required": ["prompts"]}}}"""

# =============================================================================
# Prompt construction
# =============================================================================

def get_preamble(prompt_mode="default"):
    """Return just the main-agent preamble (no tool definitions).

    With sub-agents enabled, use the sub-agent orchestration prompt. With
    sub-agents disabled, fall back to the no-sub-agent eval prompt.
    `prompt_mode` is accepted for backward-compatible env files but ignored.
    """
    enable_sub_agent = os.getenv('ENABLE_SUB_AGENT', '0') == '1'
    return _EVAL_PREAMBLE_WITH_SUBAGENT if enable_sub_agent else _SEARCH_PREAMBLE_NO_SUBAGENT


def get_system_prompt(prompt_mode="default", search_mode="multi", tool_type="four",
                      clarify_python=False, enable_sub_agent=False):
    """Return full system prompt with XML tool definitions. Used for local mode."""
    preamble = get_preamble(prompt_mode)
    tools = get_openai_tools(
        search_mode=search_mode,
        tool_type=tool_type,
        clarify_python=clarify_python,
        enable_sub_agent=enable_sub_agent,
    )
    return _compose_system_prompt(preamble, tools)


def get_openai_tools(search_mode="multi", tool_type="four", clarify_python=False,
                     enable_sub_agent=False):
    """Return tool definitions in OpenAI function calling format (list of dicts)."""
    tools = []

    if search_mode == "single":
        tools.append({"type": "function", "function": {
            "name": "search",
            "description": "Perform a Google web search then returns a string of the top search results.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "The search query."}
            }, "required": ["query"]}
        }})
    else:
        tools.append({"type": "function", "function": {
            "name": "search",
            "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}
            }, "required": ["query"]}
        }})

    tools.append({"type": "function", "function": {
        "name": "visit",
        "description": "Visit webpage(s) and return the summary of the content.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit."},
            "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}
        }, "required": ["url", "goal"]}
    }})

    if tool_type == "four":
        python_desc = (
            "Executes Python code in a sandboxed environment. Each invocation runs in a completely fresh process: "
            "variables, imports, and any other state from previous calls are NOT preserved. If you need results "
            "from an earlier execution, you must redefine or recompute them in the current code. "
            "Pass the code as a string in the 'code' argument. Any output must be printed to stdout using print()."
            if clarify_python else
            "Executes Python code in a sandboxed environment. Pass the code as a string in the 'code' argument. "
            "Any output must be printed to stdout using print()."
        )
        tools.append({"type": "function", "function": {
            "name": "PythonInterpreter",
            "description": python_desc,
            "parameters": {"type": "object", "properties": {
                "code": {"type": "string", "description": "The Python code to execute."}
            }, "required": ["code"]}
        }})

        if search_mode == "single":
            tools.append({"type": "function", "function": {
                "name": "google_scholar",
                "description": "Leverage Google Scholar to retrieve relevant information from academic publications.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string", "description": "The search query for Google Scholar."}
                }, "required": ["query"]}
            }})
        else:
            tools.append({"type": "function", "function": {
                "name": "google_scholar",
                "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries for Google Scholar."}
                }, "required": ["query"]}
            }})

    if enable_sub_agent:
        tools.append({"type": "function", "function": {
            "name": "call_sub_agent",
            "description": (
                "Dispatch research sub-tasks to independent agents running in parallel. "
                "Each agent can search the web and visit webpages. "
                "Coordinate each sub-agent as a new research collaborator joining the investigation for the first time. "
                "Make the division of labor explicit: what to investigate or verify, what evidence would be useful, "
                "and what result you need back. Then give the background needed to avoid wasted effort or the wrong target: "
                "why this sub-task matters, what is already established, what remains uncertain, which leads have been tried "
                "or ruled out, and where the weak points or contradictions are. Keep hypotheses, confirmed facts, and open gaps clearly separated. "
                "IMPORTANT: the sub-agent sees only the `prompt` field; the `goal` field is used only to label the sub-agent's response when it comes back to you."
            ),
            "parameters": {"type": "object", "properties": {
                "prompts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": (
                                    "A concrete research assignment for one sub-agent. State the task, expected output, useful evidence, "
                                    "context for the larger question, relevant constraints, and evidence state. Every sub-agent reads only "
                                    "its own brief \u2014 do NOT reduce detail in later briefs just because the first one was thorough."
                                )
                            },
                            "goal": {
                                "type": "string",
                                "description": (
                                    "A short one-line objective for this sub-task, used only to label the sub-agent's response when it returns. "
                                    "The sub-agent itself does not see this field."
                                )
                            }
                        },
                        "required": ["prompt", "goal"]
                    },
                    "minItems": 1,
                    "description": "A list of {prompt, goal} objects. Each object spawns one independent sub-agent; they run in parallel."
                }
            }, "required": ["prompts"]}
        }})

    return tools


def get_sub_agent_openai_tools(search_mode="multi", tool_type="two",
                               clarify_python=False):
    """Return OpenAI-format tool definitions for the sub-agent."""
    return get_openai_tools(search_mode=search_mode, tool_type=tool_type,
                            clarify_python=clarify_python,
                            enable_sub_agent=False)


def get_sub_agent_system_prompt(search_mode="multi", tool_type="two",
                                clarify_python=False):
    """Return full system prompt for sub-agent XML mode (preamble + tools)."""
    tools = get_sub_agent_openai_tools(
        search_mode=search_mode,
        tool_type=tool_type,
        clarify_python=clarify_python,
    )
    return _compose_system_prompt(_SUB_AGENT_PREAMBLE, tools)


ENABLE_SUB_AGENT = os.getenv('ENABLE_SUB_AGENT', '0') == '1'

PREAMBLE_ONLY = get_preamble(os.getenv("PROMPT_MODE", "default"))

OPENAI_TOOLS = get_openai_tools(
    os.getenv("SEARCH_MODE", "multi"),
    os.getenv("TOOL_TYPE", "four"),
    os.getenv("CLARIFY_PYTHON", "1") == "1",
    enable_sub_agent=ENABLE_SUB_AGENT,
)

SUB_AGENT_PREAMBLE = _SUB_AGENT_PREAMBLE

SUB_AGENT_OPENAI_TOOLS = get_sub_agent_openai_tools(
    os.getenv("SEARCH_MODE", "multi"),
    os.getenv("TOOL_TYPE", "four"),
    os.getenv("CLARIFY_PYTHON", "1") == "1",
)

SUB_AGENT_XML_SYSTEM_PROMPT = get_sub_agent_system_prompt(
    os.getenv("SEARCH_MODE", "multi"),
    os.getenv("TOOL_TYPE", "four"),
    os.getenv("CLARIFY_PYTHON", "1") == "1",
)

SYSTEM_PROMPT = _compose_system_prompt(PREAMBLE_ONLY, OPENAI_TOOLS)


def render_main_system_prompt(current_date, include_tools=False):
    base = _with_current_date(PREAMBLE_ONLY, current_date)
    return _compose_system_prompt(base, OPENAI_TOOLS) if include_tools else base


def render_sub_agent_system_prompt(current_date, include_tools=False):
    base = _with_current_date(SUB_AGENT_PREAMBLE, current_date)
    return _compose_system_prompt(base, SUB_AGENT_OPENAI_TOOLS) if include_tools else base

# =============================================================================
# Other prompts (unchanged)
# =============================================================================

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

JUDGE_PROMPT_VERIFIER = """You are an expert fact-checker. Your task is to verify whether the given answer is correct for the given question by using tools when necessary.

Question: {question}

Answer: {answer}

Is this answer factually correct? Respond ONLY with "Correct" or "Incorrect". Do not explain, justify, or output any other text.

Provide your final result in <answer>your result</answer>

"""
