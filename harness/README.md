# SearchSwarm Harness

A ReAct-style deep-research agent with optional parallel sub-agents, for evaluating against BrowseComp and similar benchmarks.

## Quick Start

```bash
# 1. Edit .env with your model path, dataset, and API keys (SERPER_API_KEY, JINA_API_KEY).

# 2. If MODEL_MODE=local (or SUB_AGENT_MODE=local), start the local vLLM servers.
#    Brings up 8 servers on ports 6001-6008 and waits until all are ready.
bash deploy_model.sh

# 3. Run inference (reads everything from .env).
bash run_react_infer.sh
```

`run_react_infer.sh` sources `.env` and passes the values to `run_multi_react.py`, so you don't pass them on the command line — change behavior by editing `.env`.

## Evaluation data

This repo ships **only a tiny synthetic example**, `eval_data/example/standardized_data.jsonl`, to show the expected input format. `DATASET` in `.env` points to it by default.

The real benchmarks (BrowseComp, BrowseComp-zh, GAIA, …) are **not bundled here**. The standard practice for evaluation benchmarks is to **not redistribute their test data in plaintext** — that is how it leaks into training corpora and contaminates future results, which is why some are distributed encrypted or carry a "do-not-train" canary. Get them from their official sources, convert to the schema below, and point `DATASET` at your file.

Each line is one JSON object:

```json
{"task_question": "<the question>", "ground_truth": "<the gold answer>", "file_name": "", "metadata": {}}
```

`question` / `answer` are accepted as aliases for `task_question` / `ground_truth`.

## Configuration

All settings live in `.env`.

### Model and inference

| Variable | Description |
|---|---|
| `MODEL_MODE` | `local` = local vLLM on ports 6001–6008; `api` = remote OpenAI-compatible endpoint (`API_BASE_URL` + `API_KEY`). |
| `MODEL_PATH` | Passed as `--model`. Local mode: HuggingFace path/name served by vLLM (also used to load the tokenizer). API mode: the remote model name. |
| `TEMPLATE` | Local-mode template/parser variant: `qwen3` or `hermes_w_py`. Both drive XML-style tool calls parsed client-side. Ignored in API mode. |
| `API_BASE_URL` | API mode only. Base URL; `/v1` is appended. Default `http://localhost:8000`. |
| `API_KEY` | API mode only. |
| `DATASET` | Path to a `.json` / `.jsonl` evaluation file. |
| `OUTPUT_PATH` | Root directory for results. |
| `ROLLOUT_COUNT` | Independent rollouts per question. |
| `TEMPERATURE`, `TOP_P`, `PRESENCE_PENALTY` | Sampling parameters. |
| `MAX_WORKERS` | Questions evaluated concurrently. |
| `MAX_LLM_CALL_PER_RUN` | Maximum LLM calls per question rollout. When one call remains, the agent is forced to emit its final answer. |
| `MAX_CONTEXT_TOKENS` | Context-window budget. When the prompt would exceed `MAX_CONTEXT_TOKENS − MAX_GENERATION_TOKENS`, the agent rolls back the last round and is forced to answer. |
| `MAX_GENERATION_TOKENS` | Per-call `max_tokens`. |
| `TOKEN_COUNTER` | `local` = count tokens with the HuggingFace tokenizer; `api` = trust the server-reported `prompt_tokens`. |
| `RUN_TIMEOUT_MINUTES` | Per-question wallclock budget. |
| `SEARCH_NUM_RESULTS` | Results returned per search query. |
| `EXPERIMENT_NAME` | Appended to the results directory: `OUTPUT_PATH/<MODEL>_<EXPERIMENT_NAME>/`. |

### Tools

| Variable | Description |
|---|---|
| `SEARCH_MODE` | `multi` = the `search` and `google_scholar` tools take an array of queries run in one batched call; `single` = they take a single query string. |
| `TOOL_TYPE` | `two` = search + visit; `four` = + google_scholar + PythonInterpreter. |
| `SERPER_API_KEY` | Serper key for the search / scholar tools. |
| `JINA_API_KEY` | Jina key for the visit tool (page reading). |

### Sub-agents

Set `ENABLE_SUB_AGENT=1` to give the main agent a `call_sub_agent` tool that dispatches research subtasks to independent workers running in parallel. Each sub-agent has its own search/visit/scholar/python tools and cannot dispatch further sub-agents.

| Variable | Description |
|---|---|
| `SUB_AGENT_MODE` | `api` or `local`. Defaults to `MODEL_MODE`. |
| `SUB_AGENT_MODEL` | Sub-agent model name. Local mode: must match what vLLM serves (`VLLM_MODEL`); falls back to `MODEL_PATH`. |
| `VLLM_MODEL` | What `deploy_model.sh` serves on ports 6001–6008. Defaults to `MODEL_PATH`. Set separately when the main agent is an API model but sub-agents run locally. |
| `SUB_AGENT_MAX_CONTEXT_TOKENS` | Per-sub-agent context budget. |
| `SUB_AGENT_MAX_GENERATION_TOKENS` | Per-call `max_tokens` for sub-agents. |
| `SUB_AGENT_MAX_LLM_CALLS` | Maximum LLM calls per sub-agent task. |
| `SUB_AGENT_TIMEOUT_MINUTES` | Per-sub-agent wallclock budget. |
| `SUB_AGENT_TEMPERATURE`, `SUB_AGENT_TOP_P`, `SUB_AGENT_PRESENCE_PENALTY` | Sub-agent sampling. |

The main agent and the sub-agents each independently support `api` or `local` — set `MODEL_MODE` and `SUB_AGENT_MODE` to any combination.

### Judging

| Variable | Description |
|---|---|
| `JUDGE_MODEL_MODE` | `local` or `api`. |
| `JUDGE_MODEL_NAME` | Model name for LLM-as-judge (`llm_judge.py`). |

## Architecture

```
run_react_infer.sh          -- Reads .env, launches run_multi_react.py
deploy_model.sh             -- Starts 8 local vLLM servers (ports 6001-6008)
run_multi_react.py          -- Entry point: parallel rollouts
  react_agent.py            -- Main agent loop (ReAct with tool calling)
    tool_search.py          -- Web search via Serper
    tool_visit.py           -- Page reading via Jina + summary extraction
    tool_scholar.py         -- Google Scholar via Serper
    tool_python.py          -- Sandboxed Python execution
    tool_sub_agent.py       -- Sub-agent dispatch (parallel research workers)
  llm_client.py             -- API model config registry
  prompt.py                 -- System preambles and tool schemas
  main_checkpoint.py        -- Per-rollout checkpoint/resume
  llm_judge.py              -- LLM-as-judge evaluation
```

## Output

Results are written to `OUTPUT_PATH/<MODEL>_<EXPERIMENT_NAME>/`, one `iterN.jsonl` per rollout. Sub-agent trajectories are appended to `subagent_trajectories.jsonl`.
