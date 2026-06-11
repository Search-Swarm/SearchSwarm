import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from judge_prompt import (
    JUDGE_PROMPT_GAIA,
    JUDGE_PROMPT_BC_en,
    JUDGE_PROMPT_BC_zh,
    JUDGE_PROMPT_BROWSECOMP_OFFICIAL,
    GRADER_TEMPLATE_SEAL,
)
import os
from pathlib import Path
import json
import glob
import re
from collections import defaultdict
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from openai import OpenAI
import threading
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln

_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

thread_local = threading.local()

LOCAL_JUDGE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
API_JUDGE_MODEL = "gpt-4o-2024-11-20"
LOCAL_JUDGE_BASE_URL = "http://127.0.0.1:6001/v1"


def _runtime_arg(name, default):
    return getattr(globals().get("args", None), name, default)


def _with_v1_suffix(base_url):
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        base_url = "http://localhost:8000"
    if base_url.endswith("/v1"):
        return base_url
    return base_url + "/v1"


def get_judge_model_mode(model_mode=None):
    mode = (model_mode or os.environ.get("JUDGE_MODEL_MODE", "local")).strip().lower()
    if not mode:
        mode = "local"
    if mode not in {"local", "api"}:
        raise ValueError(f"Unsupported JUDGE_MODEL_MODE={mode!r}; expected 'local' or 'api'")
    return mode


def get_judge_model(model_mode=None):
    mode = get_judge_model_mode(model_mode)
    configured_model = (
        os.environ.get("JUDGE_MODEL_NAME", "").strip()
        or os.environ.get("JUDGE_MODEL", "").strip()
    )
    if configured_model:
        return configured_model
    return API_JUDGE_MODEL if mode == "api" else LOCAL_JUDGE_MODEL


def get_client(model_mode=None):
    mode = get_judge_model_mode(model_mode)
    attr_name = f"{mode}_judge_client"
    if not hasattr(thread_local, attr_name):
        if mode == "api":
            api_base = os.environ.get("API_BASE_URL", "http://localhost:8000")
            api_key = os.environ.get("API_KEY", "")
            client = OpenAI(
                base_url=_with_v1_suffix(api_base),
                api_key=api_key,
                timeout=600.0,
            )
        else:
            client = OpenAI(
                base_url=LOCAL_JUDGE_BASE_URL,
                api_key="EMPTY"
            )
        setattr(thread_local, attr_name, client)
    return getattr(thread_local, attr_name)




def _is_browsecomp_dataset(dataset_name):
    return bool(dataset_name and dataset_name.startswith("browsecomp"))


def get_judge_prompt_from_folder(folder_path):
    """Determine judge prompt based on folder name (case-insensitive).

    Returns (judge_prompt, dataset_name) or (None, None) if no match.
    Priority: bc+zh > bc > gaia.
    """
    folder_name = os.path.basename(os.path.normpath(folder_path)).lower()
    is_browsecomp = "bc" in folder_name or "browsecomp" in folder_name
    if is_browsecomp and "zh" in folder_name:
        return JUDGE_PROMPT_BC_zh, "browsecomp_zh"
    elif is_browsecomp:
        return JUDGE_PROMPT_BC_en, "browsecomp_en"
    elif "gaia" in folder_name:
        return JUDGE_PROMPT_GAIA, "gaia"
    elif "seal" in folder_name:
        return GRADER_TEMPLATE_SEAL, "seal_0"
    return None, None


def _jsonl_sidecar_path(input_path, suffix):
    path = Path(input_path)
    if path.suffix == ".jsonl":
        return str(path.with_name(path.stem + suffix + path.suffix))
    return str(path) + suffix + ".jsonl"


def _tag_value(text, tag):
    match = re.search(
        rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>",
        text or "",
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _strip_think_blocks(text):
    return re.sub(
        r"<think>.*?</think>\s*",
        "",
        text or "",
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def _last_assistant_content_with_answer(item):
    messages = item.get("messages", [])
    fallback = ""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        if fallback == "":
            fallback = content
        if re.search(r"<answer>.*?</answer>", content, re.DOTALL | re.IGNORECASE):
            return content
    return fallback


def _extract_answer_only_response(item):
    prediction_key = _runtime_arg("prediction_key", "prediction")
    prediction = item.get(prediction_key, "")
    if prediction is not None and str(prediction).strip():
        return str(prediction).strip()

    final_content = _last_assistant_content_with_answer(item)
    answer = _tag_value(final_content, "answer")
    if answer is not None and answer.strip():
        return answer.strip()
    return _strip_think_blocks(final_content)


def _extract_full_final_response(item):
    final_content = _strip_think_blocks(_last_assistant_content_with_answer(item))
    explanation = _tag_value(final_content, "explanation")
    answer = _tag_value(final_content, "answer")

    if explanation is not None and answer is not None:
        return (
            f"<explanation>{explanation}</explanation>\n"
            f"<answer>{answer}</answer>"
        )
    if final_content:
        return final_content
    return _extract_answer_only_response(item)


def _judge_response_for_item(item, dataset_name=None):
    return _extract_answer_only_response(item)


def _compact_judge_record(item, judgement_result, dataset_name=None):
    question_key = _runtime_arg("question_key", "question")
    answer_key = _runtime_arg("answer_key", "answer")
    judge_response = judgement_result.get("judge_response")
    if judge_response is None:
        judge_response = item.get("judge_response")
    if judge_response is None:
        judge_response = _judge_response_for_item(item, dataset_name)
    return {
        "question": item.get(question_key, judgement_result.get("question", "")),
        "answer": item.get(answer_key, judgement_result.get("answer", "")),
        "prediction": judge_response,
        "judgement": judgement_result.get("judgement", ""),
    }


def _compact_judge_record_from_scored(scored_item, dataset_name=None):
    judgement = scored_item.get("origin_judgement")
    if judgement is None:
        judgement = "Correct" if scored_item.get("is_correct", False) else "Incorrect"
    return _compact_judge_record(scored_item, {
        "question": scored_item.get(_runtime_arg("question_key", "question"), ""),
        "answer": scored_item.get(_runtime_arg("answer_key", "answer"), ""),
        "judge_response": scored_item.get("judge_response"),
        "judgement": judgement,
    }, dataset_name)


def _write_compact_judge_file(compact_file, records):
    with open(compact_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def _official_correct_value_from_json(raw_text):
    text = (raw_text or "").strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict) and "correct" in parsed:
            return str(parsed["correct"]).strip().lower()
    return None


def _parse_official_judgement(raw_text):
    correct_value = _official_correct_value_from_json(raw_text)
    if correct_value is None:
        match = re.search(
            r"(?im)^\s*correct\s*:\s*(yes|no|true|false|correct|incorrect)\b",
            raw_text or "",
        )
        if match:
            correct_value = match.group(1).strip().lower()

    if correct_value is None:
        stripped = (raw_text or "").strip().lower().strip(".")
        if stripped in {"yes", "y", "true", "correct"}:
            correct_value = "yes"
        elif stripped in {"no", "n", "false", "incorrect"}:
            correct_value = "no"

    if correct_value in {"yes", "y", "true", "correct"}:
        return "Correct"
    return "Incorrect"


def _parse_judgement(raw_text, official_prompt=False):
    if official_prompt:
        return _parse_official_judgement(raw_text)
    return "Correct" if (raw_text or "")[:1] in ["a", "A"] else raw_text


def call_llm_judge(item, judge_prompt, max_retries=10, client=None, model=None, model_mode=None, dataset_name=None):
    """Judge if predicted answer matches ground-truth"""
    if model is None:
        model = get_judge_model(model_mode)

    question = item.get(_runtime_arg("question_key", "question"), "")
    correct_answer = item.get(_runtime_arg("answer_key", "answer"), "")
    response = _judge_response_for_item(item, dataset_name)
    if response is None or not str(response).strip():
        return {
            "question": question,
            "answer": correct_answer,
            "judge_response": response or "",
            "judgement": "Incorrect",
            "reason": "empty_prediction",
        }
    response = str(response).strip()
    official_prompt = judge_prompt == JUDGE_PROMPT_BROWSECOMP_OFFICIAL

    for _ in range(max_retries):
        try:
            if client is None:
                client = get_client(model_mode)
            prompt = judge_prompt.format(question=question, correct_answer=correct_answer, response=response)

            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )
            raw_judgement = completion.choices[0].message.content
            judgement = _parse_judgement(raw_judgement, official_prompt)

            if judgement == "Correct" and _runtime_arg("print_correct_question", False):
                print("Correct Question: ", question, "Prediction ", response, "Ground-truth", correct_answer, "\n")

            return {
                "question": question,
                "answer": correct_answer,
                "judge_response": response,
                "judgement": judgement,
                "raw_judgement": raw_judgement,
            }

        except Exception as e:
            time.sleep(1)
            if _ == max_retries - 1:
                print(f"Error judgement for question: {question}: {e}")
                return {
                    "question": question,
                    "answer": correct_answer,
                    "judge_response": response,
                    "judgement": "Error",
                    "error": str(e),
                 }


def single_round_statistics(input_file, available_tools=None):
    """Calculate statistics for a single round"""
    def avg_statistic(value_list):
        if value_list:
            return sum(value_list) / len(value_list)
        return 0

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        return {}

    num_invalid = 0
    tool_invocation = defaultdict(list)
    answer_lengths, traj_lengths = [], []

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
    except Exception as e:
        import tiktoken
        tokenizer = tiktoken.encoding_for_model("gpt-4o")

    for sample in samples:
        msgs = sample.get("messages", [])
        final_msg = msgs[-1]["content"] if len(msgs) else ""

        if "<answer>" not in final_msg or "</answer>" not in final_msg:
            num_invalid += 1
            answer_length = 0
        else:
            answer = final_msg.split("<answer>")[1].split("</answer>")[0].strip()
            answer_length = len(tokenizer.encode(answer))
        answer_lengths.append(answer_length)

        cur_tool_invocation = defaultdict(int)
        for msg in msgs:
            if msg["role"] == "assistant":
                try:
                    tool_call = msg["content"].split("<tool_call>")[1].split("</tool_call>")[0].strip()
                    tool_call = json.loads(tool_call)
                    tool_name = tool_call["name"]
                    if available_tools and tool_name in available_tools:
                        cur_tool_invocation[tool_name] += 1
                    else:
                        cur_tool_invocation["invalid"] += 1
                    cur_tool_invocation["total"] += 1
                except:
                    continue

        for k, v in cur_tool_invocation.items():
            tool_invocation[k].append(v)

        traj_length = len(tokenizer.encode("".join(msg["content"] for msg in msgs)))
        traj_lengths.append(traj_length)

    metrics = {
        "num_invalid": num_invalid,
        "avg_answer_length": avg_statistic(answer_lengths),
        "avg_traj_length": avg_statistic(traj_lengths)
    }

    for k, v in tool_invocation.items():
        if k != "invalid":
            metrics[f"avg_tool_{k}"] = avg_statistic(v)
        else:
            metrics[f"avg_tool_invalid"] = sum(v) / len(samples)

    return metrics


def process_one_prediction(prediction_file, judge_prompt, recalculate=False, client=None, model=None, model_mode=None, dataset_name=None):
    try:
        iteration_name = prediction_file.split("/")[-1].replace(".jsonl", "")

        scored_file = _jsonl_sidecar_path(prediction_file, "_scored")
        compact_file = _jsonl_sidecar_path(prediction_file, "_judge_results")
        if not recalculate and os.path.exists(scored_file):
            print(f"Found existing scored file for {iteration_name}, loading results...")
            with open(scored_file, 'r', encoding='utf-8') as f:
                scored_items = [json.loads(line) for line in f]

            compact_records = [
                _compact_judge_record_from_scored(item, dataset_name)
                for item in scored_items
            ]
            _write_compact_judge_file(compact_file, compact_records)
            print(f"Saved compact judge results to: {compact_file}")

            correct_predictions = []
            score_dict = defaultdict(bool)

            for scored_item in scored_items:
                if scored_item.get("is_correct", False):
                    correct_predictions.append({
                        "question": scored_item["question"],
                        "answer": scored_item["answer"],
                    })
                score_dict[scored_item["question"]] = scored_item.get("is_correct", False)

            acc = round(len(correct_predictions) / len(scored_items) * 100, 2)
            print(f"Loaded scored file: {scored_file} has {len(correct_predictions)} correct predictions (total {len(scored_items)}). Pass@1 {acc}%")

            return {
                "file": prediction_file,
                "accuracy": acc,
                "correct_count": len(correct_predictions),
                "total_count": len(scored_items),
                "correct_predictions": correct_predictions,
                "score_dict": score_dict
            }

        with open(prediction_file, 'r') as file:
            predictions = [json.loads(line) for line in file]

        correct_predictions, score_dict = [], defaultdict(bool)
        judgement_results = [None] * len(predictions)

        with ThreadPoolExecutor(max_workers=_runtime_arg("max_workers", 16)) as executor:
            future_to_index = {
                executor.submit(
                    call_llm_judge,
                    item,
                    judge_prompt,
                    10,
                    client,
                    model,
                    model_mode,
                    dataset_name,
                ): idx
                for idx, item in enumerate(predictions)
            }
            for future in tqdm(as_completed(future_to_index), desc=f"Judging {iteration_name}", total=len(future_to_index)):
                idx = future_to_index[future]
                result = future.result()
                judgement_results[idx] = result

                if result["judgement"] == "Correct":
                    correct_predictions.append({
                        "question": result["question"],
                        "answer": result["answer"],
                    })

                score_dict[result["question"]] = result["judgement"] == "Correct"

        acc = round(len(correct_predictions) / len(predictions) * 100, 2)
        print(f"Prediction file: {prediction_file} has {len(correct_predictions)} correct predictions (total {len(predictions)}). Pass@1 {acc}%")

        print(f"Saving scored results for {iteration_name}...")
        with open(scored_file, 'w', encoding='utf-8') as f:
            for orig_item, judgement_result in zip(predictions, judgement_results):
                save_item = orig_item.copy()
                save_item["is_correct"] = judgement_result["judgement"] == "Correct"
                save_item["origin_judgement"] = judgement_result["judgement"]
                save_item["judge_response"] = judgement_result.get("judge_response", "")

                if "error" in judgement_result:
                    save_item["error"] = judgement_result["error"]
                if "reason" in judgement_result:
                    save_item["judge_reason"] = judgement_result["reason"]
                if "raw_judgement" in judgement_result:
                    save_item["judge_raw_response"] = judgement_result["raw_judgement"]
                f.write(json.dumps(save_item, ensure_ascii=False) + '\n')

        compact_records = [
            _compact_judge_record(orig_item, judgement_result, dataset_name)
            for orig_item, judgement_result in zip(predictions, judgement_results)
        ]
        _write_compact_judge_file(compact_file, compact_records)
        print(f"Saved compact judge results to: {compact_file}")

        return {
            "file": prediction_file,
            "accuracy": acc,
            "correct_count": len(correct_predictions),
            "total_count": len(predictions),
            "correct_predictions": correct_predictions,
            "score_dict": score_dict
        }
    except Exception as e:
        print(f"Error processing file {prediction_file}: {e}")
        return {
            "file": prediction_file,
            "error": str(e)
        }


def _eb_neg_log_likelihood(params, data):
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return 1e12
    ll = 0.0
    for s_i, n_i in data:
        ll += betaln(alpha + s_i, beta + n_i - s_i) - betaln(alpha, beta)
    return -ll


def estimate_empirical_bayes_prior(data):
    p_hats = [s / n for s, n in data if n > 0]
    p_mean = np.mean(p_hats)
    p_var = np.var(p_hats, ddof=0)
    n_mean = np.mean([n for _, n in data if n > 0])

    p_var_prior = max(p_var - p_mean * (1 - p_mean) / n_mean, 1e-6)
    factor = max(p_mean * (1 - p_mean) / p_var_prior - 1, 0.1)
    alpha_init = min(max(p_mean * factor, 0.5), 100)
    beta_init = min(max((1 - p_mean) * factor, 0.5), 100)

    result = minimize(
        _eb_neg_log_likelihood,
        x0=[alpha_init, beta_init],
        args=(data,),
        method='L-BFGS-B',
        bounds=[(1e-4, 1e6), (1e-4, 1e6)]
    )

    if result.success:
        return result.x[0], result.x[1]
    return alpha_init, beta_init


def compute_hpd(samples, credible_mass):
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    k = int(np.ceil(credible_mass * n))
    widths = sorted_samples[k - 1:] - sorted_samples[:n - k + 1]
    min_idx = int(np.argmin(widths))
    return [round(float(sorted_samples[min_idx]) * 100, 2),
            round(float(sorted_samples[min_idx + k - 1]) * 100, 2)]


def bayesian_confidence_intervals(all_scores_dict, n_mc_samples=200000):
    data = []
    for scores in all_scores_dict.values():
        n_i = len(scores)
        s_i = sum(scores)
        if n_i > 0:
            data.append((s_i, n_i))
    if not data:
        return {}

    K = len(data)
    alpha0, beta0 = estimate_empirical_bayes_prior(data)

    rng = np.random.default_rng(42)
    theta_samples = np.zeros(n_mc_samples)
    for s_i, n_i in data:
        post_alpha = alpha0 + s_i
        post_beta = beta0 + n_i - s_i
        theta_samples += rng.beta(post_alpha, post_beta, n_mc_samples)
    theta_samples /= K

    return {
        "bayesian_mean": round(float(np.mean(theta_samples)) * 100, 2),
        "ci_90": compute_hpd(theta_samples, 0.90),
        "ci_95": compute_hpd(theta_samples, 0.95),
        "eb_prior_alpha": round(float(alpha0), 4),
        "eb_prior_beta": round(float(beta0), 4),
    }


def process_folder(input_folder, judge_prompt, dataset_name, available_tools, recalculate=False, skip_statistics=False):
    """Process a single result folder: judge predictions and compute metrics."""
    print(f"\n{'='*60}")
    print(f"Processing folder: {input_folder}")
    print(f"Judge prompt: {dataset_name}")
    print(f"Recalculate: {recalculate}")
    print(f"{'='*60}")

    judge_model_mode = get_judge_model_mode()
    judge_model = get_judge_model(judge_model_mode)
    print(f"Judge model mode: {judge_model_mode}")
    print(f"Judge model: {judge_model}")
    if _is_browsecomp_dataset(dataset_name):
        print("BrowseComp judge prompt mode: legacy")
        print("BrowseComp judge response mode: answer_only")

    all_scores_dict = defaultdict(list)
    acc_list = []
    file_list = []

    for path in glob.glob(os.path.join(input_folder, "iter*.jsonl")):
        if path.endswith("_scored.jsonl") or path.endswith("_judge_results.jsonl"):
            continue

        result = process_one_prediction(
            path,
            judge_prompt,
            recalculate,
            None,
            judge_model,
            judge_model_mode,
            dataset_name,
        )
        if "error" not in result:
            for question, score in result["score_dict"].items():
                all_scores_dict[question].append(score)
            acc_list.append(result["accuracy"])
            file_list.append(path)

    if not acc_list:
        print(f"No valid results found in {input_folder}!")
        return None

    avg_pass_at_1 = sum(acc_list) / len(acc_list)
    print(f"Average Pass@1: {avg_pass_at_1:.2f}")

    best_pass_at_1 = max(acc_list)
    print(f"Best Pass@1: {best_pass_at_1:.2f}")

    correct_num = 0
    for question, scores in all_scores_dict.items():
        if sum(scores) >= 1:
            correct_num += 1
    pass_at_k = correct_num / len(all_scores_dict) * 100
    print(f"Pass@{len(acc_list)}: {pass_at_k:.2f}")

    print("\n========== Bayesian Confidence Intervals ==========")
    bayes_result = bayesian_confidence_intervals(all_scores_dict)
    if bayes_result:
        print(f"EB Prior: Beta(alpha={bayes_result['eb_prior_alpha']}, beta={bayes_result['eb_prior_beta']})")
        print(f"Bayesian Mean Accuracy: {bayes_result['bayesian_mean']:.2f}%")
        print(f"90% HPD CI: [{bayes_result['ci_90'][0]:.2f}%, {bayes_result['ci_90'][1]:.2f}%]")
        print(f"95% HPD CI: [{bayes_result['ci_95'][0]:.2f}%, {bayes_result['ci_95'][1]:.2f}%]")

    avg_stats = {}
    if not skip_statistics:
        print("\n========== Statistics ==========")
        all_stats = []
        for file_path in file_list:
            stats = single_round_statistics(file_path, available_tools)
            if stats:
                all_stats.append(stats)

        if all_stats:
            for key in all_stats[0].keys():
                avg_stats[key] = round(sum(stats.get(key, 0) for stats in all_stats) / len(all_stats), 2)

            print(f"# Invalid: {avg_stats.get('num_invalid', 0)}")
            print(f"Avg. Answer Length: {avg_stats.get('avg_answer_length', 0)}")
            print(f"Avg. Trajectory Length: {avg_stats.get('avg_traj_length', 0)}")

            for k, v in avg_stats.items():
                if k.startswith("avg_tool_"):
                    print(f"{k}: {v}")

    overall_dict = {
        "avg_pass_at_1": avg_pass_at_1,
        "best_pass_at_1": best_pass_at_1,
        "pass_at_k": pass_at_k,
    }
    if bayes_result:
        overall_dict.update({
            "bayesian_mean": bayes_result["bayesian_mean"],
            "ci_90": bayes_result["ci_90"],
            "ci_95": bayes_result["ci_95"],
            "eb_prior_alpha": bayes_result["eb_prior_alpha"],
            "eb_prior_beta": bayes_result["eb_prior_beta"],
        })

    overall_eval_dict = {
        "dataset": dataset_name,
        "files": file_list,
        "overall": overall_dict,
        "individual": {f"iter{i+1}_pass_at_1": acc for i, acc in enumerate(acc_list)},
        "statistics": avg_stats
    }

    summary_path = os.path.join(input_folder, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(overall_eval_dict, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {summary_path}")

    return overall_eval_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default=None)
    parser.add_argument("--multi_input", default=None, help="Parent folder containing multiple result subfolders")
    parser.add_argument("--question_key", type=str, default="question")
    parser.add_argument("--answer_key", type=str, default="answer")
    parser.add_argument("--prediction_key", type=str, default="prediction")
    parser.add_argument("--print_correct_question", action="store_true")
    parser.add_argument("--available_tools", type=str, default="search,visit")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--recalculate", action="store_true", default=False, help="Force recalculate and overwrite existing scored files")
    parser.add_argument("--skip_statistics", action="store_true", default=False, help="Skip statistics calculation (token lengths, tool invocations, etc.)")
    args = parser.parse_args()

    available_tools = args.available_tools.split(",") if args.available_tools else ["search", "visit"]

    if args.multi_input:
        if not os.path.isdir(args.multi_input):
            print(f"Error: {args.multi_input} is not a valid directory")
            exit(1)

        subfolders = sorted([
            os.path.join(args.multi_input, d)
            for d in os.listdir(args.multi_input)
            if os.path.isdir(os.path.join(args.multi_input, d))
        ])

        multi_results = {}
        for folder in subfolders:
            judge_prompt, dataset_name = get_judge_prompt_from_folder(folder)
            if judge_prompt is None:
                print(f"\nSkipping folder: {folder} (no matching dataset pattern in folder name)")
                multi_results[os.path.basename(folder)] = {"status": "skipped", "reason": "no matching dataset pattern"}
                continue
            result = process_folder(folder, judge_prompt, dataset_name, available_tools, args.recalculate, args.skip_statistics)
            if result is not None:
                multi_results[os.path.basename(folder)] = result
            else:
                multi_results[os.path.basename(folder)] = {"status": "skipped", "reason": "no valid results"}

        multi_summary_path = os.path.join(args.multi_input, "multi_summary.json")
        with open(multi_summary_path, 'w', encoding='utf-8') as f:
            json.dump(multi_results, f, ensure_ascii=False, indent=2)
        print(f"\nMulti-input summary saved to {multi_summary_path}")

    elif args.input_folder:
        judge_prompt, dataset_name = get_judge_prompt_from_folder(args.input_folder)
        if judge_prompt is None:
            print(f"Error: Cannot determine dataset from folder name: {args.input_folder}")
            print("Folder name must contain 'bc' (for BrowseComp), 'gaia' (for GAIA), or 'seal' (for SealQA)")
            exit(1)
        process_folder(args.input_folder, judge_prompt, dataset_name, available_tools, args.recalculate, args.skip_statistics)

    else:
        print("Error: Please provide either --input_folder or --multi_input")
        exit(1)
