import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import threading
from datetime import datetime
from react_agent import MultiTurnReactAgent
import main_checkpoint as _main_ckpt
import time
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--dataset", type=str, default="gaia")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--presence_penalty", type=float, default=1.1)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--roll_out_count", type=int, default=3)
    parser.add_argument("--total_splits", type=int, default=1)
    parser.add_argument("--worker_split", type=int, default=1)
    args = parser.parse_args()

    model = args.model
    output_base = args.output
    roll_out_count = args.roll_out_count
    total_splits = args.total_splits
    worker_split = args.worker_split

    # Validate worker_split
    if worker_split < 1 or worker_split > total_splits:
        print(f"Error: worker_split ({worker_split}) must be between 1 and total_splits ({total_splits})")
        exit(1)

    model_name = os.path.basename(model.rstrip('/'))

    # model_dir = os.path.join(output_base, os.path.basename(os.path.dirname(args.dataset)), f"{model_name}_vllm")
    # dataset_dir = os.path.join(model_dir, args.dataset)
    experiment_name = os.getenv('EXPERIMENT_NAME', '')
    if experiment_name:
        model_dir = os.path.join(output_base, f"{model_name}_{experiment_name}")
    else:
        model_dir = os.path.join(output_base, f"{model_name}")

    os.makedirs(model_dir, exist_ok=True)

    print(f"Model name: {model_name}")
    print(f"Data set path: {args.dataset}")
    print(f"Output directory: {model_dir}")
    print(f"Main-agent checkpoints: {_main_ckpt.checkpoint_root(model_dir)}")
    print(f"Number of rollouts: {roll_out_count}")
    print(f"Data splitting: {worker_split}/{total_splits}")

    data_filepath = f"{args.dataset}"
    try:
        if data_filepath.endswith(".json"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
            if items and not isinstance(items[0], dict):
                raise ValueError("Input JSON list items must be objects.")
        elif data_filepath.endswith(".jsonl"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")
        items = items
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_filepath}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {data_filepath}: {e}")
        exit(1)

    # Apply data splitting
    total_items = len(items)
    items_per_split = math.ceil(total_items / total_splits)
    start_idx = (worker_split - 1) * items_per_split
    end_idx = min(worker_split * items_per_split, total_items)

    # Split the dataset
    items = items[start_idx:end_idx]

    print(f"Total items in dataset: {total_items}")
    print(f"Processing items {start_idx} to {end_idx-1} ({len(items)} items)")

    if total_splits > 1:
        # Add split suffix to output files when using splits
        output_files = {i: os.path.join(model_dir, f"iter{i}_split{worker_split}of{total_splits}.jsonl") for i in range(1, roll_out_count + 1)}
    else:
        output_files = {i: os.path.join(model_dir, f"iter{i}.jsonl") for i in range(1, roll_out_count + 1)}

    processed_records_per_rollout = {}

    for rollout_idx in range(1, roll_out_count + 1):
        output_file = output_files[rollout_idx]
        processed_item_indices = set()
        processed_question_counts = Counter()
        processed_total = 0
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "question" in data and "error" not in data:
                                processed_total += 1
                                if "item_index" in data:
                                    try:
                                        processed_item_indices.add(int(data["item_index"]))
                                        continue
                                    except (TypeError, ValueError):
                                        pass
                                processed_question_counts[data["question"].strip()] += 1
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid line in output file: {line.strip()}")
            except FileNotFoundError:
                pass
        processed_records_per_rollout[rollout_idx] = {
            "item_indices": processed_item_indices,
            "question_counts": processed_question_counts,
            "total": processed_total,
        }

    tasks_to_run_all = []
    per_rollout_task_counts = {i: 0 for i in range(1, roll_out_count + 1)}
    planning_ports = [6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008]
    planning_rr_idx = 0
    question_to_ports = {}
    model_mode = os.getenv('MODEL_MODE', 'local')
    for rollout_idx in range(1, roll_out_count + 1):
        processed_records = processed_records_per_rollout[rollout_idx]
        consumed_legacy_questions = Counter()
        for split_item_idx, item in enumerate(items):
            item_index = start_idx + split_item_idx
            question = (item.get("question") or item.get("task_question") or "").strip()
            if question == "":
                try:
                    user_msg = item["messages"][1]["content"]
                    question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                except Exception as e:
                    print(f"Extract question from user message failed: {e}")
            if question:
                item["question"] = question
                if "answer" not in item and "ground_truth" in item:
                    item["answer"] = item["ground_truth"]
            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            already_processed = item_index in processed_records["item_indices"]
            if not already_processed:
                legacy_done_count = processed_records["question_counts"].get(question, 0)
                if consumed_legacy_questions[question] < legacy_done_count:
                    consumed_legacy_questions[question] += 1
                    already_processed = True

            if not already_processed:
                if model_mode == 'api':
                    planning_port = None
                else:
                    if question not in question_to_ports:
                        planning_port = planning_ports[planning_rr_idx % len(planning_ports)]
                        question_to_ports[question] = planning_port
                        planning_rr_idx += 1
                    planning_port = question_to_ports[question]
                tasks_to_run_all.append({
                    "item": item.copy(),
                    "rollout_idx": rollout_idx,
                    "item_index": item_index,
                    "planning_port": planning_port,
                    "checkpoint_path": _main_ckpt.build_checkpoint_path(
                        model_dir=model_dir,
                        rollout_idx=rollout_idx,
                        question=question,
                        worker_split=worker_split,
                        total_splits=total_splits,
                        item_index=item_index,
                    ),
                })
                per_rollout_task_counts[rollout_idx] += 1

    print(f"Total questions in current split: {len(items)}")
    for rollout_idx in range(1, roll_out_count + 1):
        print(f"Rollout {rollout_idx}: already successfully processed: {processed_records_per_rollout[rollout_idx]['total']}, to run: {per_rollout_task_counts[rollout_idx]}")

    model_mode = os.getenv('MODEL_MODE', 'local')

    if not tasks_to_run_all:
        print("All rollouts have been completed and no execution is required.")
    else:
        llm_cfg = {
            'model': model,
            'generate_cfg': {
                'max_input_tokens': 320000,
                'max_retries': 10,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'presence_penalty': args.presence_penalty
            }
        }

        test_agent = MultiTurnReactAgent(
            llm=llm_cfg,
            function_list=["search", "visit", "google_scholar", "PythonInterpreter"]
        )
        print(f"Model mode: {model_mode}")

        write_locks = {i: threading.Lock() for i in range(1, roll_out_count + 1)}

        import _max_workers_control as _mwc
        sem = _mwc.init_control(initial_max=args.max_workers)
        pool_size = _mwc.pool_capacity()

        def _run_gated(task, model):
            with sem:
                return test_agent._run(task, model)

        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            future_to_task = {
                executor.submit(
                    _run_gated,
                    task,
                    model
                ): task for task in tasks_to_run_all
            }

            for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run_all), desc="Processing All Rollouts"):
                task_info = future_to_task[future]
                rollout_idx = task_info["rollout_idx"]
                output_file = output_files[rollout_idx]
                try:
                    result = future.result()
                    if isinstance(result, dict) and "rollout_id" not in result:
                        result["rollout_id"] = rollout_idx
                    if isinstance(result, dict) and "item_index" not in result:
                        result["item_index"] = task_info.get("item_index")
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    _main_ckpt.remove_checkpoint(task_info.get("checkpoint_path"))
                except concurrent.futures.TimeoutError:
                    question = task_info["item"].get("question", "")
                    print(f'Timeout (>1800s): "{question}" (Rollout {rollout_idx})')
                    future.cancel()
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "item_index": task_info.get("item_index"),
                        "error": "Timeout (>1800s)",
                        "messages": [],
                        "prediction": "[Failed]"
                    }
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                except Exception as exc:
                    question = task_info["item"].get("question", "")
                    print(f'Task for question "{question}" (Rollout {rollout_idx}) generated an exception: {exc}')
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "item_index": task_info.get("item_index"),
                        "error": f"Future resolution failed: {exc}",
                        "messages": [],
                        "prediction": "[Failed]",
                    }
                    print("===============================")
                    print(error_result)
                    print("===============================")
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

    print("\nAll tasks completed!")

    print(f"\nAll {roll_out_count} rollouts completed!")
