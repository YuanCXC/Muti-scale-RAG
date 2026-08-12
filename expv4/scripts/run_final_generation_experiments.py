from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPV4 = ROOT / "expv4"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPV4))
sys.path.insert(0, str(SCRIPT_DIR))

from run_generation_pilot import _evaluate_one, _generate_one, _load_jsonl, _summary_rows
from src.config import ExperimentConfig
from src.evaluation import SemanticEvaluator
from src.generation.answer_generator import AnswerGenerator


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all v4 final generation experiments.")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=500)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_batch(
    tasks: list[Any],
    worker: Callable[[Any], dict[str, Any]],
    workers: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            append_jsonl(output_path, [row])
    return rows


def normalized_contexts(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _load_jsonl(path):
        row["method"] = row.get("method") or row.get("strategy") or row.get("variant")
        contexts[(row["id"], row["method"])] = row
    return contexts


def cumulative_counts(previous: list[int], current: list[int]) -> list[int]:
    return [old + new for old, new in zip(previous, current)]


def should_stop(rows: list[dict[str, Any]]) -> bool:
    failures = [row for row in rows if row["status"] != "success"]
    unexpected = [
        row
        for row in failures
        if "RateLimit" not in row.get("error", "")
        and "contentFilter" not in row.get("error", "")
        and "APIConnectionError" not in row.get("error", "")
        and "APITimeoutError" not in row.get("error", "")
        and "Request timed out" not in row.get("error", "")
    ]
    return len(failures) / len(rows) > 0.50 or len(unexpected) / len(rows) > 0.01


def main() -> None:
    args = arguments()
    config = ExperimentConfig.load(EXPV4 / "configs" / "proposed.json")
    if len(config.generation_api_keys) != 5:
        raise RuntimeError(f"Expected five BIGMOD API keys, found {len(config.generation_api_keys)}")

    retrieval_dir = EXPV4 / "results" / "final_2000_v4" / "retrieval"
    context_path = retrieval_dir / "retrieval_contexts_14_final_variants_v4.jsonl"
    output_dir = EXPV4 / "results" / "final_2000_v4" / "generation"
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = output_dir / "answers_cache_v4.jsonl"
    evaluations_path = output_dir / "semantic_evaluations_cache_v4.jsonl"
    progress_path = output_dir / "progress_v4.json"
    report_path = output_dir / "generation_experiment_report_v4.json"

    split = json.loads(
        (ROOT / "data" / "v4" / "final_evaluation_split_2000_v4.json").read_text(
            encoding="utf-8"
        )
    )
    selected_ids = set(split["evaluation_ids"])
    contexts = normalized_contexts(context_path)
    expected = 14 * len(selected_ids)
    if len(contexts) != expected:
        raise RuntimeError(f"Expected {expected} final contexts, found {len(contexts)}")
    method_counts = Counter(method for _, method in contexts)
    if len(method_counts) != 14 or set(method_counts.values()) != {2000}:
        raise RuntimeError(f"Invalid context distribution: {dict(method_counts)}")
    if "two_hop_extension" in method_counts:
        raise RuntimeError("Removed recursive bridge mechanism entered final contexts")

    dataset = pd.read_parquet(config.validation_file, columns=["id", "answer"])
    gold_answers = dict(zip(dataset["id"], dataset["answer"]))
    previous_progress = (
        json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    )
    base_generation_requests = previous_progress.get("generation_requests_by_key_slot", [0] * 5)
    base_evaluation_requests = previous_progress.get("evaluation_requests_by_key_slot", [0] * 5)

    generator = AnswerGenerator(
        model=config.generation_model,
        api_keys=config.generation_api_keys,
        base_url=config.generation_base_url,
        temperature=0.0,
        max_tokens=config.generation_max_tokens,
    )
    answer_rows = _load_jsonl(answers_path)
    completed_answers = {
        (row["id"], row["method"]): row
        for row in answer_rows
        if row.get("status") == "success" and row.get("answer")
    }
    attempted_answer_keys = {(row["id"], row["method"]) for row in answer_rows}
    generation_tasks = [
        row
        for key, row in contexts.items()
        if key not in completed_answers and key not in attempted_answer_keys
    ]
    generation_tasks.extend(
        row
        for key, row in contexts.items()
        if key not in completed_answers and key in attempted_answer_keys
    )
    completed_generation_count = len(completed_answers)
    for offset in range(0, len(generation_tasks), args.batch_size):
        batch = generation_tasks[offset : offset + args.batch_size]
        rows = run_batch(
            batch,
            lambda row: _generate_one(
                generator, row, str(gold_answers[row["id"]]), args.max_attempts
            ),
            args.workers,
            answers_path,
        )
        batch_failures = sum(row["status"] != "success" for row in rows)
        completed_generation_count += len(batch) - batch_failures
        progress = {
            "phase": "generation",
            "expected": expected,
            "completed": completed_generation_count,
            "remaining": expected - completed_generation_count,
            "last_batch_failures": batch_failures,
            "workers": args.workers,
            "model": config.generation_model,
            "thinking_mode": "disabled",
            "generation_requests_by_key_slot": cumulative_counts(
                base_generation_requests, generator.request_counts
            ),
            "evaluation_requests_by_key_slot": base_evaluation_requests,
        }
        write_json(progress_path, progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
        if should_stop(rows):
            raise RuntimeError(
                "Generation failures exceeded the stop rule for known transient/filter errors"
            )
        failure_rate = batch_failures / len(batch)
        if failure_rate > 0.40:
            time.sleep(300)
        elif failure_rate > 0.20:
            time.sleep(180)
        elif failure_rate > 0.05:
            time.sleep(60)
        elif failure_rate > 0.01:
            time.sleep(15)

    final_answer_rows = _load_jsonl(answers_path)
    successful_answers = {
        (row["id"], row["method"]): row
        for row in final_answer_rows
        if row.get("status") == "success" and row.get("answer")
    }
    if len(successful_answers) != expected:
        raise RuntimeError(
            f"Generation ended with {expected - len(successful_answers)} missing answers"
        )

    evaluator = SemanticEvaluator(
        model=config.generation_model,
        api_keys=config.generation_api_keys,
        base_url=config.generation_base_url,
    )
    evaluation_rows = _load_jsonl(evaluations_path)
    completed_evaluations = {
        (row["id"], row["method"]): row
        for row in evaluation_rows
        if row.get("status") == "success"
    }
    attempted_evaluation_keys = {(row["id"], row["method"]) for row in evaluation_rows}
    evaluation_tasks = [
        (key, row)
        for key, row in successful_answers.items()
        if key not in completed_evaluations and key not in attempted_evaluation_keys
    ]
    evaluation_tasks.extend(
        (key, row)
        for key, row in successful_answers.items()
        if key not in completed_evaluations and key in attempted_evaluation_keys
    )
    completed_evaluation_count = len(completed_evaluations)
    for offset in range(0, len(evaluation_tasks), args.batch_size):
        batch = evaluation_tasks[offset : offset + args.batch_size]
        rows = run_batch(
            batch,
            lambda task: _evaluate_one(
                evaluator, task[1], contexts[task[0]], args.max_attempts
            ),
            args.workers,
            evaluations_path,
        )
        batch_failures = sum(row["status"] != "success" for row in rows)
        completed_evaluation_count += len(batch) - batch_failures
        progress = {
            "phase": "evaluation",
            "expected": expected,
            "completed": completed_evaluation_count,
            "remaining": expected - completed_evaluation_count,
            "last_batch_failures": batch_failures,
            "workers": args.workers,
            "model": config.generation_model,
            "thinking_mode": "disabled",
            "evaluation_protocol": evaluator.protocol_version,
            "generation_requests_by_key_slot": cumulative_counts(
                base_generation_requests, generator.request_counts
            ),
            "evaluation_requests_by_key_slot": cumulative_counts(
                base_evaluation_requests, evaluator.request_counts
            ),
        }
        write_json(progress_path, progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
        if should_stop(rows):
            raise RuntimeError(
                "Evaluation failures exceeded the stop rule for known transient/filter errors"
            )
        failure_rate = batch_failures / len(batch)
        if failure_rate > 0.40:
            time.sleep(300)
        elif failure_rate > 0.20:
            time.sleep(180)
        elif failure_rate > 0.05:
            time.sleep(60)
        elif failure_rate > 0.01:
            time.sleep(15)

    final_evaluation_rows = _load_jsonl(evaluations_path)
    final_evaluations = {
        (row["id"], row["method"]): row
        for row in final_evaluation_rows
        if row.get("status") == "success"
    }
    if len(final_evaluations) != expected:
        raise RuntimeError(
            f"Evaluation ended with {expected - len(final_evaluations)} missing rows"
        )

    final_answers_path = output_dir / "answers_final_v4.jsonl"
    final_evaluations_path = output_dir / "semantic_evaluations_final_v4.jsonl"
    with final_answers_path.open("w", encoding="utf-8") as handle:
        for row in sorted(successful_answers.values(), key=lambda item: (item["id"], item["method"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with final_evaluations_path.open("w", encoding="utf-8") as handle:
        for row in sorted(final_evaluations.values(), key=lambda item: (item["id"], item["method"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summaries = _summary_rows(list(final_evaluations.values()))
    write_json(output_dir / "summary_v4.json", summaries)
    with (output_dir / "summary_v4.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    report = {
        "experiment": "final_generation_2000_v4",
        "model": config.generation_model,
        "thinking_mode": "disabled",
        "temperature": 0.0,
        "evaluation_protocol": evaluator.protocol_version,
        "samples": len(selected_ids),
        "type_counts": split["type_counts"],
        "context_variants": 14,
        "recursive_bridge_completion_included": False,
        "answers": len(successful_answers),
        "evaluations": len(final_evaluations),
        "generation_failures": 0,
        "evaluation_failures": 0,
        "generation_requests_by_key_slot": cumulative_counts(
            base_generation_requests, generator.request_counts
        ),
        "evaluation_requests_by_key_slot": cumulative_counts(
            base_evaluation_requests, evaluator.request_counts
        ),
        "answer_tokens": sum(row.get("total_tokens", 0) for row in successful_answers.values()),
        "evaluation_tokens": sum(row.get("total_tokens", 0) for row in final_evaluations.values()),
        "summary": summaries,
    }
    write_json(report_path, report)
    write_json(
        progress_path,
        {
            "phase": "complete",
            "expected": expected,
            "completed": expected,
            "remaining": 0,
            "generation_requests_by_key_slot": report["generation_requests_by_key_slot"],
            "evaluation_requests_by_key_slot": report["evaluation_requests_by_key_slot"],
        },
    )
    print(json.dumps({key: value for key, value in report.items() if key != "summary"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
