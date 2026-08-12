from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPV4 = ROOT / "expv4"
sys.path.insert(0, str(EXPV4))

from src.config import ExperimentConfig
from src.evaluation import SemanticEvaluator, answer_exact_match, answer_f1
from src.generation.answer_generator import AnswerGenerator
from src.models import EvidenceUnit


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v4 answer-generation pilot.")
    parser.add_argument("--sample-per-type", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--output-name", default="generation_pilot_10_v4")
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _build_pilot_split(
    final_split_path: Path,
    pilot_split_path: Path,
    sample_per_type: int,
    seed: int,
) -> dict[str, Any]:
    if pilot_split_path.exists():
        return json.loads(pilot_split_path.read_text(encoding="utf-8"))

    final_split = json.loads(final_split_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in final_split["examples"]:
        grouped[example["type"]].append(example)

    rng = random.Random(seed)
    selected = []
    for question_type in ("bridge", "comparison"):
        selected.extend(rng.sample(grouped[question_type], sample_per_type))
    split = {
        "dataset_version": "v4-generation-pilot",
        "source_split": str(final_split_path.relative_to(ROOT)).replace("\\", "/"),
        "seed": seed,
        "sample_per_type": sample_per_type,
        "total_examples": len(selected),
        "type_counts": {"bridge": sample_per_type, "comparison": sample_per_type},
        "evaluation_ids": [row["id"] for row in selected],
        "examples": selected,
    }
    _write_json(pilot_split_path, split)
    return split


def _evidence(rows: list[dict[str, Any]]) -> list[EvidenceUnit]:
    return [EvidenceUnit(**row) for row in rows]


def _retry(action: Callable[[], dict[str, Any]], max_attempts: int) -> tuple[dict[str, Any], int]:
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            return action(), attempt
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            if "contentFilter" in last_error or "Error code: 400" in last_error:
                break
            if attempt < max_attempts:
                time.sleep(float(attempt * 3))
    raise RuntimeError(last_error)


def _generate_one(
    generator: AnswerGenerator,
    context_row: dict[str, Any],
    gold_answer: str,
    max_attempts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result, attempts = _retry(
            lambda: generator.generate_with_metadata(
                context_row["question"], _evidence(context_row["context_evidence"])
            ),
            max_attempts,
        )
        return {
            "id": context_row["id"],
            "method": context_row["method"],
            "label": context_row["label"],
            "question_type": context_row["question_type"],
            "question": context_row["question"],
            "gold_answer": gold_answer,
            "answer": result["answer"],
            "status": "success" if result["answer"] else "failed",
            "attempts": attempts,
            "latency_seconds": time.perf_counter() - started,
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
            "evidence_units": len(context_row["context_evidence"]),
            "error": "" if result["answer"] else "empty_answer",
        }
    except Exception as error:
        return {
            "id": context_row["id"],
            "method": context_row["method"],
            "label": context_row["label"],
            "question_type": context_row["question_type"],
            "question": context_row["question"],
            "gold_answer": gold_answer,
            "answer": "",
            "status": "failed",
            "attempts": max_attempts,
            "latency_seconds": time.perf_counter() - started,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "evidence_units": len(context_row["context_evidence"]),
            "error": str(error),
        }


def _evaluate_one(
    evaluator: SemanticEvaluator,
    answer_row: dict[str, Any],
    context_row: dict[str, Any],
    max_attempts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    base = {
        "id": answer_row["id"],
        "method": answer_row["method"],
        "label": answer_row["label"],
        "question_type": answer_row["question_type"],
        "answer_em": answer_exact_match(answer_row["answer"], answer_row["gold_answer"]),
        "answer_f1": answer_f1(answer_row["answer"], answer_row["gold_answer"]),
    }
    try:
        result, attempts = _retry(
            lambda: evaluator.evaluate(
                answer_row["question"],
                answer_row["gold_answer"],
                answer_row["answer"],
                _evidence(context_row["context_evidence"]),
            ),
            max_attempts,
        )
        return {
            **base,
            **result,
            "status": "success",
            "attempts": attempts,
            "latency_seconds": time.perf_counter() - started,
            "error": "",
        }
    except Exception as error:
        return {
            **base,
            "accuracy": None,
            "faithfulness": None,
            "answer_relevance": None,
            "context_relevance": None,
            "is_refusal": SemanticEvaluator.is_refusal(answer_row["answer"]),
            "raw_evaluation": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "status": "failed",
            "attempts": max_attempts,
            "latency_seconds": time.perf_counter() - started,
            "error": str(error),
        }


def _run_parallel(
    tasks: list[Any],
    worker: Callable[[Any], dict[str, Any]],
    output_path: Path,
    workers: int,
    phase: str,
) -> None:
    if not tasks:
        print(f"{phase}: nothing to run")
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            _append_jsonl(output_path, future.result())
            if completed == len(tasks) or completed % 10 == 0:
                print(f"{phase}: {completed}/{len(tasks)}")


def _summary_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluations:
        grouped[row["method"]].append(row)
    summary = []
    metrics = (
        "answer_em",
        "answer_f1",
        "accuracy",
        "faithfulness",
        "answer_relevance",
        "context_relevance",
    )
    for method, rows in grouped.items():
        valid = [row for row in rows if row["status"] == "success"]
        item: dict[str, Any] = {
            "method": method,
            "label": rows[0]["label"],
            "examples": len(rows),
            "successful_evaluations": len(valid),
            "refusal_rate": mean(float(row["is_refusal"]) for row in valid) if valid else None,
        }
        for metric in metrics:
            values = [float(row[metric]) for row in valid if row.get(metric) is not None]
            item[metric] = mean(values) if values else None
        summary.append(item)
    return sorted(summary, key=lambda row: row["method"])


def main() -> None:
    args = _arguments()
    config = ExperimentConfig.load(EXPV4 / "configs" / "proposed.json")
    if not config.generation_api_keys:
        raise RuntimeError("No BIGMOD API keys are configured")

    final_split_path = ROOT / "data" / "v4" / "final_evaluation_split_2000_v4.json"
    pilot_split_path = ROOT / "data" / "v4" / f"{args.output_name}_split.json"
    context_path = (
        EXPV4
        / "results"
        / "final_2000_v4"
        / "retrieval"
        / "retrieval_contexts_13_variants_v4.jsonl"
    )
    output_dir = EXPV4 / "results" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = output_dir / "answers_v4.jsonl"
    evaluations_path = output_dir / "semantic_evaluations_v4.jsonl"
    report_path = output_dir / "report_v4.json"
    previous_report = (
        json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    )

    pilot_split = _build_pilot_split(
        final_split_path,
        pilot_split_path,
        args.sample_per_type,
        args.seed,
    )
    selected_ids = set(pilot_split["evaluation_ids"])
    context_rows = [row for row in _load_jsonl(context_path) if row["id"] in selected_ids]
    for row in context_rows:
        row["method"] = row.get("method") or row.get("strategy") or row.get("variant")
    contexts = {(row["id"], row["method"]): row for row in context_rows}
    expected = len(selected_ids) * 13
    if len(contexts) != expected:
        raise RuntimeError(f"Expected {expected} pilot contexts, found {len(contexts)}")

    dataset = pd.read_parquet(config.validation_file, columns=["id", "answer"])
    gold_answers = dict(zip(dataset["id"], dataset["answer"]))

    generator = AnswerGenerator(
        model=config.generation_model,
        api_keys=config.generation_api_keys,
        base_url=config.generation_base_url,
        temperature=0.0,
        max_tokens=config.generation_max_tokens,
    )
    completed_answers = {
        (row["id"], row["method"]): row
        for row in _load_jsonl(answers_path)
        if row.get("status") == "success" and row.get("answer")
    }
    generation_tasks = [
        row for key, row in contexts.items() if key not in completed_answers
    ]
    _run_parallel(
        generation_tasks,
        lambda row: _generate_one(generator, row, str(gold_answers[row["id"]]), args.max_attempts),
        answers_path,
        args.workers,
        "generation",
    )

    answer_rows = _load_jsonl(answers_path)
    successful_answers = {
        (row["id"], row["method"]): row
        for row in answer_rows
        if row.get("status") == "success" and row.get("answer")
    }
    evaluator = SemanticEvaluator(
        model=config.generation_model,
        api_keys=config.generation_api_keys,
        base_url=config.generation_base_url,
    )
    completed_evaluations = {
        (row["id"], row["method"]): row
        for row in _load_jsonl(evaluations_path)
        if row.get("status") == "success"
    }
    evaluation_tasks = [
        (key, row)
        for key, row in successful_answers.items()
        if key not in completed_evaluations
    ]
    _run_parallel(
        evaluation_tasks,
        lambda task: _evaluate_one(
            evaluator,
            task[1],
            contexts[task[0]],
            args.max_attempts,
        ),
        evaluations_path,
        args.workers,
        "evaluation",
    )

    final_answers = {
        (row["id"], row["method"]): row for row in _load_jsonl(answers_path)
    }
    final_evaluations = {
        (row["id"], row["method"]): row for row in _load_jsonl(evaluations_path)
    }
    _write_jsonl(
        output_dir / "answers_final_v4.jsonl",
        sorted(final_answers.values(), key=lambda row: (row["id"], row["method"])),
    )
    _write_jsonl(
        output_dir / "semantic_evaluations_final_v4.jsonl",
        sorted(final_evaluations.values(), key=lambda row: (row["id"], row["method"])),
    )
    evaluation_rows = list(final_evaluations.values())
    summaries = _summary_rows(evaluation_rows)
    summary_csv = output_dir / "summary_v4.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    generation_failures = expected - sum(
        row.get("status") == "success" and bool(row.get("answer"))
        for row in final_answers.values()
    )
    evaluation_failures = expected - sum(
        row.get("status") == "success" for row in final_evaluations.values()
    )
    previous_generation_requests = previous_report.get(
        "generation_requests_by_key_slot", [0] * len(config.generation_api_keys)
    )
    previous_evaluation_requests = previous_report.get(
        "evaluation_requests_by_key_slot", [0] * len(config.generation_api_keys)
    )
    report = {
        "experiment": args.output_name,
        "model": config.generation_model,
        "thinking_mode": "disabled",
        "temperature": 0.0,
        "workers": args.workers,
        "configured_api_keys": len(config.generation_api_keys),
        "sample_count": len(selected_ids),
        "type_counts": pilot_split["type_counts"],
        "context_variants": 13,
        "expected_answers": expected,
        "successful_answers": expected - generation_failures,
        "generation_failures": generation_failures,
        "successful_evaluations": expected - evaluation_failures,
        "evaluation_failures": evaluation_failures,
        "generation_requests_by_key_slot": [
            previous + current
            for previous, current in zip(previous_generation_requests, generator.request_counts)
        ],
        "evaluation_requests_by_key_slot": [
            previous + current
            for previous, current in zip(previous_evaluation_requests, evaluator.request_counts)
        ],
        "answer_tokens": sum(row.get("total_tokens", 0) for row in final_answers.values()),
        "evaluation_tokens": sum(row.get("total_tokens", 0) for row in final_evaluations.values()),
        "answer_cache_file": "answers_v4.jsonl",
        "evaluation_cache_file": "semantic_evaluations_v4.jsonl",
        "final_answers_file": "answers_final_v4.jsonl",
        "final_evaluations_file": "semantic_evaluations_final_v4.jsonl",
        "summary": summaries,
    }
    _write_json(report_path, report)
    _write_json(output_dir / "summary_v4.json", summaries)
    print(json.dumps({key: report[key] for key in report if key != "summary"}, ensure_ascii=False, indent=2))

    if generation_failures or evaluation_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
