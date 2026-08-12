from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPV4 = ROOT / "expv4"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPV4))
sys.path.insert(0, str(SCRIPT_DIR))

from run_generation_pilot import _evaluate_one, _generate_one
from recover_content_filtered_answers import recover_one as recover_filtered_answer
from recover_content_filtered_evaluations import recover_one as recover_filtered_evaluation
from src.config import ExperimentConfig
from src.evaluation import SemanticEvaluator
from src.generation.answer_generator import AnswerGenerator


METHODS = {
    "hybrid_rerank": "Hybrid + Rerank（混合检索加重排）",
    "graphrag": "GraphRAG（图检索增强生成）",
    "kg2rag": "KG²RAG / KG-RAG（知识图谱检索增强生成）",
    "macrag": "MacRAG（多尺度自适应检索增强生成）",
    "hipporag2": "HippoRAG 2（图记忆检索增强生成）",
    "ours": "Ours（本文方法）",
}
METRICS = (
    "answer_em",
    "answer_f1",
    "accuracy",
    "faithfulness",
    "answer_relevance",
    "context_relevance",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified-resource external generation comparison")
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=4)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def evidence_signature(row: dict[str, Any]) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (str(unit["title"]), int(unit["sent_id"]), str(unit["text"]))
        for unit in row["context_evidence"]
    )


def normalized_method(row: dict[str, Any]) -> str:
    return str(row.get("method") or row.get("strategy") or row.get("variant"))


def clone_answer(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "method": context["method"],
        "label": context["label"],
        "question_type": context["question_type"],
        "reused_from_method": row["method"],
        "resource_reuse": "identical_question_and_evidence",
    }


def clone_evaluation(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "method": context["method"],
        "label": context["label"],
        "question_type": context["question_type"],
        "reused_from_method": row["method"],
        "resource_reuse": "identical_question_answer_and_evidence",
    }


def run_parallel(
    tasks: list[Any],
    worker: Callable[[Any], dict[str, Any]],
    output: Path,
    workers: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            append_jsonl(output, row)
            rows.append(row)
    return rows


def should_stop(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    failures = [row for row in rows if row["status"] != "success"]
    unexpected = [
        row for row in failures
        if not any(
            text in row.get("error", "")
            for text in (
                "RateLimit", "contentFilter", "APIConnectionError",
                "APITimeoutError", "Request timed out", "Error code: 429",
            )
        )
    ]
    return len(failures) / len(rows) > 0.50 or len(unexpected) / len(rows) > 0.01


def recoverable_only(rows: list[dict[str, Any]]) -> bool:
    failures = [row for row in rows if row.get("status") != "success"]
    return bool(failures) and all(
        any(
            text in row.get("error", "")
            for text in (
                "RateLimit", "APIConnectionError", "APITimeoutError",
                "Request timed out", "Error code: 429",
            )
        )
        for row in failures
    )


def mean_ci(values: list[float]) -> tuple[float, float]:
    average = mean(values)
    if len(values) < 2:
        return average, 0.0
    return average, 1.96 * stdev(values) / math.sqrt(len(values))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "examples": len(rows),
        "refusal_rate": mean(float(row["is_refusal"]) for row in rows),
    }
    _, result["refusal_rate_ci95"] = mean_ci([float(row["is_refusal"]) for row in rows])
    for metric in METRICS:
        result[metric], result[f"{metric}_ci95"] = mean_ci([float(row[metric]) for row in rows])
    return result


def main() -> None:
    args = arguments()
    config = ExperimentConfig.load(EXPV4 / "configs" / "proposed.json")
    if len(config.generation_api_keys) != 5:
        raise RuntimeError(f"Expected five GLM API keys, found {len(config.generation_api_keys)}")

    output_dir = EXPV4 / "results" / "final_2000_v4" / "unified_external"
    contexts_path = output_dir / "retrieval_contexts_unified_external_v4.jsonl"
    answers_path = output_dir / "answers_unified_external_v4.jsonl"
    evaluations_path = output_dir / "semantic_evaluations_unified_external_v4.jsonl"
    progress_path = output_dir / "generation_progress_v4.json"
    contexts = {(row["id"], row["method"]): row for row in load_jsonl(contexts_path)}
    expected = 2000 * len(METHODS)
    if len(contexts) != expected:
        raise RuntimeError(f"Expected {expected} contexts, found {len(contexts)}")

    gold_frame = pd.read_parquet(config.validation_file, columns=["id", "answer"])
    gold_answers = {str(row["id"]): str(row["answer"]) for _, row in gold_frame.iterrows()}

    old_context_path = EXPV4 / "results" / "final_2000_v4" / "retrieval" / "retrieval_contexts_14_final_variants_v4.jsonl"
    old_contexts: dict[tuple[str, tuple], list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(old_context_path):
        row["method"] = normalized_method(row)
        old_contexts[(str(row["id"]), evidence_signature(row))].append(row)
    old_answers = {
        (str(row["id"]), str(row["method"])): row
        for row in load_jsonl(EXPV4 / "results" / "final_2000_v4" / "generation" / "answers_final_v4.jsonl")
    }
    old_evaluations = {
        (str(row["id"]), str(row["method"])): row
        for row in load_jsonl(EXPV4 / "results" / "final_2000_v4" / "generation" / "semantic_evaluations_final_v4.jsonl")
    }

    completed_answers = {
        (str(row["id"]), str(row["method"])): row
        for row in load_jsonl(answers_path)
        if row.get("status") == "success" and row.get("answer")
    }
    reused_answers = sum(
        row.get("resource_reuse") == "identical_question_and_evidence"
        for row in completed_answers.values()
    )
    for key, context in contexts.items():
        if key in completed_answers:
            continue
        candidates = old_contexts.get((str(context["id"]), evidence_signature(context)), [])
        source = next(
            (old_answers.get((str(context["id"]), candidate["method"])) for candidate in candidates
             if old_answers.get((str(context["id"]), candidate["method"]), {}).get("status") == "success"),
            None,
        )
        if source is not None:
            cloned = clone_answer(source, context)
            append_jsonl(answers_path, cloned)
            completed_answers[key] = cloned
            reused_answers += 1

    generator = AnswerGenerator(
        model=config.generation_model,
        api_keys=config.generation_api_keys,
        base_url=config.generation_base_url,
        temperature=0.0,
        max_tokens=config.generation_max_tokens,
    )
    generation_round = 0
    while len(completed_answers) < expected:
        generation_round += 1
        pending = [context for key, context in contexts.items() if key not in completed_answers]
        batch = pending[:args.batch_size]
        rows = run_parallel(
            batch,
            lambda row: _generate_one(generator, row, gold_answers[str(row["id"])], args.max_attempts),
            answers_path,
            args.workers,
        )
        filtered = [row for row in rows if "contentFilter" in row.get("error", "")]
        if filtered:
            filtered_keys = {(str(row["id"]), str(row["method"])) for row in filtered}
            fallback_rows = run_parallel(
                [contexts[key] for key in filtered_keys],
                lambda row: recover_filtered_answer(
                    generator, row, gold_answers[str(row["id"])]
                ),
                answers_path,
                min(args.workers, 5),
            )
            rows = [
                row for row in rows
                if (str(row["id"]), str(row["method"])) not in filtered_keys
            ] + fallback_rows
        for row in rows:
            if row.get("status") == "success" and row.get("answer"):
                completed_answers[(str(row["id"]), str(row["method"]))] = row
        progress = {
            "phase": "generation",
            "expected": expected,
            "completed": len(completed_answers),
            "remaining": expected - len(completed_answers),
            "reused": reused_answers,
            "new_api_requests_by_key_slot": generator.request_counts,
            "model": config.generation_model,
            "thinking_mode": "disabled",
            "workers": args.workers,
            "retry_round": generation_round,
        }
        write_json(progress_path, progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
        if should_stop(rows) and not recoverable_only(rows):
            raise RuntimeError("Generation failures exceeded the stop rule")
        if any(row["status"] != "success" for row in rows):
            time.sleep(60)

    answer_rows = load_jsonl(answers_path)
    completed_answers = {
        (str(row["id"]), str(row["method"])): row
        for row in answer_rows
        if row.get("status") == "success" and row.get("answer")
    }
    if len(completed_answers) != expected:
        raise RuntimeError(f"Missing {expected - len(completed_answers)} successful answers")

    completed_evaluations = {
        (str(row["id"]), str(row["method"])): row
        for row in load_jsonl(evaluations_path)
        if row.get("status") == "success"
    }
    reused_evaluations = sum(
        row.get("resource_reuse") == "identical_question_answer_and_evidence"
        for row in completed_evaluations.values()
    )
    for key, context in contexts.items():
        if key in completed_evaluations:
            continue
        answer = completed_answers[key]
        reused_from = answer.get("reused_from_method")
        source = old_evaluations.get((str(context["id"]), str(reused_from))) if reused_from else None
        old_answer = old_answers.get((str(context["id"]), str(reused_from))) if reused_from else None
        if source is not None and old_answer is not None and old_answer.get("answer") == answer.get("answer"):
            cloned = clone_evaluation(source, context)
            append_jsonl(evaluations_path, cloned)
            completed_evaluations[key] = cloned
            reused_evaluations += 1

    evaluator = SemanticEvaluator(config.generation_model, config.generation_api_keys, config.generation_base_url)
    evaluation_round = 0
    while len(completed_evaluations) < expected:
        evaluation_round += 1
        pending_eval = [(key, completed_answers[key]) for key in contexts if key not in completed_evaluations]
        batch = pending_eval[:args.batch_size]
        rows = run_parallel(
            batch,
            lambda task: _evaluate_one(evaluator, task[1], contexts[task[0]], args.max_attempts),
            evaluations_path,
            args.workers,
        )
        filtered = [row for row in rows if "contentFilter" in row.get("error", "")]
        if filtered:
            filtered_keys = {(str(row["id"]), str(row["method"])) for row in filtered}
            fallback_rows = run_parallel(
                [key for key in filtered_keys],
                lambda key: recover_filtered_evaluation(
                    evaluator, completed_answers[key], contexts[key]
                ),
                evaluations_path,
                min(args.workers, 5),
            )
            rows = [
                row for row in rows
                if (str(row["id"]), str(row["method"])) not in filtered_keys
            ] + fallback_rows
        for row in rows:
            if row.get("status") == "success":
                completed_evaluations[(str(row["id"]), str(row["method"]))] = row
        progress = {
            "phase": "evaluation",
            "expected": expected,
            "completed": len(completed_evaluations),
            "remaining": expected - len(completed_evaluations),
            "reused": reused_evaluations,
            "new_generation_requests_by_key_slot": generator.request_counts,
            "new_evaluation_requests_by_key_slot": evaluator.request_counts,
            "model": config.generation_model,
            "thinking_mode": "disabled",
            "workers": args.workers,
            "retry_round": evaluation_round,
        }
        write_json(progress_path, progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
        if should_stop(rows) and not recoverable_only(rows):
            raise RuntimeError("Evaluation failures exceeded the stop rule")
        if any(row["status"] != "success" for row in rows):
            time.sleep(60)

    evaluation_rows = load_jsonl(evaluations_path)
    final_evaluations = {
        (str(row["id"]), str(row["method"])): row
        for row in evaluation_rows
        if row.get("status") == "success"
    }
    if len(final_evaluations) != expected:
        raise RuntimeError(f"Missing {expected - len(final_evaluations)} successful evaluations")
    counts = Counter(method for _, method in final_evaluations)
    if counts != Counter({method: 2000 for method in METHODS}):
        raise RuntimeError(f"Invalid evaluation distribution: {dict(counts)}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (_, method), row in final_evaluations.items():
        grouped[method].append(row)
    table = [{"method": method, "label": label, **aggregate(grouped[method])} for method, label in METHODS.items()]
    payload = {
        "dataset": "HotpotQA v4 Final 2000（桥接型和比较型各 1,000 条）",
        "comparison_protocol": "Unified-resource reproduction（统一资源复现）",
        "generation_model": config.generation_model,
        "thinking_mode": "disabled",
        "temperature": 0.0,
        "evaluation_protocol": evaluator.protocol_version,
        "reused_identical_answers": reused_answers,
        "reused_identical_evaluations": reused_evaluations,
        "new_generation_requests": sum(generator.request_counts),
        "new_evaluation_requests": sum(evaluator.request_counts),
        "table": table,
    }
    json_path = output_dir / "table3_unified_external_generation_v4.json"
    csv_path = output_dir / "table3_unified_external_generation_v4.csv"
    write_json(json_path, payload)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(progress_path, {"phase": "complete", "expected": expected, "completed": expected, **{k: payload[k] for k in ("reused_identical_answers", "reused_identical_evaluations", "new_generation_requests", "new_evaluation_requests")}})
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
