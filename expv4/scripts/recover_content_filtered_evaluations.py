from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPV4 = ROOT / "expv4"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPV4))
sys.path.insert(0, str(SCRIPT_DIR))

from recover_content_filtered_answers import (
    CONTEXT_PATH,
    evidence_objects,
    neutralize,
    ranked_evidence,
)
from run_final_generation_experiments import normalized_contexts
from run_generation_pilot import _load_jsonl
from src.config import ExperimentConfig
from src.evaluation import SemanticEvaluator
from src.evaluation.answer_metrics import answer_exact_match, answer_f1


OUTPUT_DIR = EXPV4 / "results" / "final_2000_v4" / "generation"
ANSWERS_PATH = OUTPUT_DIR / "answers_cache_v4.jsonl"
EVALUATIONS_PATH = OUTPUT_DIR / "semantic_evaluations_cache_v4.jsonl"


def append_row(row: dict[str, Any]) -> None:
    with EVALUATIONS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_with_retry(
    evaluator: SemanticEvaluator,
    question: str,
    reference_answer: str,
    generated_answer: str,
    evidence: list[Any],
) -> tuple[dict[str, Any], int]:
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            return (
                evaluator.evaluate(
                    question,
                    reference_answer,
                    generated_answer,
                    evidence,
                ),
                attempt,
            )
        except Exception as error:
            last_error = error
            message = f"{type(error).__name__}: {error}"
            if "contentFilter" in message or "Error code: 400" in message:
                raise
            if attempt < 4:
                time.sleep(float(attempt * 3))
    assert last_error is not None
    raise last_error


def recover_one(
    evaluator: SemanticEvaluator,
    answer_row: dict[str, Any],
    context_row: dict[str, Any],
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
    ranked = ranked_evidence(context_row)
    original_count = len(ranked)
    sizes = []
    for size in (12, 8, 6, 4, 3, 2, 1):
        actual = min(size, original_count)
        if actual and actual not in sizes:
            sizes.append(actual)

    total_attempts = 0
    last_error = "content_filter_fallback_exhausted"
    stages = [(False, size) for size in sizes] + [(True, size) for size in sizes]
    for use_neutral_text, size in stages:
        question = answer_row["question"]
        reference_answer = answer_row["gold_answer"]
        generated_answer = answer_row["answer"]
        if use_neutral_text:
            question = neutralize(question)
            reference_answer = neutralize(reference_answer)
            generated_answer = neutralize(generated_answer)
        try:
            result, attempts = call_with_retry(
                evaluator,
                question,
                reference_answer,
                generated_answer,
                evidence_objects(ranked[:size], use_neutral_text),
            )
            total_attempts += attempts
            return {
                **base,
                **result,
                "status": "success",
                "attempts": total_attempts,
                "latency_seconds": time.perf_counter() - started,
                "error": "",
                "provider_filter_fallback": True,
                "fallback_strategy": (
                    "neutral_rephrasing_and_relevance_reduction"
                    if use_neutral_text
                    else "relevance_reduction"
                ),
                "fallback_evidence_units": size,
                "original_evidence_units": original_count,
                "thinking_mode": "disabled",
            }
        except Exception as error:
            total_attempts += 1
            last_error = f"{type(error).__name__}: {error}"

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
        "attempts": total_attempts,
        "latency_seconds": time.perf_counter() - started,
        "error": last_error,
        "provider_filter_fallback": True,
        "fallback_strategy": "exhausted",
        "fallback_evidence_units": 0,
        "original_evidence_units": original_count,
        "thinking_mode": "disabled",
    }


def main() -> None:
    config = ExperimentConfig.load(EXPV4 / "configs" / "proposed.json")
    if len(config.generation_api_keys) != 5:
        raise RuntimeError("Five BIGMOD API keys are required")

    contexts = normalized_contexts(CONTEXT_PATH)
    answer_rows = _load_jsonl(ANSWERS_PATH)
    answers = {
        (row["id"], row["method"]): row
        for row in answer_rows
        if row.get("status") == "success" and row.get("answer")
    }
    evaluation_rows = _load_jsonl(EVALUATIONS_PATH)
    completed = {
        (row["id"], row["method"])
        for row in evaluation_rows
        if row.get("status") == "success"
    }
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in evaluation_rows:
        latest[(row["id"], row["method"])] = row
    missing = [key for key in latest if key not in completed]
    if not missing:
        print(json.dumps({"recovered": 0, "remaining": 0}))
        return
    if any("contentFilter" not in latest[key].get("error", "") for key in missing):
        raise RuntimeError("Non-content-filter failures remain; run standard recovery first")

    evaluator = SemanticEvaluator(
        model=config.generation_model,
        api_keys=config.generation_api_keys,
        base_url=config.generation_base_url,
    )
    recovered = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(recover_one, evaluator, answers[key], contexts[key]): key
            for key in missing
        }
        for future in as_completed(futures):
            row = future.result()
            append_row(row)
            recovered += int(row["status"] == "success")
            print(
                json.dumps(
                    {
                        "successful": recovered,
                        "id": row["id"],
                        "method": row["method"],
                        "status": row["status"],
                        "strategy": row["fallback_strategy"],
                    },
                    ensure_ascii=False,
                )
            )

    all_rows = _load_jsonl(EVALUATIONS_PATH)
    final_completed = {
        (row["id"], row["method"])
        for row in all_rows
        if row.get("status") == "success"
    }
    print(
        json.dumps(
            {
                "recovered": recovered,
                "unique_success": len(final_completed),
                "remaining": 28000 - len(final_completed),
                "model": config.generation_model,
                "thinking_mode": "disabled",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
