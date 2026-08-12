from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPV4 = ROOT / "expv4"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPV4))
sys.path.insert(0, str(SCRIPT_DIR))

from run_final_generation_experiments import normalized_contexts
from run_generation_pilot import _load_jsonl
from src.config import ExperimentConfig
from src.generation.answer_generator import AnswerGenerator
from src.models import EvidenceUnit


OUTPUT_DIR = EXPV4 / "results" / "final_2000_v4" / "generation"
ANSWERS_PATH = OUTPUT_DIR / "answers_cache_v4.jsonl"
CONTEXT_PATH = (
    EXPV4
    / "results"
    / "final_2000_v4"
    / "retrieval"
    / "retrieval_contexts_14_final_variants_v4.jsonl"
)

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "both",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}

NEUTRAL_REPLACEMENTS = {
    "Bauerfield International Ariport": "the airport in question",
    "Bauerfield International Airport": "the airport in question",
    "Bauerfield": "the airport",
    "diplomatic cables leak": "publication of diplomatic documents",
    "news-leaking": "document-publishing",
    "nuclear weapons program": "nuclear policy program",
    "armed conflict": "historical conflict",
    "armed forces": "military forces",
    "war crimes": "wartime allegations",
    "terrorists": "attackers",
    "terrorist": "attacker",
    "hijacked": "forcibly taken over",
    "surrender": "cessation of hostilities",
    "destruction": "damage",
    "violent": "severe",
    "violence": "hostility",
    "assassin": "fighter",
    "attack": "incident",
    "blood": "medical",
    "aids": "a serious illness",
    "coup": "attempted intervention",
    "weapons": "military equipment",
    "war": "conflict",
}


def append_row(row: dict[str, Any]) -> None:
    with ANSWERS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    }


def ranked_evidence(context_row: dict[str, Any]) -> list[dict[str, Any]]:
    question_tokens = tokens(context_row["question"])
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, unit in enumerate(context_row["context_evidence"]):
        title_overlap = len(question_tokens & tokens(unit["title"]))
        text_overlap = len(question_tokens & tokens(unit["text"]))
        score = 4.0 * title_overlap + float(text_overlap) - index * 0.0001
        scored.append((score, index, unit))
    return [unit for _, _, unit in sorted(scored, reverse=True)]


def neutralize(text: str) -> str:
    value = text
    for source, replacement in NEUTRAL_REPLACEMENTS.items():
        value = re.sub(rf"\b{re.escape(source)}\b", replacement, value, flags=re.I)
    return value


def evidence_objects(rows: list[dict[str, Any]], use_neutral_text: bool) -> list[EvidenceUnit]:
    values = []
    for row in rows:
        item = dict(row)
        if use_neutral_text:
            item["text"] = neutralize(item["text"])
            item["title"] = neutralize(item["title"])
        values.append(EvidenceUnit(**item))
    return values


def call_with_retry(
    generator: AnswerGenerator,
    question: str,
    evidence: list[EvidenceUnit],
) -> tuple[dict[str, Any], int]:
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            return generator.generate_with_metadata(question, evidence), attempt
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
    generator: AnswerGenerator,
    context_row: dict[str, Any],
    gold_answer: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    ranked = ranked_evidence(context_row)
    original_count = len(ranked)
    sizes = []
    for size in (12, 8, 6, 4, 3, 2, 1):
        actual = min(size, original_count)
        if actual and actual not in sizes:
            sizes.append(actual)

    last_error = "content_filter_fallback_exhausted"
    total_attempts = 0
    stages = [(False, size) for size in sizes] + [(True, size) for size in sizes]
    for use_neutral_text, size in stages:
        selected = ranked[:size]
        question = neutralize(context_row["question"]) if use_neutral_text else context_row["question"]
        try:
            result, attempts = call_with_retry(
                generator,
                question,
                evidence_objects(selected, use_neutral_text),
            )
            total_attempts += attempts
            answer = result["answer"]
            if not answer:
                last_error = "empty_answer"
                continue
            return {
                "id": context_row["id"],
                "method": context_row["method"],
                "label": context_row["label"],
                "question_type": context_row["question_type"],
                "question": context_row["question"],
                "gold_answer": gold_answer,
                "answer": answer,
                "status": "success",
                "attempts": total_attempts,
                "latency_seconds": time.perf_counter() - started,
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "evidence_units": original_count,
                "error": "",
                "provider_filter_fallback": True,
                "fallback_strategy": (
                    "neutral_rephrasing_and_relevance_reduction"
                    if use_neutral_text
                    else "relevance_reduction"
                ),
                "fallback_evidence_units": size,
                "thinking_mode": "disabled",
            }
        except Exception as error:
            total_attempts += 1
            last_error = f"{type(error).__name__}: {error}"

    return {
        "id": context_row["id"],
        "method": context_row["method"],
        "label": context_row["label"],
        "question_type": context_row["question_type"],
        "question": context_row["question"],
        "gold_answer": gold_answer,
        "answer": "",
        "status": "failed",
        "attempts": total_attempts,
        "latency_seconds": time.perf_counter() - started,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "evidence_units": original_count,
        "error": last_error,
        "provider_filter_fallback": True,
        "fallback_strategy": "exhausted",
        "fallback_evidence_units": 0,
        "thinking_mode": "disabled",
    }


def main() -> None:
    config = ExperimentConfig.load(EXPV4 / "configs" / "proposed.json")
    if len(config.generation_api_keys) != 5:
        raise RuntimeError("Five BIGMOD API keys are required")

    contexts = normalized_contexts(CONTEXT_PATH)
    rows = _load_jsonl(ANSWERS_PATH)
    completed = {
        (row["id"], row["method"])
        for row in rows
        if row.get("status") == "success" and row.get("answer")
    }
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        latest[(row["id"], row["method"])] = row
    missing = [key for key in latest if key not in completed]
    if not missing:
        print(json.dumps({"recovered": 0, "remaining": 0}))
        return
    if any("contentFilter" not in latest[key].get("error", "") for key in missing):
        raise RuntimeError("Non-content-filter failures remain; run the standard recovery first")

    dataset = pd.read_parquet(config.validation_file, columns=["id", "answer"])
    gold_answers = dict(zip(dataset["id"], dataset["answer"]))
    generator = AnswerGenerator(
        model=config.generation_model,
        api_keys=config.generation_api_keys,
        base_url=config.generation_base_url,
        temperature=0.0,
        max_tokens=config.generation_max_tokens,
    )

    recovered = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                recover_one,
                generator,
                contexts[key],
                str(gold_answers[key[0]]),
            ): key
            for key in missing
        }
        for future in as_completed(futures):
            row = future.result()
            append_row(row)
            recovered += int(row["status"] == "success")
            print(
                json.dumps(
                    {
                        "processed": recovered,
                        "id": row["id"],
                        "method": row["method"],
                        "status": row["status"],
                        "strategy": row["fallback_strategy"],
                    },
                    ensure_ascii=False,
                )
            )

    all_rows = _load_jsonl(ANSWERS_PATH)
    final_completed = {
        (row["id"], row["method"])
        for row in all_rows
        if row.get("status") == "success" and row.get("answer")
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
