from __future__ import annotations

import re
import string
from collections import Counter
from statistics import mean
from typing import Iterable


def normalize_answer(text: str) -> str:
    value = str(text or "").lower()
    value = "".join(
        character for character in value if character not in set(string.punctuation)
    )
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_scores(prediction: str, gold: str) -> dict[str, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    exact_match = float(normalized_prediction == normalized_gold)
    if (
        normalized_prediction in {"yes", "no", "noanswer"}
        and normalized_prediction != normalized_gold
    ):
        return {
            "answer_em": exact_match,
            "answer_precision": 0.0,
            "answer_recall": 0.0,
            "answer_f1": 0.0,
        }
    if (
        normalized_gold in {"yes", "no", "noanswer"}
        and normalized_prediction != normalized_gold
    ):
        return {
            "answer_em": exact_match,
            "answer_precision": 0.0,
            "answer_recall": 0.0,
            "answer_f1": 0.0,
        }

    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    precision = same / len(prediction_tokens) if prediction_tokens else 0.0
    recall = same / len(gold_tokens) if gold_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "answer_em": exact_match,
        "answer_precision": precision,
        "answer_recall": recall,
        "answer_f1": f1,
    }


def supporting_fact_scores(
    prediction: Iterable[tuple[str, int]],
    gold: Iterable[tuple[str, int]],
) -> dict[str, float]:
    predicted_set = set(prediction)
    gold_set = set(gold)
    true_positive = len(predicted_set & gold_set)
    precision = true_positive / len(predicted_set) if predicted_set else 0.0
    recall = true_positive / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "support_em": float(predicted_set == gold_set),
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
    }


def joint_scores(
    answer: dict[str, float], support: dict[str, float]
) -> dict[str, float]:
    precision = answer["answer_precision"] * support["support_precision"]
    recall = answer["answer_recall"] * support["support_recall"]
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "joint_em": answer["answer_em"] * support["support_em"],
        "joint_precision": precision,
        "joint_recall": recall,
        "joint_f1": f1,
    }


def recovery_scores(
    initial_facts: Iterable[tuple[str, int]],
    final_context_facts: Iterable[tuple[str, int]],
    gold_facts: Iterable[tuple[str, int]],
) -> dict[str, float | None]:
    initial = set(initial_facts)
    final = set(final_context_facts)
    gold = set(gold_facts)
    initial_documents = {title for title, _ in initial}
    final_documents = {title for title, _ in final}
    gold_documents = {title for title, _ in gold}

    missing_facts = gold - initial
    missing_documents = gold_documents - initial_documents
    msfr = len(missing_facts & final) / len(missing_facts) if missing_facts else None
    msdr = (
        len(missing_documents & final_documents) / len(missing_documents)
        if missing_documents
        else None
    )
    return {
        "msfr": msfr,
        "msdr": msdr,
        "ccr": float(gold <= final),
        "dccr": float(gold_documents <= final_documents),
        "initial_sentence_chain_complete": float(gold <= initial),
        "initial_document_chain_complete": float(gold_documents <= initial_documents),
    }


def evidence_coverage_scores(
    evidence_facts: Iterable[tuple[str, int]],
    gold_facts: Iterable[tuple[str, int]],
    prefix: str,
) -> dict[str, float]:
    evidence = set(evidence_facts)
    gold = set(gold_facts)
    evidence_documents = {title for title, _ in evidence}
    gold_documents = {title for title, _ in gold}
    return {
        f"{prefix}_support_recall": len(evidence & gold) / len(gold) if gold else 0.0,
        f"{prefix}_ccr": float(gold <= evidence),
        f"{prefix}_dccr": float(gold_documents <= evidence_documents),
    }


def evaluate_prediction(
    prediction_answer: str,
    prediction_support: Iterable[tuple[str, int]],
    initial_facts: Iterable[tuple[str, int]],
    final_context_facts: Iterable[tuple[str, int]],
    gold_answer: str,
    gold_facts: Iterable[tuple[str, int]],
) -> dict[str, float | None]:
    answer = answer_scores(prediction_answer, gold_answer)
    support = supporting_fact_scores(prediction_support, gold_facts)
    return {
        **answer,
        **support,
        **joint_scores(answer, support),
        **recovery_scores(initial_facts, final_context_facts, gold_facts),
    }


def aggregate_metrics(rows: list[dict[str, float | None]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = set().union(*(row.keys() for row in rows))
    aggregated: dict[str, float] = {}
    for key in sorted(keys):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            aggregated[key] = mean(values)
    return aggregated
