from __future__ import annotations

import re
import string
from collections import Counter


def _normalize_answer(text: str) -> str:
    lowered = str(text or "").lower()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def answer_exact_match(prediction: str, reference: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(reference))


def answer_f1(prediction: str, reference: str) -> float:
    predicted_tokens = _normalize_answer(prediction).split()
    reference_tokens = _normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    overlap = Counter(predicted_tokens) & Counter(reference_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(predicted_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)
