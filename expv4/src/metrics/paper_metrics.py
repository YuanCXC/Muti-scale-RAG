from __future__ import annotations

import math
from collections.abc import Iterable


def unique_titles(evidence_facts: Iterable[tuple[str, int]]) -> list[str]:
    titles = []
    seen = set()
    for title, _ in evidence_facts:
        title = str(title)
        if title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def title_ranking_scores(
    predicted_titles: Iterable[str],
    gold_titles: Iterable[str],
) -> dict[str, float]:
    ranked = []
    seen = set()
    for title in predicted_titles:
        title = str(title)
        if title not in seen:
            seen.add(title)
            ranked.append(title)
    gold = {str(title) for title in gold_titles}
    relevant = [float(title in gold) for title in ranked]
    hits = int(sum(relevant))
    precision = hits / len(ranked) if ranked else 0.0
    recall = hits / len(gold) if gold else 0.0

    first_relevant_rank = next(
        (rank for rank, value in enumerate(relevant, start=1) if value),
        None,
    )
    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    dcg = sum(
        value / math.log2(rank + 1)
        for rank, value in enumerate(relevant, start=1)
    )
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, len(gold) + 1)
    )
    ndcg = dcg / idcg if idcg else 0.0

    relevant_seen = 0
    precision_sum = 0.0
    for rank, value in enumerate(relevant, start=1):
        if value:
            relevant_seen += 1
            precision_sum += relevant_seen / rank
    average_precision = precision_sum / len(gold) if gold else 0.0
    return {
        "title_recall": recall,
        "title_precision": precision,
        "title_mrr": mrr,
        "title_ndcg": ndcg,
        "title_average_precision": average_precision,
    }
