from __future__ import annotations

import numpy as np

from ..config import ExperimentConfig
from ..indexing.sentence_index import SentenceIndex
from ..models import EvidenceUnit


def deduplicate(units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    unique: dict[tuple[str, int], EvidenceUnit] = {}
    for unit in units:
        current = unique.get(unit.key)
        if current is None or unit.score > current.score:
            unique[unit.key] = unit
    return list(unique.values())


class ContextSelector:
    def __init__(self, config: ExperimentConfig, sentence_index: SentenceIndex) -> None:
        self.config = config
        self.sentence_index = sentence_index

    def select(
        self,
        candidates: list[EvidenceUnit],
        query_vector: np.ndarray,
    ) -> list[EvidenceUnit]:
        remaining = deduplicate(candidates)
        selected: list[EvidenceUnit] = []
        selected_vectors: list[np.ndarray] = []
        used_tokens = 0

        while remaining and len(selected) < self.config.max_context_units:
            best_index = -1
            best_value = -float("inf")
            for index, unit in enumerate(remaining):
                if (
                    selected
                    and used_tokens + unit.token_count > self.config.context_budget
                ):
                    continue
                vector = self.sentence_index.vector(unit.vector_id)
                relevance = max(0.0, float(np.dot(query_vector, vector)))
                redundancy = max(
                    (
                        float(np.dot(vector, selected_vector))
                        for selected_vector in selected_vectors
                    ),
                    default=0.0,
                )
                information_gain = relevance * (1.0 - max(0.0, redundancy))
                bridge_gain = float(unit.metadata.get("bridge_prior", 0.0))
                retrieval_score = min(max(unit.score, 0.0), 1.0)
                cost = unit.token_count / max(self.config.context_budget, 1)
                value = (
                    0.35 * retrieval_score
                    + 0.25 * relevance
                    + 0.20 * information_gain
                    + 0.20 * bridge_gain
                    - 0.10 * cost
                )
                if value > best_value:
                    best_value = value
                    best_index = index
            if best_index < 0:
                break
            unit = remaining.pop(best_index)
            unit.metadata["selection_value"] = best_value
            selected.append(unit)
            selected_vectors.append(self.sentence_index.vector(unit.vector_id))
            used_tokens += unit.token_count
        return selected


class SupportPredictor:
    def __init__(self, config: ExperimentConfig, sentence_index: SentenceIndex) -> None:
        self.config = config
        self.sentence_index = sentence_index

    def predict(
        self,
        context: list[EvidenceUnit],
        query_vector: np.ndarray,
    ) -> list[tuple[str, int]]:
        scored: list[tuple[float, EvidenceUnit]] = []
        for unit in context:
            vector = self.sentence_index.vector(unit.vector_id)
            relevance = max(0.0, float(np.dot(query_vector, vector)))
            recovery_score = float(unit.metadata.get("probe_gain", 0.0))
            selection_value = float(unit.metadata.get("selection_value", 0.0))
            score = 0.55 * relevance + 0.25 * selection_value + 0.20 * recovery_score
            scored.append((score, unit))
        scored.sort(key=lambda item: item[0], reverse=True)

        selected: list[tuple[float, EvidenceUnit]] = []
        used_titles: set[str] = set()
        for item in scored:
            if item[1].title in used_titles:
                continue
            selected.append(item)
            used_titles.add(item[1].title)
            if len(selected) >= self.config.support_min_facts:
                break
        for item in scored:
            if item in selected or len(selected) >= self.config.support_max_facts:
                continue
            cutoff = selected[-1][0] - self.config.support_margin if selected else 0.0
            if item[0] >= cutoff:
                selected.append(item)

        selected.sort(key=lambda item: item[0], reverse=True)
        return [unit.key for _, unit in selected[: self.config.support_max_facts]]
