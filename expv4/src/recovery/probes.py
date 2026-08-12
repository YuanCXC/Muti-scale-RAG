from __future__ import annotations

from collections import Counter

import numpy as np

from ..config import ExperimentConfig
from ..indexing.sentence_index import SentenceIndex
from ..models import BridgeLink, EvidenceUnit, ProbeCandidate
from ..text_utils import anchor_score, normalize_title, terms
from .gating import GainCalculator


class ProbeEngine:
    def __init__(
        self,
        config: ExperimentConfig,
        sentence_index: SentenceIndex,
        bridge_index: dict[str, list[BridgeLink]],
    ) -> None:
        self.config = config
        self.sentence_index = sentence_index
        self.bridge_index = bridge_index
        self.gain_calculator = GainCalculator(config)

    def _base_vectors(self, units: list[EvidenceUnit]) -> list[np.ndarray]:
        return [self.sentence_index.vector(unit.vector_id) for unit in units]

    def structural(
        self,
        initial: list[EvidenceUnit],
        query_vector: np.ndarray,
    ) -> list[ProbeCandidate]:
        base_vectors = self._base_vectors(initial)
        existing = {unit.key for unit in initial}
        candidates: dict[tuple[str, int], ProbeCandidate] = {}

        for seed in initial:
            for record in self.sentence_index.title_sentences(seed.title):
                distance = abs(record.sent_id - seed.sent_id)
                if distance == 0 or distance > self.config.structural_window:
                    continue
                evidence = EvidenceUnit(
                    title=record.title,
                    sent_id=record.sent_id,
                    text=record.text,
                    score=seed.score,
                    source="structural_probe",
                    vector_id=record.vector_id,
                    metadata={"anchor_sent_id": seed.sent_id},
                )
                if evidence.key in existing:
                    continue
                prior = anchor_score(
                    seed.sent_id, record.sent_id, self.config.anchor_sigma
                )
                probe = self.gain_calculator.score(
                    "structural",
                    evidence,
                    self.sentence_index.vector(record.vector_id),
                    query_vector,
                    base_vectors,
                    prior,
                )
                previous = candidates.get(evidence.key)
                if previous is None or probe.gain > previous.gain:
                    candidates[evidence.key] = probe

        return sorted(candidates.values(), key=lambda item: item.gain, reverse=True)[
            : self.config.max_structural_probes
        ]

    def bridge(
        self,
        seeds: list[EvidenceUnit],
        base_evidence: list[EvidenceUnit],
        query: str,
        query_vector: np.ndarray,
        hop: int,
    ) -> list[ProbeCandidate]:
        base_vectors = self._base_vectors(base_evidence)
        base_keys = {unit.key for unit in base_evidence}
        ranked_seeds: list[EvidenceUnit] = []
        seeds_per_title: Counter[str] = Counter()
        for seed in sorted(seeds, key=lambda unit: unit.score, reverse=True):
            if seeds_per_title[seed.title] >= 2:
                continue
            ranked_seeds.append(seed)
            seeds_per_title[seed.title] += 1
            if len(ranked_seeds) >= self.config.max_bridge_seeds:
                break
        candidates: dict[tuple[str, int], ProbeCandidate] = {}

        for seed in ranked_seeds:
            links = self.bridge_index.get(seed.title, [])
            ranked_links = sorted(
                links,
                key=lambda link: link.bridge_prior
                * anchor_score(
                    seed.sent_id, link.source_sentence_id, self.config.anchor_sigma
                ),
                reverse=True,
            )[: self.config.max_bridge_links_per_seed]

            for link in ranked_links:
                target = self._best_target_sentence(
                    query, query_vector, link, base_keys
                )
                if target is None:
                    continue
                evidence, target_score = target
                evidence.score = target_score
                evidence.source = "bridge_probe"
                evidence.metadata.update(
                    {
                        "source_title": seed.title,
                        "source_sent_id": seed.sent_id,
                        "bridge_entity": link.bridge_entity,
                        "predicate": link.predicate,
                        "bridge_prior": link.bridge_prior,
                        "evidence_hop": hop,
                    }
                )
                prior = link.bridge_prior * anchor_score(
                    seed.sent_id,
                    link.source_sentence_id,
                    self.config.anchor_sigma,
                )
                probe = self.gain_calculator.score(
                    "bridge",
                    evidence,
                    self.sentence_index.vector(evidence.vector_id),
                    query_vector,
                    base_vectors,
                    prior,
                )
                probe.bridge_link = link
                previous = candidates.get(evidence.key)
                if previous is None or probe.gain > previous.gain:
                    candidates[evidence.key] = probe

        return sorted(candidates.values(), key=lambda item: item.gain, reverse=True)

    def _best_target_sentence(
        self,
        query: str,
        query_vector: np.ndarray,
        link: BridgeLink,
        excluded_keys: set[tuple[str, int]],
    ) -> tuple[EvidenceUnit, float] | None:
        best: tuple[EvidenceUnit, float] | None = None
        query_predicate_terms = terms(f"{query} {link.predicate}")
        for record in self.sentence_index.title_sentences(link.target_title):
            if (record.title, record.sent_id) in excluded_keys:
                continue
            vector = self.sentence_index.vector(record.vector_id)
            relevance = max(0.0, float(np.dot(query_vector, vector)))
            sentence_normalized = normalize_title(record.text)
            bridge_consistency = float(
                normalize_title(link.bridge_entity) in sentence_normalized
                or normalize_title(link.target_title) in sentence_normalized
            )
            sentence_terms = terms(record.text)
            relation_alignment = len(query_predicate_terms & sentence_terms) / max(
                len(query_predicate_terms), 1
            )
            score = (
                0.60 * relevance + 0.25 * bridge_consistency + 0.15 * relation_alignment
            )
            evidence = EvidenceUnit(
                title=record.title,
                sent_id=record.sent_id,
                text=record.text,
                score=score,
                source="bridge_probe",
                vector_id=record.vector_id,
            )
            if best is None or score > best[1]:
                best = evidence, score
        return best
