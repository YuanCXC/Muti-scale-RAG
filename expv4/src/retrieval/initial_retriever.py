from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ..config import ExperimentConfig
from ..indexing.sentence_index import SentenceIndex
from ..models import EvidenceUnit
from ..text_utils import terms
from .embedding import EmbeddingClient
from .reranker import APIReranker


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class KeywordTitleIndex:
    def __init__(self, sentence_index: SentenceIndex) -> None:
        self.sentence_index = sentence_index
        self.postings: dict[str, set[int]] = defaultdict(set)
        for record in sentence_index.records:
            for token in terms(record.title) - STOPWORDS:
                self.postings[token].add(record.vector_id)

    def search(self, query: str, top_k: int) -> list[EvidenceUnit]:
        query_terms = terms(query) - STOPWORDS
        counts: Counter[int] = Counter()
        for token in query_terms:
            for vector_id in self.postings.get(token, ()):
                counts[vector_id] += 1
        units: list[EvidenceUnit] = []
        for vector_id, matched in counts.most_common(top_k):
            record = self.sentence_index.record(vector_id)
            units.append(
                EvidenceUnit(
                    title=record.title,
                    sent_id=record.sent_id,
                    text=record.text,
                    score=matched / max(len(query_terms), 1),
                    source="keyword",
                    vector_id=record.vector_id,
                )
            )
        return units


class InitialRetriever:
    def __init__(
        self,
        config: ExperimentConfig,
        sentence_index: SentenceIndex,
        embedder: EmbeddingClient,
    ) -> None:
        self.config = config
        self.sentence_index = sentence_index
        self.embedder = embedder
        self.keyword_index = KeywordTitleIndex(sentence_index)
        self.reranker = None
        if config.rerank_enabled:
            self.reranker = APIReranker(
                config.rerank_model,
                config.rerank_api_key,
                config.rerank_base_url,
            )

    def retrieve(self, query: str) -> tuple[list[EvidenceUnit], np.ndarray]:
        query_vector = self.embedder.embed(query)[0]
        return self.retrieve_with_vector(query, query_vector)

    def retrieve_with_vector(
        self,
        query: str,
        query_vector: np.ndarray,
    ) -> tuple[list[EvidenceUnit], np.ndarray]:
        vector_units = self.sentence_index.search(
            query_vector, self.config.vector_top_k
        )
        keyword_units = self.keyword_index.search(query, self.config.keyword_top_k)

        merged: dict[tuple[str, int], EvidenceUnit] = {}
        for unit in vector_units + keyword_units:
            current = merged.get(unit.key)
            if current is None or unit.score > current.score:
                merged[unit.key] = unit
        candidates = list(merged.values())
        candidates.sort(key=lambda unit: unit.score, reverse=True)

        if self.reranker:
            try:
                candidates = self.reranker.rerank(
                    query, candidates, self.config.rerank_top_k
                )
            except Exception as exc:
                for unit in candidates:
                    unit.metadata["rerank_error"] = str(exc)
                candidates = candidates[: self.config.rerank_top_k]
        else:
            candidates = candidates[: self.config.rerank_top_k]
        return candidates, query_vector
