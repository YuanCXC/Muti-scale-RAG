from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path

import faiss
import numpy as np

from ..config import ExperimentConfig
from ..indexing.bridge_index import load_bridge_index
from ..indexing.sentence_index import SentenceIndex
from ..models import EvidenceUnit, MethodResult
from .reranker import APIReranker


def unique_by_title(units: list[EvidenceUnit], limit: int | None = None) -> list[EvidenceUnit]:
    selected: list[EvidenceUnit] = []
    seen: set[str] = set()
    for unit in units:
        if unit.title in seen:
            continue
        selected.append(unit)
        seen.add(unit.title)
        if limit is not None and len(selected) >= limit:
            break
    return selected


class ParagraphIndex:
    def __init__(self, index_dir: Path) -> None:
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        self.internal_to_id = {
            int(key): value for key, value in metadata["internal_id_to_id"].items()
        }
        self.metadata = metadata["id_to_metadata"]

    def search(self, query_vector: np.ndarray, top_k: int) -> list[EvidenceUnit]:
        vector = query_vector.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        scores, ids = self.index.search(vector, top_k)
        units: list[EvidenceUnit] = []
        for score, internal_id in zip(scores[0], ids[0]):
            if internal_id < 0:
                continue
            item = self.metadata[self.internal_to_id[int(internal_id)]]
            title = str(item.get("extra", {}).get("title", item["doc_id"]))
            units.append(
                EvidenceUnit(
                    title=title,
                    sent_id=-1,
                    text=str(item["content"]),
                    score=float(score),
                    source="paragraph_vector",
                    vector_id=-1,
                )
            )
        return units

    def batch_search(
        self, query_vectors: np.ndarray, top_k: int
    ) -> list[list[EvidenceUnit]]:
        vectors = query_vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        scores, ids = self.index.search(vectors, top_k)
        batches: list[list[EvidenceUnit]] = []
        for row_scores, row_ids in zip(scores, ids):
            units: list[EvidenceUnit] = []
            for score, internal_id in zip(row_scores, row_ids):
                if internal_id < 0:
                    continue
                item = self.metadata[self.internal_to_id[int(internal_id)]]
                title = str(item.get("extra", {}).get("title", item["doc_id"]))
                units.append(
                    EvidenceUnit(
                        title=title,
                        sent_id=-1,
                        text=str(item["content"]),
                        score=float(score),
                        source="paragraph_vector",
                        vector_id=-1,
                    )
                )
            batches.append(units)
        return batches


class QueryComplexityScorer:
    QUESTION_WORDS = {
        "a", "an", "and", "are", "did", "do", "does", "for", "from", "in",
        "is", "of", "the", "to", "was", "were", "what", "when", "where",
        "which", "who", "whom", "whose",
    }
    MULTI_HOP_PATTERNS = [
        r"\b(same|different|both|either|neither)\b",
        r"\b(compare|comparison|versus|vs\.?|difference|similar)\b",
        r"\b(before|after|during|while|until|since)\b",
        r"\b(author|director|founder|producer|creator).*\b(born|birth|nationality|country|city)\b",
        r"\b(who|what|which|where|when)\b.*\b(who|what|which|where|when)\b",
    ]
    RELATION_PATTERNS = [
        r"\b(same|different)\s+(nationality|country|state|city|language|genre)\b",
        r"\b(older|younger|earlier|later|larger|smaller|more|less|fewer|greater)\b",
        r"\b(belong|owned|founded|created|established|directed|written|produced|born)\b",
        r"\b(nationality|country|birthplace|occupation|genre|release|location)\b",
    ]

    def compute(self, query: str) -> float:
        entities: set[str] = set()
        for quoted in re.findall(r'"([^"]+)"|\'([^\']+)\'', query):
            value = quoted[0] or quoted[1]
            if value:
                entities.add(value.strip().lower())
        for match in re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b", query):
            cleaned = match.strip()
            if cleaned.lower() not in self.QUESTION_WORDS and len(cleaned) > 1:
                entities.add(cleaned.lower())
        relation_count = sum(
            bool(re.search(pattern, query, re.IGNORECASE))
            for pattern in self.RELATION_PATTERNS
        )
        multi_hop = any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in self.MULTI_HOP_PATTERNS
        )
        return min(
            0.35 * min(len(entities) / 2.0, 1.0)
            + 0.25 * min(relation_count, 1)
            + 0.25 * float(multi_hop)
            + 0.15 * min(len(query) / 120.0, 1.0),
            1.0,
        )


class OfflineBaselines:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.sentence_index = SentenceIndex(config.sentence_index_dir)
        self.paragraph_index = ParagraphIndex(
            config.resolve("data/hotpotqa/vector_stores/valid_title_sentence")
        )
        self.raw_bridge_index = load_bridge_index(config.raw_bridge_index_file)
        self.neo4j_graph = self._load_neo4j_graph(config.neo4j_bridge_paths_file)
        self.reranker = APIReranker(
            config.rerank_model,
            config.rerank_api_key,
            config.rerank_base_url,
        )
        self.complexity = QueryComplexityScorer()
        self.rerank_calls = 0

    @staticmethod
    def _load_neo4j_graph(path: Path) -> dict[str, list[tuple[str, float]]]:
        weighted: dict[str, dict[str, float]] = defaultdict(dict)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                source = str(item["source_title"])
                target = str(item["target_title"])
                score = min(
                    float(item.get("source_title_match_confidence", 0.0)),
                    float(item.get("target_title_match_confidence", 0.0)),
                )
                weighted[source][target] = max(weighted[source].get(target, 0.0), score)
                weighted[target][source] = max(weighted[target].get(source, 0.0), score)
        return {
            title: sorted(neighbors.items(), key=lambda item: item[1], reverse=True)
            for title, neighbors in weighted.items()
        }

    def _rerank(
        self, query: str, candidates: list[EvidenceUnit], top_k: int
    ) -> list[EvidenceUnit]:
        for attempt in range(6):
            try:
                self.rerank_calls += 1
                return self.reranker.rerank(query, deepcopy(candidates), top_k)
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 20))
        return []

    def _best_title_sentence(
        self, title: str, query_vector: np.ndarray, source: str, score: float
    ) -> EvidenceUnit | None:
        best: EvidenceUnit | None = None
        best_relevance = -math.inf
        for record in self.sentence_index.title_sentences(title):
            relevance = float(np.dot(query_vector, self.sentence_index.vector(record.vector_id)))
            if relevance > best_relevance:
                best_relevance = relevance
                best = EvidenceUnit(
                    title=record.title,
                    sent_id=record.sent_id,
                    text=record.text,
                    score=max(score, relevance),
                    source=source,
                    vector_id=record.vector_id,
                )
        return best

    def _graph_expand(
        self,
        seed_titles: list[str],
        query_vector: np.ndarray,
        hops: int,
        limit_per_seed: int = 3,
    ) -> list[EvidenceUnit]:
        expanded: list[EvidenceUnit] = []
        seen = set(seed_titles)
        for seed in seed_titles:
            queue = deque([(seed, 0, 1.0)])
            found = 0
            visited = {seed}
            while queue and found < limit_per_seed:
                title, depth, path_score = queue.popleft()
                if depth >= hops:
                    continue
                for neighbor, edge_score in self.neo4j_graph.get(title, []):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_score = min(path_score, edge_score)
                    queue.append((neighbor, depth + 1, next_score))
                    if neighbor in seen:
                        continue
                    unit = self._best_title_sentence(
                        neighbor, query_vector, f"neo4j_graph_{depth + 1}hop", next_score
                    )
                    if unit is not None:
                        unit.metadata.update({"seed_title": seed, "graph_hop": depth + 1})
                        expanded.append(unit)
                        seen.add(neighbor)
                        found += 1
                    if found >= limit_per_seed:
                        break
        return expanded

    def _kg_expand(
        self, seeds: list[EvidenceUnit], query_vector: np.ndarray
    ) -> list[EvidenceUnit]:
        expanded: list[EvidenceUnit] = []
        seen = {unit.title for unit in seeds}
        for seed in seeds:
            links = self.raw_bridge_index.get(seed.title, [])[:3]
            for link in links:
                if link.target_title in seen:
                    continue
                unit = self._best_title_sentence(
                    link.target_title, query_vector, "kg_one_hop", link.bridge_prior
                )
                if unit is None:
                    continue
                unit.metadata.update(
                    {
                        "seed_title": seed.title,
                        "predicate": link.predicate,
                        "bridge_prior": link.bridge_prior,
                    }
                )
                expanded.append(unit)
                seen.add(unit.title)
        return expanded

    def _select_with_budget(
        self, query: str, units: list[EvidenceUnit], max_units: int
    ) -> list[EvidenceUnit]:
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        remaining = unique_by_title(units)
        selected: list[EvidenceUnit] = []
        selected_terms: set[str] = set()
        used_tokens = 0
        while remaining and len(selected) < max_units:
            best_index = -1
            best_value = -math.inf
            for index, unit in enumerate(remaining):
                if used_tokens + unit.token_count > self.config.context_budget:
                    continue
                unit_terms = set(re.findall(r"[a-z0-9]+", f"{unit.title} {unit.text}".lower()))
                relevance = len(query_terms & unit_terms) / max(len(query_terms), 1)
                novelty = (
                    1.0
                    if not selected_terms
                    else len(unit_terms - selected_terms) / max(len(unit_terms), 1)
                )
                value = (
                    0.45 * unit.score
                    + 0.30 * relevance
                    + 0.20 * novelty
                    - 0.15 * unit.token_count / self.config.context_budget
                    + (0.04 if unit.sent_id >= 0 else 0.0)
                )
                if value > best_value:
                    best_value = value
                    best_index = index
            if best_index < 0:
                break
            unit = remaining.pop(best_index)
            selected.append(unit)
            selected_terms.update(
                re.findall(r"[a-z0-9]+", f"{unit.title} {unit.text}".lower())
            )
            used_tokens += unit.token_count
        return selected

    def semantic_rag(
        self,
        query_vector: np.ndarray,
        sentence_candidates: list[EvidenceUnit] | None = None,
    ) -> MethodResult:
        started = time.perf_counter()
        candidates = sentence_candidates or self.sentence_index.search(query_vector, 10)
        context = unique_by_title(candidates, 7)
        return self._result(context, started, "semantic_rag", 0)

    def rerank_rag(
        self,
        query: str,
        query_vector: np.ndarray,
        sentence_candidates: list[EvidenceUnit] | None = None,
    ) -> MethodResult:
        started = time.perf_counter()
        candidates = sentence_candidates or self.sentence_index.search(query_vector, 10)
        context = unique_by_title(self._rerank(query, candidates, 7), 7)
        return self._result(context, started, "rerank_rag", 0)

    def graph_rag(
        self,
        query: str,
        query_vector: np.ndarray,
        sentence_candidates: list[EvidenceUnit] | None = None,
    ) -> MethodResult:
        started = time.perf_counter()
        candidates = sentence_candidates or self.sentence_index.search(query_vector, 10)
        seeds = unique_by_title(candidates, 7)
        expanded = self._graph_expand([unit.title for unit in seeds], query_vector, 2)
        context = self._select_with_budget(query, seeds + expanded, 20)
        return self._result(context, started, "graph_rag", len(expanded))

    def kg_rag(
        self, query: str, query_vector: np.ndarray, cached_reranked: list[EvidenceUnit]
    ) -> MethodResult:
        started = time.perf_counter()
        seeds = unique_by_title(deepcopy(cached_reranked), 7)
        expanded = self._kg_expand(seeds, query_vector)
        context = self._select_with_budget(query, seeds + expanded, 10)
        return self._result(context, started, "kg_rag", len(expanded))

    def macrag(
        self,
        query: str,
        query_vector: np.ndarray,
        sentence_candidates: list[EvidenceUnit] | None = None,
        paragraph_candidates: list[EvidenceUnit] | None = None,
    ) -> MethodResult:
        started = time.perf_counter()
        complexity = self.complexity.compute(query)
        if complexity < 0.45:
            candidates = sentence_candidates or self.sentence_index.search(query_vector, 10)
            route = "macrag_sentence"
        elif complexity < 0.80:
            sentence_units = sentence_candidates or self.sentence_index.search(query_vector, 10)
            paragraph_units = paragraph_candidates or self.paragraph_index.search(query_vector, 10)
            candidates = deepcopy(sentence_units[:5]) + deepcopy(paragraph_units[:5])
            candidates = self._rerank(query, candidates, 7)
            route = "macrag_mixed"
        else:
            candidates = paragraph_candidates or self.paragraph_index.search(query_vector, 10)
            route = "macrag_paragraph"
        context = self._select_with_budget(query, candidates, 7)
        result = self._result(context, started, route, 0)
        result.stats["complexity_score"] = complexity
        return result

    @staticmethod
    def _result(
        context: list[EvidenceUnit], started: float, route: str, expanded: int
    ) -> MethodResult:
        return MethodResult(
            initial_evidence=[],
            candidate_evidence=context,
            context_evidence=context,
            supporting_facts=[],
            stats={
                "route": route,
                "selected_context_units": len(context),
                "selected_context_tokens": sum(unit.token_count for unit in context),
                "extended_nodes": expanded,
                "time_ms": (time.perf_counter() - started) * 1000,
            },
        )
