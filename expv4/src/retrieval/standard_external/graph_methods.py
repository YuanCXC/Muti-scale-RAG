from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from ...models import EvidenceUnit
from .protocol import BaselineResult, EvidenceBudget, StandardCorpus, Timer


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class RelationGraph:
    """Shared, LLM-validated relation storage for unified-resource reproduction."""

    def __init__(self) -> None:
        self.neighbors: dict[str, dict[str, float]] = defaultdict(dict)

    def add(self, source: str, target: str, weight: float) -> None:
        if source == target:
            return
        self.neighbors[source][target] = max(self.neighbors[source].get(target, 0.0), weight)
        self.neighbors[target][source] = max(self.neighbors[target].get(source, 0.0), weight)

    def sorted_neighbors(self, title: str) -> list[tuple[str, float]]:
        return sorted(self.neighbors.get(title, {}).items(), key=lambda x: x[1], reverse=True)


class KG2RAGAdapter:
    """KG²RAG core: retrieved seeds -> KG expansion -> coherent organization."""

    source = "nju-websoft/KG2RAG@7d626c77"

    def __init__(self, corpus: StandardCorpus, graph: RelationGraph, budget: EvidenceBudget) -> None:
        self.corpus = corpus
        self.graph = graph
        self.budget = budget

    def retrieve(self, seeds: list[tuple[str, float]], query: str) -> BaselineResult:
        with Timer() as timer:
            query_terms = _terms(query)
            scores: dict[str, float] = {}
            parent: dict[str, str] = {}
            for rank, (title, score) in enumerate(seeds[:5]):
                scores[title] = max(scores.get(title, 0.0), float(score) + 1.0 / (rank + 1))
                for target, edge_weight in self.graph.sorted_neighbors(title)[:5]:
                    document = self.corpus.by_title.get(target)
                    if document is None:
                        continue
                    lexical = len(query_terms & _terms(target + " " + document.text)) / max(len(query_terms), 1)
                    expansion_score = 0.55 * float(score) + 0.30 * edge_weight + 0.15 * lexical
                    if expansion_score > scores.get(target, -math.inf):
                        scores[target] = expansion_score
                        parent[target] = title
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            # KG²RAG's organization principle keeps a seed and its expansions adjacent.
            organized: list[tuple[str, float]] = []
            used: set[str] = set()
            for title, score in ranked:
                root = parent.get(title)
                if root and root not in used and root in scores:
                    organized.append((root, scores[root]))
                    used.add(root)
                if title not in used:
                    organized.append((title, score))
                    used.add(title)
            evidence = self.budget.apply(self.corpus.evidence_for_titles(organized, "kg2rag"))
        return BaselineResult(
            method="kg2rag",
            evidence=evidence,
            retrieval_seconds=timer.seconds,
            metadata={
                "implementation_source": self.source,
                "reproduction": "unified_resource_core_mechanism",
                "seed_documents": min(5, len(seeds)),
            },
        )


class HippoRAG2Adapter:
    """HippoRAG 2 core: query-seeded personalized PageRank plus retrieval signal."""

    source = "OSU-NLP-Group/HippoRAG@c617143f"

    def __init__(self, corpus: StandardCorpus, graph: RelationGraph, budget: EvidenceBudget) -> None:
        self.corpus = corpus
        self.graph = graph
        self.budget = budget

    def retrieve(self, seeds: list[tuple[str, float]], query: str) -> BaselineResult:
        with Timer() as timer:
            nodes = set(title for title, _ in seeds)
            for title, _ in seeds:
                nodes.update(target for target, _ in self.graph.sorted_neighbors(title)[:30])
            nodes = {title for title in nodes if title in self.corpus.by_title}
            reset = {title: max(0.0, score) for title, score in seeds if title in nodes}
            total_reset = sum(reset.values()) or 1.0
            reset = {title: value / total_reset for title, value in reset.items()}
            probability = dict(reset)
            damping = 0.5
            for _ in range(30):
                updated = {title: (1.0 - damping) * reset.get(title, 0.0) for title in nodes}
                for source in nodes:
                    outgoing = [(target, weight) for target, weight in self.graph.sorted_neighbors(source) if target in nodes]
                    norm = sum(weight for _, weight in outgoing)
                    if norm <= 0:
                        continue
                    for target, weight in outgoing:
                        updated[target] += damping * probability.get(source, 0.0) * weight / norm
                probability = updated
            dense = dict(seeds)
            ranked = sorted(
                ((title, probability.get(title, 0.0) + 0.05 * max(0.0, dense.get(title, 0.0))) for title in nodes),
                key=lambda item: item[1],
                reverse=True,
            )
            evidence = self.budget.apply(self.corpus.evidence_for_titles(ranked, "hipporag2"))
        return BaselineResult(
            method="hipporag2",
            evidence=evidence,
            retrieval_seconds=timer.seconds,
            metadata={
                "implementation_source": self.source,
                "reproduction": "unified_resource_core_mechanism",
                "ppr_damping": damping,
            },
        )


class GraphRAGLocalAdapter:
    """GraphRAG Local Search evidence mixing over a shared BYOG index."""

    source = "microsoft/graphrag@14a00ad8"

    def __init__(self, corpus: StandardCorpus, graph: RelationGraph, budget: EvidenceBudget) -> None:
        self.corpus = corpus
        self.graph = graph
        self.budget = budget

    def retrieve(self, seeds: list[tuple[str, float]], query: str) -> BaselineResult:
        with Timer() as timer:
            scores: dict[str, float] = dict(seeds)
            for seed_rank, (seed, seed_score) in enumerate(seeds[:5]):
                for target, weight in self.graph.sorted_neighbors(seed)[:10]:
                    scores[target] = max(
                        scores.get(target, 0.0),
                        0.65 * max(0.0, seed_score) + 0.35 * weight - 0.01 * seed_rank,
                    )
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            evidence = self.budget.apply(self.corpus.evidence_for_titles(ranked, "graphrag_local"))
        return BaselineResult(
            method="graphrag_local",
            evidence=evidence,
            retrieval_seconds=timer.seconds,
            metadata={
                "implementation_source": self.source,
                "reproduction": "unified_resource_core_mechanism",
                "query_mode": "local",
                "index_mode": "BYOG",
            },
        )


class MacRAGAdapter:
    """MacRAG core: fine evidence -> parent ranking -> local document scale-up."""

    source = "Leezekun/MacRAG@b1b28122"

    def __init__(self, corpus: StandardCorpus, budget: EvidenceBudget) -> None:
        self.corpus = corpus
        self.budget = budget

    def retrieve(self, sentence_candidates: list[EvidenceUnit], query: str) -> BaselineResult:
        with Timer() as timer:
            query_terms = _terms(query)
            parent_scores: dict[str, float] = {}
            best_sentence: dict[str, int] = {}
            for unit in sentence_candidates[:100]:
                lexical = len(query_terms & _terms(unit.text)) / max(len(query_terms), 1)
                value = 0.75 * unit.score + 0.25 * lexical
                if value > parent_scores.get(unit.title, -math.inf):
                    parent_scores[unit.title] = value
                    best_sentence[unit.title] = unit.sent_id
            ranked_titles = sorted(parent_scores.items(), key=lambda item: item[1], reverse=True)[:28]
            evidence: list[EvidenceUnit] = []
            for document_rank, (title, score) in enumerate(ranked_titles):
                document = self.corpus.by_title.get(title)
                if document is None:
                    continue
                center = best_sentence[title]
                for sent_id in range(max(0, center - 1), min(len(document.sentences), center + 2)):
                    evidence.append(
                        EvidenceUnit(
                            title=title,
                            sent_id=sent_id,
                            text=document.sentences[sent_id],
                            score=score,
                            source="macrag",
                            vector_id=-1,
                            metadata={"document_rank": document_rank + 1, "scale": "parent-neighbor"},
                        )
                    )
            evidence = self.budget.apply(evidence)
        return BaselineResult(
            method="macrag",
            evidence=evidence,
            retrieval_seconds=timer.seconds,
            metadata={
                "implementation_source": self.source,
                "reproduction": "unified_resource_core_mechanism",
                "input_candidates": len(sentence_candidates),
                "scale_up_factor": 4,
            },
        )
