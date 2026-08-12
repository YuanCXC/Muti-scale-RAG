from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ..models import BridgeLink, SentenceRecord
from ..text_utils import normalize_title, title_aliases
from .official_sentences import load_sentence_records


@dataclass(frozen=True)
class ResolvedTitle:
    title: str
    matched_alias: str
    confidence: float


class TitleResolver:
    def __init__(self, titles: list[str]) -> None:
        candidates: dict[str, set[str]] = defaultdict(set)
        confidences: dict[tuple[str, str], float] = {}
        for title in titles:
            canonical = normalize_title(title)
            aliases = title_aliases(title)
            for alias in aliases:
                candidates[alias].add(title)
                confidences[(alias, title)] = 1.0 if alias == canonical else 0.75
        self.alias_to_title = {
            alias: next(iter(values))
            for alias, values in candidates.items()
            if len(values) == 1
        }
        self.alias_confidence = {
            alias: confidences[(alias, title)]
            for alias, title in self.alias_to_title.items()
        }

    def resolve(self, value: str) -> ResolvedTitle | None:
        normalized = normalize_title(value)
        if not normalized or normalized.isdigit():
            return None
        direct = self.alias_to_title.get(normalized)
        if direct:
            return ResolvedTitle(direct, normalized, self.alias_confidence[normalized])

        tokens = normalized.split()
        for length in range(min(len(tokens), 12), 1, -1):
            for start in range(0, len(tokens) - length + 1):
                alias = " ".join(tokens[start : start + length])
                title = self.alias_to_title.get(alias)
                if title:
                    return ResolvedTitle(
                        title, alias, min(0.70, self.alias_confidence[alias])
                    )
        return None


def _load_sentences_by_title(path: Path) -> dict[str, list[SentenceRecord]]:
    grouped: dict[str, list[SentenceRecord]] = defaultdict(list)
    for record in load_sentence_records(path):
        grouped[record.title].append(record)
    return dict(grouped)


def _ground_sentence(
    sentences: list[SentenceRecord],
    matched_alias: str,
    target_title: str,
) -> int | None:
    aliases = {matched_alias, *title_aliases(target_title)}
    aliases = {alias for alias in aliases if alias and len(alias) >= 3}
    for record in sentences:
        sentence = f" {normalize_title(record.text)} "
        if any(f" {alias} " in sentence for alias in aliases):
            return record.sent_id
    return None


def build_bridge_index(
    triplet_documents_path: Path,
    sentence_path: Path,
    output_path: Path,
    limit_documents: int | None = None,
    max_entity_document_frequency: int = 300,
) -> dict[str, int | float]:
    sentences_by_title = _load_sentences_by_title(sentence_path)
    titles = list(sentences_by_title)
    resolver = TitleResolver(titles)
    normalized_sources: dict[str, set[str]] = defaultdict(set)
    for title in titles:
        normalized_sources[normalize_title(title)].add(title)
    source_title_lookup = {
        normalized: next(iter(values))
        for normalized, values in normalized_sources.items()
        if len(values) == 1
    }
    exact_titles = set(titles)

    with triplet_documents_path.open("r", encoding="utf-8") as handle:
        documents = json.load(handle)
    if limit_documents:
        documents = documents[:limit_documents]

    raw_links: list[tuple[str, int, str, str, str, str, float]] = []
    skipped_sources = 0
    unresolved_entities = 0
    ungrounded_entities = 0

    for document in tqdm(documents, desc="Grounding triplet bridges"):
        raw_source_title = str(document.get("title", ""))
        source_title = (
            raw_source_title
            if raw_source_title in exact_titles
            else source_title_lookup.get(normalize_title(raw_source_title))
        )
        if not source_title:
            skipped_sources += 1
            continue
        source_sentences = sentences_by_title[source_title]

        for triplet in document.get("triplets", []):
            predicate = str(triplet.get("Predicate", "")).strip()
            for field, role in (("Subject", "subject"), ("Object", "object")):
                raw_entity = str(triplet.get(field, "")).strip()
                resolved = resolver.resolve(raw_entity)
                if not resolved or resolved.title == source_title:
                    unresolved_entities += 1
                    continue
                sent_id = _ground_sentence(
                    source_sentences,
                    resolved.matched_alias,
                    resolved.title,
                )
                if sent_id is None:
                    ungrounded_entities += 1
                    continue
                raw_links.append(
                    (
                        source_title,
                        sent_id,
                        resolved.title,
                        resolved.title,
                        predicate,
                        role,
                        resolved.confidence,
                    )
                )

    entity_documents: dict[str, set[str]] = defaultdict(set)
    for source_title, _, entity, _, _, _, _ in raw_links:
        entity_documents[entity].add(source_title)
    total_documents = max(len(sentences_by_title), 1)

    deduplicated: dict[tuple[str, int, str, str, str], BridgeLink] = {}
    hub_filtered_links = 0
    for (
        source_title,
        sent_id,
        entity,
        target_title,
        predicate,
        role,
        match_confidence,
    ) in raw_links:
        df = len(entity_documents[entity])
        if df > max_entity_document_frequency:
            hub_filtered_links += 1
            continue
        specificity = math.log((total_documents + 1) / (df + 1)) / math.log(
            total_documents + 1
        )
        bridge_prior = specificity * match_confidence
        link = BridgeLink(
            source_title=source_title,
            source_sentence_id=sent_id,
            bridge_entity=entity,
            target_title=target_title,
            predicate=predicate,
            entity_role=role,
            match_confidence=match_confidence,
            grounding_confidence=1.0,
            entity_specificity=specificity,
            bridge_prior=bridge_prior,
        )
        key = (source_title, sent_id, target_title, predicate, role)
        previous = deduplicated.get(key)
        if previous is None or link.bridge_prior > previous.bridge_prior:
            deduplicated[key] = link

    adjacency: dict[str, list[BridgeLink]] = defaultdict(list)
    for link in deduplicated.values():
        adjacency[link.source_title].append(link)
    for links in adjacency.values():
        links.sort(key=lambda item: item.bridge_prior, reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(dict(adjacency), handle, protocol=pickle.HIGHEST_PROTOCOL)

    report: dict[str, int | float] = {
        "input_documents": len(documents),
        "aligned_titles": len(sentences_by_title),
        "skipped_source_documents": skipped_sources,
        "unresolved_entity_sides": unresolved_entities,
        "ungrounded_entity_sides": ungrounded_entities,
        "raw_grounded_links": len(raw_links),
        "hub_filtered_links": hub_filtered_links,
        "max_entity_document_frequency": max_entity_document_frequency,
        "bridge_links": len(deduplicated),
        "exact_match_links": sum(
            link.match_confidence >= 0.99 for link in deduplicated.values()
        ),
        "alias_match_links": sum(
            0.74 <= link.match_confidence < 0.99 for link in deduplicated.values()
        ),
        "composite_match_links": sum(
            link.match_confidence < 0.74 for link in deduplicated.values()
        ),
        "source_titles_with_links": len(adjacency),
        "average_links_per_source": round(
            len(deduplicated) / max(len(adjacency), 1), 4
        ),
    }
    output_path.with_name("bridge_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def load_bridge_index(path: Path) -> dict[str, list[BridgeLink]]:
    with path.open("rb") as handle:
        return pickle.load(handle)
