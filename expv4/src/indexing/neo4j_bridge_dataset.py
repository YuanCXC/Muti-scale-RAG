from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from openai import OpenAI
from tqdm import tqdm

from ..retrieval.embedding import EmbeddingClient
from ..text_utils import normalize_space, normalize_title, title_aliases
from .sentence_index import SentenceIndex


DOCUMENT_BRIDGE_QUERY = """
MATCH (source:Section)-[:SEMANTIC_LINKS]->(source_entity:semantic)
      -[relation]->(target_entity:semantic)<-[:SEMANTIC_LINKS]-(target:Section)
WHERE source <> target AND type(relation) <> 'SEPARATES'
RETURN source.title AS source_title,
       source_entity.name AS source_entity,
       type(relation) AS predicate,
       target_entity.name AS target_entity,
       target.title AS target_title
"""


def _predicate_text(value: str) -> str:
    return normalize_space(re.sub(r"_+", " ", str(value or ""))).casefold()


def _title_entity_confidence(title: str, entity: str) -> float:
    entity_value = normalize_title(entity)
    if not entity_value:
        return 0.0
    aliases = title_aliases(title)
    canonical = normalize_title(title)
    if entity_value == canonical:
        return 1.0
    if entity_value in aliases:
        return 0.90
    if len(entity_value) >= 4 and (
        entity_value in canonical or canonical in entity_value
    ):
        return 0.70
    return 0.0


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_neo4j_bridge_paths(
    uri: str,
    user: str,
    password: str,
    database: str,
    output_path: Path,
    minimum_target_match: float = 0.90,
    limit: int | None = None,
) -> dict[str, Any]:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    raw_paths = 0
    filtered_target_mismatch = 0
    deduplicated: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    try:
        with driver.session(database=database, fetch_size=2000) as session:
            result = session.run(DOCUMENT_BRIDGE_QUERY)
            for record in tqdm(result, desc="Exporting Neo4j bridge paths"):
                raw_paths += 1
                source_title = str(record["source_title"] or "").strip()
                source_entity = str(record["source_entity"] or "").strip()
                predicate = _predicate_text(str(record["predicate"] or ""))
                target_entity = str(record["target_entity"] or "").strip()
                target_title = str(record["target_title"] or "").strip()
                target_match = _title_entity_confidence(target_title, target_entity)
                if target_match < minimum_target_match:
                    filtered_target_mismatch += 1
                    continue
                key = (
                    source_title,
                    source_entity,
                    predicate,
                    target_entity,
                    target_title,
                )
                deduplicated[key] = {
                    "source_title": source_title,
                    "source_entity": source_entity,
                    "predicate": predicate,
                    "target_entity": target_entity,
                    "target_title": target_title,
                    "source_title_match_confidence": _title_entity_confidence(
                        source_title, source_entity
                    ),
                    "target_title_match_confidence": target_match,
                }
                if limit and raw_paths >= limit:
                    break
    finally:
        driver.close()

    rows = sorted(
        deduplicated.values(),
        key=lambda item: (
            item["source_title"],
            item["target_title"],
            item["predicate"],
            item["source_entity"],
            item["target_entity"],
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for path_id, row in enumerate(rows):
            row["path_id"] = path_id
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "version": "v4_neo4j_bridge_paths",
        "raw_document_paths": raw_paths,
        "filtered_target_title_mismatch": filtered_target_mismatch,
        "retained_document_paths": len(rows),
        "source_titles": len({row["source_title"] for row in rows}),
        "target_titles": len({row["target_title"] for row in rows}),
        "predicate_types": len({row["predicate"] for row in rows}),
        "minimum_target_match": minimum_target_match,
    }
    _write_report(
        output_path.with_name("neo4j_bridge_paths_v4.report.json"), report
    )
    return report


def _read_jsonl(
    path: Path,
    skip_invalid: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not skip_invalid:
                    raise
    return rows


def _unique_titles(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    titles: list[str] = []
    for value in values:
        title = str(value)
        if title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def build_bridge_supervision_dataset(
    validation_path: Path,
    bridge_paths_path: Path,
    output_path: Path,
    max_positive_paths: int = 3,
    max_hard_negatives: int = 3,
) -> dict[str, Any]:
    paths = _read_jsonl(bridge_paths_path)
    pair_paths: dict[frozenset[str], list[dict[str, Any]]] = defaultdict(list)
    title_paths: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        pair = frozenset((path["source_title"], path["target_title"]))
        pair_paths[pair].append(path)
        title_paths[path["source_title"]].append(path)
        title_paths[path["target_title"]].append(path)

    frame = pd.read_parquet(
        validation_path,
        columns=["id", "question", "type", "level", "supporting_facts"],
    )
    frame = frame[frame["type"] == "bridge"]
    connected_questions = 0
    positive_path_count = 0
    hard_negative_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for _, row in tqdm(
            frame.iterrows(), total=len(frame), desc="Building bridge supervision"
        ):
            facts = row["supporting_facts"]
            fact_titles = [str(value) for value in facts["title"]]
            fact_sent_ids = [int(value) for value in facts["sent_id"]]
            gold_titles = _unique_titles(fact_titles)
            gold_pair = frozenset(gold_titles)
            positives = sorted(
                pair_paths.get(gold_pair, []),
                key=lambda item: (
                    item["target_title_match_confidence"],
                    item["source_title_match_confidence"],
                ),
                reverse=True,
            )[:max_positive_paths]
            if positives:
                connected_questions += 1

            negative_candidates: list[dict[str, Any]] = []
            negative_targets: set[str] = set()
            for title in gold_titles:
                for path in title_paths.get(title, []):
                    other = (
                        path["target_title"]
                        if path["source_title"] == title
                        else path["source_title"]
                    )
                    if other in gold_pair or other in negative_targets:
                        continue
                    negative_targets.add(other)
                    negative_candidates.append(path)
            negatives = sorted(
                negative_candidates,
                key=lambda item: (
                    item["target_title_match_confidence"],
                    item["source_title_match_confidence"],
                ),
                reverse=True,
            )[:max_hard_negatives]

            positive_path_count += len(positives)
            hard_negative_count += len(negatives)
            item = {
                "version": "v4_neo4j_bridge_supervision",
                "question_id": str(row["id"]),
                "question": str(row["question"]),
                "question_type": "bridge",
                "level": str(row["level"]),
                "gold_titles": gold_titles,
                "gold_supporting_facts": [
                    {"title": title, "sent_id": sent_id}
                    for title, sent_id in zip(fact_titles, fact_sent_ids)
                ],
                "graph_connected": bool(positives),
                "positive_paths": positives,
                "hard_negative_paths": negatives,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "version": "v4_neo4j_bridge_supervision",
        "bridge_questions": len(frame),
        "graph_connected_questions": connected_questions,
        "graph_connected_rate": round(connected_questions / max(len(frame), 1), 6),
        "positive_paths": positive_path_count,
        "hard_negative_paths": hard_negative_count,
        "max_positive_paths_per_question": max_positive_paths,
        "max_hard_negatives_per_question": max_hard_negatives,
    }
    _write_report(
        output_path.with_name("neo4j_bridge_supervision_v4.report.json"), report
    )
    return report


class BridgeSemanticAnnotator:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def annotate(
        self,
        question: str,
        path: dict[str, Any],
        source_sentences: list[dict[str, Any]],
        target_sentences: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "bridge_path": {
                "source_title": path["source_title"],
                "source_entity": path["source_entity"],
                "predicate": path["predicate"],
                "target_entity": path["target_entity"],
                "target_title": path["target_title"],
            },
            "source_sentences": [
                {"sent_id": item["sent_id"], "text": item["text"]}
                for item in source_sentences
            ],
            "target_sentences": [
                {"sent_id": item["sent_id"], "text": item["text"]}
                for item in target_sentences
            ],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You validate graph-derived evidence for multi-hop QA. "
                    "Use only the supplied sentences. Return JSON with: "
                    "relation_supported (boolean), bridge_useful_for_question "
                    "(boolean), source_sentence_id (integer or null), "
                    "target_sentence_id (integer or null), "
                    "evidence_type (explicit, implicit, or unsupported), and "
                    "reason (at most 20 words). Do not estimate numerical scores."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
        last_content = ""
        last_error = ""
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
                last_content = response.choices[0].message.content or ""
                annotation = json.loads(last_content)
                annotation["annotation_attempts"] = attempt
                return annotation
            except json.JSONDecodeError:
                last_error = "invalid_or_empty_json"
            except Exception as exc:
                last_error = f"api_error: {type(exc).__name__}: {exc}"
                time.sleep(min(8, 2**attempt))
        return {
            "annotation_error": last_error,
            "raw_response": last_content,
            "annotation_attempts": 3,
        }


def _supervision_candidates(
    supervision: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in supervision:
        for label, field in (
            ("positive", "positive_paths"),
            ("hard_negative", "hard_negative_paths"),
        ):
            for position, path in enumerate(item[field]):
                candidates.append(
                    {
                        "annotation_id": (
                            f"{item['question_id']}:{label}:{position}"
                        ),
                        "question_id": item["question_id"],
                        "question": item["question"],
                        "expected_label": label,
                        "gold_supporting_facts": item["gold_supporting_facts"],
                        "path": path,
                    }
                )
    return candidates


def _rank_title_sentences(
    sentence_index: SentenceIndex,
    title: str,
    query_vector: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, Any]] = []
    for record in sentence_index.title_sentences(title):
        score = float(np.dot(query_vector, sentence_index.vector(record.vector_id)))
        ranked.append((score, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "sent_id": record.sent_id,
            "text": record.text,
            "embedding_score": round(score, 6),
        }
        for score, record in ranked[:top_k]
    ]


def annotate_bridge_supervision(
    supervision_path: Path,
    sentence_index_dir: Path,
    output_path: Path,
    embedder: EmbeddingClient,
    annotator: BridgeSemanticAnnotator,
    embedding_batch_size: int = 64,
    top_sentences: int = 3,
    workers: int = 4,
    limit: int | None = None,
    audit_positive: int | None = None,
    audit_negative: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    supervision = _read_jsonl(supervision_path)
    all_candidates = _supervision_candidates(supervision)
    candidates = all_candidates
    if audit_positive is not None or audit_negative is not None:
        sampler = random.Random(42)
        positives = [
            item for item in candidates if item["expected_label"] == "positive"
        ]
        negatives = [
            item
            for item in candidates
            if item["expected_label"] == "hard_negative"
        ]
        positive_count = min(audit_positive or 0, len(positives))
        negative_count = min(audit_negative or 0, len(negatives))
        candidates = sampler.sample(positives, positive_count) + sampler.sample(
            negatives, negative_count
        )
    elif limit:
        candidates = candidates[:limit]

    retained_rows: list[dict[str, Any]] = []
    if resume and output_path.exists():
        existing = {
            item["annotation_id"]: item
            for item in _read_jsonl(output_path, skip_invalid=True)
        }
        retained_rows = [
            item
            for item in existing.values()
            if "annotation_error" not in item["llm_annotation"]
        ]
        completed = {item["annotation_id"] for item in retained_rows}
        candidates = [
            item for item in candidates if item["annotation_id"] not in completed
        ]

    sentence_index = SentenceIndex(sentence_index_dir)
    query_texts = [
        normalize_space(
            f"{item['question']} {item['path']['source_entity']} "
            f"{item['path']['predicate']} {item['path']['target_entity']}"
        )
        for item in candidates
    ]
    query_vectors: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(query_texts), embedding_batch_size),
        desc="Embedding bridge questions",
    ):
        query_vectors.extend(
            embedder.embed(query_texts[start : start + embedding_batch_size])
        )

    prepared: list[dict[str, Any]] = []
    for item, query_vector in zip(candidates, query_vectors):
        path = item["path"]
        prepared.append(
            {
                **item,
                "source_sentences": _rank_title_sentences(
                    sentence_index,
                    path["source_title"],
                    query_vector,
                    top_sentences,
                ),
                "target_sentences": _rank_title_sentences(
                    sentence_index,
                    path["target_title"],
                    query_vector,
                    top_sentences,
                ),
            }
        )

    def process(item: dict[str, Any]) -> dict[str, Any]:
        annotation = annotator.annotate(
            item["question"],
            item["path"],
            item["source_sentences"],
            item["target_sentences"],
        )
        source_scores = {
            sentence["sent_id"]: sentence["embedding_score"]
            for sentence in item["source_sentences"]
        }
        target_scores = {
            sentence["sent_id"]: sentence["embedding_score"]
            for sentence in item["target_sentences"]
        }
        source_sentence_id = annotation.get("source_sentence_id")
        target_sentence_id = annotation.get("target_sentence_id")
        annotation["relation_alignment_score"] = (
            source_scores.get(source_sentence_id, 0.0)
            if annotation.get("relation_supported")
            else 0.0
        )
        annotation["target_relevance_score"] = (
            target_scores.get(target_sentence_id, 0.0)
            if annotation.get("bridge_useful_for_question")
            else 0.0
        )
        return {
            "version": "v4_neo4j_bridge_semantic_annotation",
            **item,
            "llm_annotation": annotation,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(process, item) for item in prepared]
        with output_path.open("w", encoding="utf-8") as handle:
            for item in retained_rows:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Validating bridge semantics",
            ):
                item = future.result()
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                processed.append(item)

    all_rows = _read_jsonl(output_path)
    positive = [row for row in all_rows if row["expected_label"] == "positive"]
    negatives = [
        row for row in all_rows if row["expected_label"] == "hard_negative"
    ]
    valid_positive = [
        row
        for row in positive
        if "annotation_error" not in row["llm_annotation"]
    ]
    valid_negatives = [
        row
        for row in negatives
        if "annotation_error" not in row["llm_annotation"]
    ]
    failed_annotations = len(all_rows) - len(valid_positive) - len(valid_negatives)

    def embedding_hit(row: dict[str, Any], side: str) -> bool:
        path = row["path"]
        title = path[f"{side}_title"]
        ranked = row[f"{side}_sentences"]
        if not ranked:
            return False
        gold_ids = {
            fact["sent_id"]
            for fact in row["gold_supporting_facts"]
            if fact["title"] == title
        }
        return ranked[0]["sent_id"] in gold_ids

    report = {
        "version": "v4_neo4j_bridge_semantic_annotation",
        "annotations": len(all_rows),
        "new_annotations": len(processed),
        "available_supervision_candidates": len(all_candidates),
        "annotation_complete": (
            len(all_rows) >= len(all_candidates) and failed_annotations == 0
        ),
        "requested_limit": limit,
        "requested_audit_positive": audit_positive,
        "requested_audit_negative": audit_negative,
        "positive_annotations": len(positive),
        "hard_negative_annotations": len(negatives),
        "valid_positive_annotations": len(valid_positive),
        "valid_hard_negative_annotations": len(valid_negatives),
        "failed_annotations": failed_annotations,
        "positive_relation_supported_rate": round(
            sum(
                bool(row["llm_annotation"].get("relation_supported"))
                for row in valid_positive
            )
            / max(len(valid_positive), 1),
            6,
        ),
        "positive_bridge_useful_rate": round(
            sum(
                bool(
                    row["llm_annotation"].get("bridge_useful_for_question")
                )
                for row in valid_positive
            )
            / max(len(valid_positive), 1),
            6,
        ),
        "hard_negative_rejection_rate": round(
            sum(
                not bool(
                    row["llm_annotation"].get("bridge_useful_for_question")
                )
                for row in valid_negatives
            )
            / max(len(valid_negatives), 1),
            6,
        ),
        "positive_source_embedding_top1_gold_rate": round(
            sum(embedding_hit(row, "source") for row in positive)
            / max(len(positive), 1),
            6,
        ),
        "positive_target_embedding_top1_gold_rate": round(
            sum(embedding_hit(row, "target") for row in positive)
            / max(len(positive), 1),
            6,
        ),
        "embedding_model": embedder.model,
        "annotation_model": annotator.model,
    }
    _write_report(
        output_path.with_name(
            "neo4j_bridge_semantic_annotations_v4.report.json"
        ),
        report,
    )
    return report
