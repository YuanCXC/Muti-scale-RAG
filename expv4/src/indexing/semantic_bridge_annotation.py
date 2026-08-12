from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from ..retrieval.embedding import EmbeddingClient
from ..text_utils import normalize_space
from .sentence_index import SentenceIndex


def _read_jsonl(path: Path, skip_invalid: bool = False) -> list[dict[str, Any]]:
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


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
        semantic_keys = (
            "source_title",
            "source_sentence_id",
            "source_entity",
            "bridge_entity",
            "predicate",
            "entity_role",
            "target_entity",
            "target_title",
        )
        payload = {
            "question": question,
            "bridge_path": {
                key: path[key] for key in semantic_keys if key in path
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
                    "You validate corpus-derived bridge evidence for multi-hop QA. "
                    "Use only the supplied sentences. Return JSON with: "
                    "relation_supported (boolean), bridge_useful_for_question "
                    "(boolean), source_sentence_id (integer or null), "
                    "target_sentence_id (integer or null), evidence_type "
                    "(explicit, implicit, or unsupported), and reason "
                    "(at most 20 words). Do not estimate numerical scores."
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
                        "annotation_id": f"{item['question_id']}:{label}:{position}",
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
    required_sent_id: int | None = None,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, Any]] = []
    for record in sentence_index.title_sentences(title):
        score = float(np.dot(query_vector, sentence_index.vector(record.vector_id)))
        ranked.append((score, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[:top_k]
    if required_sent_id is not None and all(
        record.sent_id != required_sent_id for _, record in selected
    ):
        required = next(
            (
                item
                for item in ranked
                if item[1].sent_id == required_sent_id
            ),
            None,
        )
        if required is not None:
            selected.append(required)
    return [
        {
            "sent_id": record.sent_id,
            "text": record.text,
            "embedding_score": round(score, 6),
        }
        for score, record in selected
    ]


def _path_query(question: str, path: dict[str, Any]) -> str:
    source_entity = str(
        path.get("source_entity") or path.get("source_title") or ""
    )
    target_entity = str(
        path.get("target_entity")
        or path.get("bridge_entity")
        or path.get("target_title")
        or ""
    )
    return normalize_space(
        f"{question} {source_entity} {path.get('predicate', '')} {target_entity}"
    )


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
        positives = [x for x in candidates if x["expected_label"] == "positive"]
        negatives = [
            x for x in candidates if x["expected_label"] == "hard_negative"
        ]
        candidates = sampler.sample(
            positives, min(audit_positive or 0, len(positives))
        ) + sampler.sample(
            negatives, min(audit_negative or 0, len(negatives))
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
    query_texts = [_path_query(item["question"], item["path"]) for item in candidates]
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
                    path.get("source_sentence_id"),
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
        annotation["relation_alignment_score"] = (
            source_scores.get(annotation.get("source_sentence_id"), 0.0)
            if annotation.get("relation_supported")
            else 0.0
        )
        annotation["target_relevance_score"] = (
            target_scores.get(annotation.get("target_sentence_id"), 0.0)
            if annotation.get("bridge_useful_for_question")
            else 0.0
        )
        return {
            "version": "v4_bridge_semantic_annotation",
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
    positive = [x for x in all_rows if x["expected_label"] == "positive"]
    negatives = [x for x in all_rows if x["expected_label"] == "hard_negative"]
    valid_positive = [
        x for x in positive if "annotation_error" not in x["llm_annotation"]
    ]
    valid_negatives = [
        x for x in negatives if "annotation_error" not in x["llm_annotation"]
    ]
    failed = len(all_rows) - len(valid_positive) - len(valid_negatives)

    def embedding_hit(row: dict[str, Any], side: str) -> bool:
        title = row["path"][f"{side}_title"]
        ranked = row[f"{side}_sentences"]
        gold_ids = {
            fact["sent_id"]
            for fact in row["gold_supporting_facts"]
            if fact["title"] == title
        }
        return bool(ranked) and ranked[0]["sent_id"] in gold_ids

    report = {
        "version": "v4_bridge_semantic_annotation",
        "annotations": len(all_rows),
        "new_annotations": len(processed),
        "available_supervision_candidates": len(all_candidates),
        "annotation_complete": len(all_rows) >= len(all_candidates) and failed == 0,
        "positive_annotations": len(positive),
        "hard_negative_annotations": len(negatives),
        "valid_positive_annotations": len(valid_positive),
        "valid_hard_negative_annotations": len(valid_negatives),
        "failed_annotations": failed,
        "positive_relation_supported_rate": round(
            sum(bool(x["llm_annotation"].get("relation_supported")) for x in valid_positive)
            / max(len(valid_positive), 1),
            6,
        ),
        "positive_bridge_useful_rate": round(
            sum(
                bool(x["llm_annotation"].get("bridge_useful_for_question"))
                for x in valid_positive
            )
            / max(len(valid_positive), 1),
            6,
        ),
        "hard_negative_rejection_rate": round(
            sum(
                not bool(x["llm_annotation"].get("bridge_useful_for_question"))
                for x in valid_negatives
            )
            / max(len(valid_negatives), 1),
            6,
        ),
        "positive_source_embedding_top1_gold_rate": round(
            sum(embedding_hit(x, "source") for x in positive)
            / max(len(positive), 1),
            6,
        ),
        "positive_target_embedding_top1_gold_rate": round(
            sum(embedding_hit(x, "target") for x in positive)
            / max(len(positive), 1),
            6,
        ),
        "embedding_model": embedder.model,
        "annotation_model": annotator.model,
    }
    _write_report(
        output_path.with_name("bridge_semantic_annotations_v4.report.json"),
        report,
    )
    return report
