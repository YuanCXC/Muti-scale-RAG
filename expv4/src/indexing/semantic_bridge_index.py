from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..models import BridgeLink
from .bridge_index import load_bridge_index


def _link_key(link: BridgeLink) -> tuple[str, int, str, str, str]:
    return (
        link.source_title,
        int(link.source_sentence_id),
        link.target_title,
        link.predicate,
        link.entity_role,
    )


def _row_key(row: dict[str, Any]) -> tuple[str, int, str, str, str]:
    return (
        str(row["source_title"]),
        int(row["source_sentence_id"]),
        str(row["target_title"]),
        str(row["predicate"]),
        str(row["entity_role"]),
    )


def build_semantic_bridge_index(
    raw_index_path: Path,
    edge_annotations_path: Path,
    relation_dataset_path: Path,
    output_index_path: Path,
) -> dict[str, Any]:
    annotations: dict[tuple[str, int, str, str, str], dict[str, Any]] = {}
    with edge_annotations_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "annotation_error" not in row:
                annotations[_row_key(row)] = row

    raw_index = load_bridge_index(raw_index_path)
    input_edges = sum(len(links) for links in raw_index.values())
    if len(annotations) != input_edges:
        raise ValueError(
            f"Semantic annotations are incomplete: {len(annotations)}/{input_edges}"
        )

    confidence_scores = {"high": 1.0, "medium": 0.85, "low": 0.70}
    final_adjacency: dict[str, list[BridgeLink]] = defaultdict(list)
    verdict_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    relation_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with relation_dataset_path.open("w", encoding="utf-8") as dataset_handle:
        for source_title, links in tqdm(
            raw_index.items(), desc="Building direct semantic bridge index"
        ):
            for link in links:
                annotation = annotations[_link_key(link)]
                verdict = annotation["verdict"]
                confidence = annotation.get("confidence", "low")
                keep = verdict == "supported"
                semantic_score = confidence_scores.get(confidence, 0.70) if keep else 0.0
                final_prior = link.bridge_prior * semantic_score
                verdict_counts[verdict] += 1
                source_counts[annotation["annotation_source"]] += 1

                record = {
                    "version": "v4_direct_semantic_bridge_relation",
                    "edge_id": annotation["edge_id"],
                    "source_title": link.source_title,
                    "source_sentence_id": link.source_sentence_id,
                    "source_sentence_text": annotation["source_sentence_text"],
                    "bridge_entity": link.bridge_entity,
                    "predicate": link.predicate,
                    "entity_role": link.entity_role,
                    "target_title": link.target_title,
                    "target_context": annotation["target_context"],
                    "match_confidence": link.match_confidence,
                    "entity_specificity": link.entity_specificity,
                    "original_bridge_prior": link.bridge_prior,
                    "semantic_verdict": verdict,
                    "semantic_confidence": confidence,
                    "semantic_reason": annotation.get("reason", ""),
                    "annotation_source": annotation["annotation_source"],
                    "annotation_model": annotation["annotation_model"],
                    "semantic_relation_score": semantic_score,
                    "semantic_bridge_prior": round(final_prior, 6),
                    "included_in_final_index": keep,
                }
                dataset_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                if keep:
                    final_adjacency[source_title].append(
                        BridgeLink(
                            source_title=link.source_title,
                            source_sentence_id=link.source_sentence_id,
                            bridge_entity=link.bridge_entity,
                            target_title=link.target_title,
                            predicate=link.predicate,
                            entity_role=link.entity_role,
                            match_confidence=link.match_confidence,
                            grounding_confidence=semantic_score,
                            entity_specificity=link.entity_specificity,
                            bridge_prior=final_prior,
                        )
                    )

    for links in final_adjacency.values():
        links.sort(key=lambda item: item.bridge_prior, reverse=True)
    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    with output_index_path.open("wb") as handle:
        pickle.dump(dict(final_adjacency), handle, protocol=pickle.HIGHEST_PROTOCOL)

    output_edges = sum(len(links) for links in final_adjacency.values())
    report = {
        "version": "v4_direct_semantic_bridge_index",
        "uses_neo4j": False,
        "contains_questions": False,
        "contains_answers": False,
        "contains_gold_supporting_facts": False,
        "input_edges": input_edges,
        "direct_llm_validated_edges": len(annotations),
        "output_edges": output_edges,
        "removed_edges": input_edges - output_edges,
        "source_titles": len(final_adjacency),
        "semantic_verdict_counts": dict(verdict_counts),
        "annotation_source_counts": dict(source_counts),
        "selection_rule": "supported_only",
    }
    relation_dataset_path.with_name(
        "bridge_relation_dataset_v4.report.json"
    ).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_index_path.with_name(
        "bridge_index_semantic_v4.report.json"
    ).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
