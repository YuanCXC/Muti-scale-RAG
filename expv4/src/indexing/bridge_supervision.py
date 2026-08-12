from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .bridge_index import load_bridge_index


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _unique_titles(values: Any) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def build_bridge_supervision_dataset(
    validation_path: Path,
    bridge_index_path: Path,
    output_path: Path,
    max_positive_paths: int = 3,
    max_hard_negatives: int = 3,
) -> dict[str, Any]:
    bridge_index = load_bridge_index(bridge_index_path)
    pair_paths: dict[frozenset[str], list[dict[str, Any]]] = defaultdict(list)
    title_paths: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for links in bridge_index.values():
        for link in links:
            path = asdict(link)
            pair = frozenset((link.source_title, link.target_title))
            pair_paths[pair].append(path)
            title_paths[link.source_title].append(path)
            title_paths[link.target_title].append(path)

    frame = pd.read_parquet(
        validation_path,
        columns=["id", "question", "type", "level", "supporting_facts"],
    )
    frame = frame[frame["type"] == "bridge"]
    connected_questions = 0
    positive_count = 0
    negative_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for _, row in tqdm(
            frame.iterrows(),
            total=len(frame),
            desc="Building final bridge supervision",
        ):
            facts = row["supporting_facts"]
            fact_titles = [str(value) for value in facts["title"]]
            fact_sent_ids = [int(value) for value in facts["sent_id"]]
            gold_titles = _unique_titles(fact_titles)
            gold_pair = frozenset(gold_titles)
            positives = sorted(
                pair_paths.get(gold_pair, []),
                key=lambda item: item["bridge_prior"],
                reverse=True,
            )[:max_positive_paths]
            if positives:
                connected_questions += 1

            negative_candidates: list[dict[str, Any]] = []
            seen_paths: set[tuple[Any, ...]] = set()
            seen_targets: set[str] = set()
            for title in gold_titles:
                for path in title_paths.get(title, []):
                    other = (
                        path["target_title"]
                        if path["source_title"] == title
                        else path["source_title"]
                    )
                    key = (
                        path["source_title"],
                        path["source_sentence_id"],
                        path["target_title"],
                        path["predicate"],
                        path["entity_role"],
                    )
                    if other in gold_pair or other in seen_targets or key in seen_paths:
                        continue
                    seen_targets.add(other)
                    seen_paths.add(key)
                    negative_candidates.append(path)
            negatives = sorted(
                negative_candidates,
                key=lambda item: item["bridge_prior"],
                reverse=True,
            )[:max_hard_negatives]

            positive_count += len(positives)
            negative_count += len(negatives)
            item = {
                "version": "v4_bridge_supervision",
                "question_id": str(row["id"]),
                "question": str(row["question"]),
                "question_type": "bridge",
                "level": str(row["level"]),
                "gold_titles": gold_titles,
                "gold_supporting_facts": [
                    {"title": title, "sent_id": sent_id}
                    for title, sent_id in zip(fact_titles, fact_sent_ids)
                ],
                "bridge_connected": bool(positives),
                "positive_paths": positives,
                "hard_negative_paths": negatives,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "version": "v4_bridge_supervision",
        "source": "bridge_index.pkl",
        "uses_neo4j": False,
        "bridge_questions": len(frame),
        "bridge_connected_questions": connected_questions,
        "bridge_connected_rate": round(connected_questions / max(len(frame), 1), 6),
        "positive_paths": positive_count,
        "hard_negative_paths": negative_count,
        "supervision_candidates": positive_count + negative_count,
        "max_positive_paths_per_question": max_positive_paths,
        "max_hard_negatives_per_question": max_hard_negatives,
    }
    _write_report(output_path.with_name("bridge_supervision_v4.report.json"), report)
    return report


def build_clean_bridge_dataset(
    annotations_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    annotations = _read_jsonl(annotations_path)
    counts: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in annotations:
            annotation = item["llm_annotation"]
            relation_supported = bool(annotation.get("relation_supported"))
            bridge_useful = bool(annotation.get("bridge_useful_for_question"))
            expected = item["expected_label"]
            if expected == "positive" and relation_supported and bridge_useful:
                clean_label = "valid_positive"
            elif expected == "hard_negative" and not bridge_useful:
                clean_label = "confirmed_negative"
            elif expected == "hard_negative" and relation_supported and bridge_useful:
                clean_label = "alternative_useful"
            else:
                clean_label = "ambiguous"
            counts[clean_label] += 1
            handle.write(
                json.dumps(
                    {
                        "version": "v4_clean_bridge_dataset",
                        "clean_label": clean_label,
                        **item,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "version": "v4_clean_bridge_dataset",
        "uses_neo4j": False,
        "examples": len(annotations),
        "label_counts": dict(counts),
    }
    _write_report(output_path.with_name("bridge_clean_dataset_v4.report.json"), report)
    return report
