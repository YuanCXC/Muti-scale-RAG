from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from .bridge_index import load_bridge_index
from .official_sentences import load_sentence_records


def _link_key(link: Any) -> tuple[str, int, str, str, str]:
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _existing_consensus(path: Path) -> dict[tuple[str, int, str, str, str], bool]:
    votes: dict[tuple[str, int, str, str, str], list[bool]] = defaultdict(list)
    for item in _read_jsonl(path):
        annotation = item.get("llm_annotation", {})
        if "annotation_error" in annotation or "relation_supported" not in annotation:
            continue
        edge = item["path"]
        key = (
            str(edge["source_title"]),
            int(edge["source_sentence_id"]),
            str(edge["target_title"]),
            str(edge["predicate"]),
            str(edge["entity_role"]),
        )
        votes[key].append(bool(annotation["relation_supported"]))
    return {
        key: values[0]
        for key, values in votes.items()
        if len(set(values)) == 1
    }


class EdgeSemanticAnnotator:
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

    def annotate_batch(self, edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = {
            "candidate_edges": [
                {
                    "edge_id": edge["edge_id"],
                    "source_title": edge["source_title"],
                    "source_sentence": edge["source_sentence_text"],
                    "predicate": edge["predicate"],
                    "entity_role": edge["entity_role"],
                    "bridge_entity": edge["bridge_entity"],
                    "target_title": edge["target_title"],
                    "target_context": edge["target_context"],
                }
                for edge in edges
            ]
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You validate cross-document bridge edges for multi-hop QA. "
                    "For every candidate, decide whether the source sentence truly "
                    "supports the stated predicate and direction, and whether its "
                    "entity refers to the exact target article described by target "
                    "context. Target context is identity evidence, not permission to "
                    "invent a relation. Use verdict supported only when both relation "
                    "and target identity are valid; rejected when either is wrong; "
                    "ambiguous when supplied text is insufficient. Return one JSON "
                    "object with decisions, an array containing every edge_id exactly "
                    "once. Each decision must contain edge_id, verdict (supported, "
                    "rejected, or ambiguous), confidence (high, medium, or low), and "
                    "reason of at most 15 words. Do not use outside knowledge."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        expected_ids = {edge["edge_id"] for edge in edges}
        last_error = ""
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                    messages=messages,
                )
                content = response.choices[0].message.content or ""
                decisions = json.loads(content).get("decisions", [])
                returned_ids = {int(item["edge_id"]) for item in decisions}
                verdicts = {"supported", "rejected", "ambiguous"}
                if returned_ids != expected_ids or any(
                    item.get("verdict") not in verdicts for item in decisions
                ):
                    raise ValueError("incomplete_or_invalid_decisions")
                return decisions
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < 3:
                    time.sleep(2**attempt)
        return [
            {
                "edge_id": edge["edge_id"],
                "annotation_error": last_error,
            }
            for edge in edges
        ]


def annotate_bridge_edges(
    raw_index_path: Path,
    sentence_path: Path,
    previous_annotations_path: Path,
    output_path: Path,
    annotator: EdgeSemanticAnnotator,
    batch_size: int = 20,
    workers: int = 500,
    resume: bool = True,
) -> dict[str, Any]:
    sentences_by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sentence_lookup: dict[tuple[str, int], str] = {}
    for record in load_sentence_records(sentence_path):
        sentence_lookup[(record.title, record.sent_id)] = record.text
        sentences_by_title[record.title].append(
            {"sent_id": record.sent_id, "text": record.text}
        )

    raw_index = load_bridge_index(raw_index_path)
    edges: list[dict[str, Any]] = []
    for links in raw_index.values():
        for link in links:
            edges.append(
                {
                    "edge_id": len(edges),
                    "source_title": link.source_title,
                    "source_sentence_id": link.source_sentence_id,
                    "source_sentence_text": sentence_lookup.get(
                        (link.source_title, link.source_sentence_id), ""
                    ),
                    "bridge_entity": link.bridge_entity,
                    "predicate": link.predicate,
                    "entity_role": link.entity_role,
                    "target_title": link.target_title,
                    "target_context": sentences_by_title.get(link.target_title, [])[:3],
                }
            )

    edge_by_id = {edge["edge_id"]: edge for edge in edges}
    consensus = _existing_consensus(previous_annotations_path)
    retained: dict[int, dict[str, Any]] = {}
    if resume:
        for row in _read_jsonl(output_path):
            if "annotation_error" not in row:
                retained[int(row["edge_id"])] = row

    for edge in edges:
        if edge["edge_id"] in retained:
            continue
        decision = consensus.get(_row_key(edge))
        if decision is None:
            continue
        retained[edge["edge_id"]] = {
            "version": "v4_edge_semantic_annotation",
            **edge,
            "verdict": "supported" if decision else "rejected",
            "confidence": "high",
            "reason": "Consistent prior direct LLM relation judgment.",
            "annotation_source": "reused_llm_consensus",
            "annotation_model": annotator.model,
        }

    pending = [edge for edge in edges if edge["edge_id"] not in retained]
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for edge_id in sorted(retained):
            handle.write(json.dumps(retained[edge_id], ensure_ascii=False) + "\n")
        handle.flush()

        def process(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            decisions = {
                int(item["edge_id"]): item
                for item in annotator.annotate_batch(batch)
            }
            rows: list[dict[str, Any]] = []
            for edge in batch:
                decision = decisions[edge["edge_id"]]
                if "annotation_error" in decision:
                    rows.append({**edge, **decision})
                else:
                    rows.append(
                        {
                            "version": "v4_edge_semantic_annotation",
                            **edge,
                            "verdict": decision["verdict"],
                            "confidence": decision.get("confidence", "low"),
                            "reason": decision.get("reason", ""),
                            "annotation_source": "direct_edge_llm",
                            "annotation_model": annotator.model,
                        }
                    )
            return rows

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(process, batch) for batch in batches]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Validating bridge edge batches",
            ):
                for row in future.result():
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    rows = _read_jsonl(output_path)
    valid = {int(row["edge_id"]): row for row in rows if "annotation_error" not in row}
    failed = [row for row in rows if "annotation_error" in row]
    verdict_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    for row in valid.values():
        verdict_counts[row["verdict"]] += 1
        source_counts[row["annotation_source"]] += 1
    missing = len(edge_by_id) - len(valid)
    report = {
        "version": "v4_edge_semantic_annotation",
        "candidate_edges": len(edges),
        "valid_annotations": len(valid),
        "failed_annotations": len(failed),
        "missing_annotations": missing,
        "annotation_complete": missing == 0,
        "verdict_counts": dict(verdict_counts),
        "annotation_source_counts": dict(source_counts),
        "new_api_batches": len(batches),
        "batch_size": batch_size,
        "annotation_model": annotator.model,
        "contains_questions": False,
        "contains_answers": False,
        "contains_gold_supporting_facts": False,
        "uses_neo4j": False,
    }
    output_path.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
