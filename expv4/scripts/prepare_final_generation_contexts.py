from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPV4 = ROOT / "expv4"
sys.path.insert(0, str(EXPV4))

from src.config import ExperimentConfig


RAW_VARIANT = "without_semantic_validation"
RAW_METHOD = "raw_bridge"
RAW_LABEL = "Raw Bridge（未经大模型语义验证的桥接关系）"


def load_sentences(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    sentences: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sentences[(str(row["title"]), int(row["sent_id"]))] = row
    return sentences


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    config = ExperimentConfig.load(EXPV4 / "configs" / "proposed.json")
    retrieval_dir = EXPV4 / "results" / "final_2000_v4" / "retrieval"
    split = json.loads(
        (ROOT / "data" / "v4" / "final_evaluation_split_2000_v4.json").read_text(
            encoding="utf-8"
        )
    )
    example_meta = {row["id"]: row for row in split["examples"]}
    sentence_map = load_sentences(config.sentence_file)

    raw_source = retrieval_dir / "frozen_ablation_one_hop_v4.jsonl"
    raw_contexts: list[dict[str, Any]] = []
    with raw_source.open(encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            if source.get("variant") != RAW_VARIANT:
                continue
            evidence = []
            for title, sent_id in source["context_evidence"]:
                sentence = sentence_map[(str(title), int(sent_id))]
                evidence.append(
                    {
                        "title": str(title),
                        "sent_id": int(sent_id),
                        "text": sentence["text"],
                        "score": 0.0,
                        "source": RAW_METHOD,
                        "vector_id": int(sentence["vector_id"]),
                        "metadata": {},
                    }
                )
            meta = example_meta[source["id"]]
            raw_contexts.append(
                {
                    "id": source["id"],
                    "question": source["question"],
                    "question_type": meta["type"],
                    "level": meta["level"],
                    "method": RAW_METHOD,
                    "label": RAW_LABEL,
                    "gold_supporting_facts": source["gold_supporting_facts"],
                    "context_evidence": evidence,
                    "metrics": source["metrics"],
                    "stats": source["stats"],
                }
            )

    if len(raw_contexts) != 2000:
        raise RuntimeError(f"Expected 2,000 Raw Bridge contexts, found {len(raw_contexts)}")

    raw_path = retrieval_dir / "raw_bridge_contexts_2000_v4.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in raw_contexts:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    contexts_13_path = retrieval_dir / "retrieval_contexts_13_variants_v4.jsonl"
    contexts_14_path = retrieval_dir / "retrieval_contexts_14_final_variants_v4.jsonl"
    with contexts_14_path.open("w", encoding="utf-8") as output:
        with contexts_13_path.open(encoding="utf-8") as source:
            for line in source:
                output.write(line)
        with raw_path.open(encoding="utf-8") as source:
            for line in source:
                output.write(line)

    old_table = json.loads(
        (retrieval_dir / "table8_multihop_semantic_retrieval_v4.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        row
        for row in old_table["table"]
        if row["variant"] in {"full_adaptive", "without_semantic_validation"}
    ]
    table = {
        **{key: value for key, value in old_table.items() if key not in {"table", "csv_file"}},
        "experiment": "Bridge Semantic Validation（桥接关系语义验证）",
        "recursive_bridge_completion": "excluded_from_final_method",
        "table": rows,
    }
    table_json = retrieval_dir / "table8_bridge_semantic_validation_retrieval_v4.json"
    table_csv = retrieval_dir / "table8_bridge_semantic_validation_retrieval_v4.csv"
    table["csv_file"] = str(table_csv)
    write_json(table_json, table)
    with table_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "examples": 2000,
        "base_context_variants": 13,
        "raw_bridge_contexts": len(raw_contexts),
        "final_context_variants": 14,
        "final_context_rows": sum(
            1 for line in contexts_14_path.open(encoding="utf-8") if line.strip()
        ),
        "recursive_bridge_completion_included": False,
        "raw_bridge_average_context_units": mean(
            len(row["context_evidence"]) for row in raw_contexts
        ),
        "contexts_file": str(contexts_14_path),
    }
    write_json(retrieval_dir / "final_generation_contexts_report_v4.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
