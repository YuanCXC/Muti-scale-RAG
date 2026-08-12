from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..models import SentenceRecord


def build_official_sentence_file(
    validation_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    frame = pd.read_parquet(validation_path, columns=["context"])
    variants: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    occurrences = 0

    for context in frame["context"]:
        for title, sentences in zip(context["title"], context["sentences"]):
            variants[str(title)][
                tuple(str(sentence).strip() for sentence in sentences)
            ] += 1
            occurrences += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vector_id = 0
    sentence_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for title in sorted(variants):
            selected = variants[title].most_common(1)[0][0]
            for sent_id, text in enumerate(selected):
                record = {
                    "vector_id": vector_id,
                    "title": title,
                    "sent_id": sent_id,
                    "text": text,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                vector_id += 1
                sentence_count += 1

    report = {
        "question_count": len(frame),
        "document_occurrences": occurrences,
        "unique_titles": len(variants),
        "sentence_count": sentence_count,
        "titles_with_multiple_variants": sum(
            len(counter) > 1 for counter in variants.values()
        ),
    }
    report_path = output_path.with_name("official_sentences_report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def audit_gold_alignment(
    validation_path: Path,
    sentence_path: Path,
) -> dict[str, Any]:
    sentence_counts: Counter[str] = Counter()
    with sentence_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sentence_counts[json.loads(line)["title"]] += 1

    frame = pd.read_parquet(validation_path, columns=["id", "supporting_facts"])
    invalid_gold_facts: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        supporting_facts = row["supporting_facts"]
        for title, sent_id in zip(
            supporting_facts["title"], supporting_facts["sent_id"]
        ):
            title = str(title)
            sent_id = int(sent_id)
            sentence_count = sentence_counts.get(title, 0)
            if sent_id < 0 or sent_id >= sentence_count:
                invalid_gold_facts.append(
                    {
                        "id": str(row["id"]),
                        "title": title,
                        "sent_id": sent_id,
                        "sentence_count": sentence_count,
                    }
                )
    return {"invalid_gold_supporting_facts": invalid_gold_facts}


def load_sentence_records(path: Path) -> list[SentenceRecord]:
    records: list[SentenceRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            records.append(SentenceRecord(**item))
    return records
