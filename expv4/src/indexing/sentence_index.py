from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from tqdm import tqdm

from ..models import EvidenceUnit, SentenceRecord
from ..retrieval.embedding import EmbeddingClient
from .official_sentences import load_sentence_records


def _batches(items: list[SentenceRecord], size: int) -> Iterable[list[SentenceRecord]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_sentence_index(
    sentence_path: Path,
    index_dir: Path,
    embedder: EmbeddingClient,
    batch_size: int,
    workers: int = 1,
    limit: int | None = None,
) -> dict[str, int]:
    records = load_sentence_records(sentence_path)
    if limit:
        records = records[:limit]

    index: faiss.Index | None = None
    index_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = index_dir / "metadata.jsonl"

    batches = list(_batches(records, batch_size))

    def embed_batch(batch: list[SentenceRecord]) -> np.ndarray:
        return embedder.embed([record.text for record in batch])

    with metadata_path.open("w", encoding="utf-8") as metadata_handle:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            embedded_batches = executor.map(embed_batch, batches)
            iterator = zip(batches, embedded_batches)
            for batch, vectors in tqdm(
                iterator,
                total=len(batches),
                desc="Embedding official sentences",
            ):
                if len(vectors) != len(batch):
                    raise ValueError(
                        "Embedding API returned a different number of vectors"
                    )
                if index is None:
                    index = faiss.IndexFlatIP(vectors.shape[1])
                index.add(vectors)
                for record in batch:
                    metadata_handle.write(
                        json.dumps(
                            {
                                "vector_id": record.vector_id,
                                "title": record.title,
                                "sent_id": record.sent_id,
                                "text": record.text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    if index is None:
        raise ValueError("No official sentences were available for indexing")
    faiss.write_index(index, str(index_dir / "faiss.index"))
    report = {"sentence_count": len(records), "dimension": index.d}
    (index_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


class SentenceIndex:
    def __init__(self, index_dir: Path) -> None:
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.records: list[SentenceRecord] = []
        self.title_to_vector_ids: dict[str, list[int]] = defaultdict(list)
        with (index_dir / "metadata.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                record = SentenceRecord(**json.loads(line))
                self.records.append(record)
                self.title_to_vector_ids[record.title].append(record.vector_id)

    def vector(self, vector_id: int) -> np.ndarray:
        return np.asarray(self.index.reconstruct(vector_id), dtype=np.float32)

    def record(self, vector_id: int) -> SentenceRecord:
        return self.records[vector_id]

    def search(self, query_vector: np.ndarray, top_k: int) -> list[EvidenceUnit]:
        scores, ids = self.index.search(
            query_vector.reshape(1, -1).astype(np.float32), top_k
        )
        units: list[EvidenceUnit] = []
        for score, vector_id in zip(scores[0], ids[0]):
            if vector_id < 0:
                continue
            record = self.records[int(vector_id)]
            units.append(
                EvidenceUnit(
                    title=record.title,
                    sent_id=record.sent_id,
                    text=record.text,
                    score=float(score),
                    source="vector",
                    vector_id=record.vector_id,
                )
            )
        return units

    def title_sentences(self, title: str) -> list[SentenceRecord]:
        return [
            self.records[vector_id]
            for vector_id in self.title_to_vector_ids.get(title, [])
        ]
