from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Iterable

from ...models import EvidenceUnit


@dataclass(frozen=True)
class CorpusDocument:
    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


@dataclass
class BaselineResult:
    method: str
    evidence: list[EvidenceUnit]
    retrieval_seconds: float
    online_llm_calls: int = 0
    metadata: dict = field(default_factory=dict)


class EvidenceBudget:
    """Common post-retrieval budget; it never changes a method's ranking."""

    def __init__(self, max_tokens: int = 3600, max_units: int = 12) -> None:
        self.max_tokens = max_tokens
        self.max_units = max_units

    def apply(self, ranked: Iterable[EvidenceUnit]) -> list[EvidenceUnit]:
        selected: list[EvidenceUnit] = []
        seen: set[tuple[str, int]] = set()
        used = 0
        for unit in ranked:
            if unit.key in seen:
                continue
            if selected and used + unit.token_count > self.max_tokens:
                continue
            selected.append(unit)
            seen.add(unit.key)
            used += unit.token_count
            if len(selected) >= self.max_units:
                break
        return selected


class StandardCorpus:
    def __init__(self, documents: list[CorpusDocument]) -> None:
        self.documents = documents
        self.by_title = {document.title: document for document in documents}

    @classmethod
    def from_official_sentences(cls, path: Path) -> "StandardCorpus":
        grouped: dict[str, list[tuple[int, str]]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                grouped.setdefault(str(item["title"]), []).append(
                    (int(item["sent_id"]), str(item["text"]))
                )
        documents = [
            CorpusDocument(
                title=title,
                sentences=tuple(text for _, text in sorted(sentences)),
            )
            for title, sentences in grouped.items()
        ]
        documents.sort(key=lambda item: item.title)
        return cls(documents)

    def write_shared_formats(self, output_dir: Path) -> dict[str, str | int]:
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "corpus_documents_v4.jsonl"
        hippo_path = output_dir / "hipporag_corpus_v4.json"
        mdr_path = output_dir / "mdr_corpus_v4.jsonl"
        macrag_path = output_dir / "macrag_corpus_v4.json"

        with jsonl_path.open("w", encoding="utf-8") as common, mdr_path.open(
            "w", encoding="utf-8"
        ) as mdr:
            for index, document in enumerate(self.documents):
                row = {"id": index, **asdict(document), "text": document.text}
                common.write(json.dumps(row, ensure_ascii=False) + "\n")
                mdr.write(
                    json.dumps(
                        {"title": document.title, "text": document.text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        hippo_path.write_text(
            json.dumps(
                [
                    {"title": doc.title, "text": doc.text, "idx": index}
                    for index, doc in enumerate(self.documents)
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        macrag_path.write_text(
            json.dumps(
                [
                    {
                        "document_id": index,
                        "title": doc.title,
                        "context": doc.text,
                    }
                    for index, doc in enumerate(self.documents)
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {
            "documents": len(self.documents),
            "common_jsonl": str(jsonl_path),
            "mdr_jsonl": str(mdr_path),
            "hipporag_json": str(hippo_path),
            "macrag_json": str(macrag_path),
        }

    def evidence_for_titles(
        self,
        titles: Iterable[tuple[str, float]],
        source: str,
    ) -> list[EvidenceUnit]:
        output: list[EvidenceUnit] = []
        for rank, (title, score) in enumerate(titles):
            document = self.by_title.get(str(title))
            if document is None:
                continue
            for sent_id, text in enumerate(document.sentences):
                output.append(
                    EvidenceUnit(
                        title=document.title,
                        sent_id=sent_id,
                        text=text,
                        score=float(score),
                        source=source,
                        vector_id=-1,
                        metadata={"document_rank": rank + 1},
                    )
                )
        return output


class Timer:
    def __enter__(self) -> "Timer":
        self.started = perf_counter()
        return self

    def __exit__(self, *_args) -> None:
        self.seconds = perf_counter() - self.started
