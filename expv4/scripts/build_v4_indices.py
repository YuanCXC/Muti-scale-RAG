from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ruff: noqa: E402


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from src.config import ExperimentConfig
from src.indexing.bridge_index import build_bridge_index
from src.indexing.official_sentences import build_official_sentence_file
from src.indexing.sentence_index import build_sentence_index
from src.retrieval.embedding import EmbeddingClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v4 sentence and bridge indexes")
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument("--only", choices=["all", "sentence", "bridge"], default="all")
    parser.add_argument("--limit-sentences", type=int)
    parser.add_argument("--limit-documents", type=int)
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    if not config.sentence_file.exists():
        build_official_sentence_file(config.validation_file, config.sentence_file)

    reports = {}
    if args.only in {"all", "sentence"}:
        embedder = EmbeddingClient(
            config.embedding_model,
            config.embedding_mode,
            config.embedding_api_key,
            config.embedding_base_url,
        )
        reports["sentence_index"] = build_sentence_index(
            config.sentence_file,
            config.sentence_index_dir,
            embedder,
            config.embedding_batch_size,
            workers=config.embedding_workers,
            limit=args.limit_sentences,
        )
    if args.only in {"all", "bridge"}:
        reports["bridge_index"] = build_bridge_index(
            config.triplet_documents_file,
            config.sentence_file,
            config.raw_bridge_index_file,
            limit_documents=args.limit_documents,
            max_entity_document_frequency=config.bridge_max_document_frequency,
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
