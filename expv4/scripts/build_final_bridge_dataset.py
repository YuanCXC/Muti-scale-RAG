from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ruff: noqa: E402


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from src.config import ExperimentConfig
from src.indexing.bridge_supervision import (
    build_bridge_supervision_dataset,
    build_clean_bridge_dataset,
)
from src.indexing.semantic_bridge_annotation import (
    BridgeSemanticAnnotator,
    annotate_bridge_supervision,
)
from src.retrieval.embedding import EmbeddingClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final non-Neo4j v4 bridge dataset"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument(
        "--stage",
        choices=["supervision", "annotate", "clean", "offline", "all"],
        default="offline",
    )
    parser.add_argument("--max-positive-paths", type=int, default=3)
    parser.add_argument("--max-hard-negatives", type=int, default=3)
    parser.add_argument("--top-sentences", type=int, default=3)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--annotation-max-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    reports = {}
    if args.stage in {"supervision", "offline", "all"}:
        reports["bridge_supervision"] = build_bridge_supervision_dataset(
            validation_path=config.validation_file,
            bridge_index_path=config.raw_bridge_index_file,
            output_path=config.bridge_supervision_file,
            max_positive_paths=args.max_positive_paths,
            max_hard_negatives=args.max_hard_negatives,
        )

    if args.stage in {"annotate", "all"}:
        embedder = EmbeddingClient(
            config.embedding_model,
            config.embedding_mode,
            config.embedding_api_key,
            config.embedding_base_url,
        )
        annotator = BridgeSemanticAnnotator(
            config.bridge_annotation_model,
            config.generation_api_key,
            config.generation_base_url,
            args.annotation_max_tokens or config.bridge_annotation_max_tokens,
        )
        reports["semantic_annotation"] = annotate_bridge_supervision(
            supervision_path=config.bridge_supervision_file,
            sentence_index_dir=config.sentence_index_dir,
            output_path=config.bridge_semantic_annotations_file,
            embedder=embedder,
            annotator=annotator,
            embedding_batch_size=config.embedding_batch_size,
            top_sentences=args.top_sentences,
            workers=args.workers or config.bridge_annotation_workers,
            limit=args.limit,
            resume=args.resume,
        )

    if args.stage in {"clean", "all"}:
        reports["clean_dataset"] = build_clean_bridge_dataset(
            annotations_path=config.bridge_semantic_annotations_file,
            output_path=config.bridge_clean_dataset_file,
        )

    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
