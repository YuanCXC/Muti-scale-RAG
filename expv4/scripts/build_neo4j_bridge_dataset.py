from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ruff: noqa: E402


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from src.config import ExperimentConfig
from src.indexing.neo4j_bridge_dataset import (
    BridgeSemanticAnnotator,
    annotate_bridge_supervision,
    build_bridge_supervision_dataset,
    extract_neo4j_bridge_paths,
)
from src.retrieval.embedding import EmbeddingClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build v4 Neo4j-derived bridge supervision data"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument(
        "--stage",
        choices=["extract", "supervision", "annotate", "offline", "all"],
        default="offline",
    )
    parser.add_argument("--minimum-target-match", type=float, default=0.90)
    parser.add_argument("--max-positive-paths", type=int, default=3)
    parser.add_argument("--max-hard-negatives", type=int, default=3)
    parser.add_argument("--top-sentences", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audit-positive", type=int)
    parser.add_argument("--audit-negative", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    reports = {}

    if args.stage in {"extract", "offline", "all"}:
        reports["neo4j_bridge_paths"] = extract_neo4j_bridge_paths(
            uri=config.neo4j_uri,
            user=config.neo4j_user,
            password=config.neo4j_password,
            database=config.neo4j_database,
            output_path=config.neo4j_bridge_paths_file,
            minimum_target_match=args.minimum_target_match,
            limit=args.limit if args.stage == "extract" else None,
        )

    if args.stage in {"supervision", "offline", "all"}:
        reports["bridge_supervision"] = build_bridge_supervision_dataset(
            validation_path=config.validation_file,
            bridge_paths_path=config.neo4j_bridge_paths_file,
            output_path=config.neo4j_bridge_supervision_file,
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
            config.bridge_annotation_max_tokens,
        )
        reports["semantic_annotation"] = annotate_bridge_supervision(
            supervision_path=config.neo4j_bridge_supervision_file,
            sentence_index_dir=config.sentence_index_dir,
            output_path=config.neo4j_bridge_annotations_file,
            embedder=embedder,
            annotator=annotator,
            embedding_batch_size=config.embedding_batch_size,
            top_sentences=args.top_sentences,
            workers=config.bridge_annotation_workers,
            limit=args.limit,
            audit_positive=args.audit_positive,
            audit_negative=args.audit_negative,
            resume=args.resume,
        )

    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
