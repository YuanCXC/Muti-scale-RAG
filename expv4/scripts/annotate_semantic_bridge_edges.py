from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from src.config import ExperimentConfig
from src.indexing.edge_semantic_annotation import (
    EdgeSemanticAnnotator,
    annotate_bridge_edges,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Directly validate every v4 bridge edge with an LLM"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    annotator = EdgeSemanticAnnotator(
        model=config.bridge_annotation_model,
        api_key=config.generation_api_key,
        base_url=config.generation_base_url,
        max_tokens=args.max_tokens,
    )
    report = annotate_bridge_edges(
        raw_index_path=config.raw_bridge_index_file,
        sentence_path=config.sentence_file,
        previous_annotations_path=config.bridge_semantic_annotations_file,
        output_path=config.bridge_edge_annotations_file,
        annotator=annotator,
        batch_size=args.batch_size,
        workers=args.workers or config.bridge_annotation_workers,
        resume=not args.no_resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
