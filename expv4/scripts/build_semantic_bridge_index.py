from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ruff: noqa: E402


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from src.config import ExperimentConfig
from src.indexing.semantic_bridge_index import build_semantic_bridge_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the final question-independent semantic bridge dataset"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    report = build_semantic_bridge_index(
        raw_index_path=config.raw_bridge_index_file,
        edge_annotations_path=config.bridge_edge_annotations_file,
        relation_dataset_path=config.bridge_relation_dataset_file,
        output_index_path=config.bridge_index_file,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
