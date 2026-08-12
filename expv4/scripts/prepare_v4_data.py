from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ruff: noqa: E402


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from src.config import ExperimentConfig
from src.indexing.official_sentences import (
    audit_gold_alignment,
    build_official_sentence_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build official HotpotQA sentence records for experiment v4"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    report = build_official_sentence_file(config.validation_file, config.sentence_file)
    report.update(audit_gold_alignment(config.validation_file, config.sentence_file))
    config.sentence_file.with_name("official_sentences_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
