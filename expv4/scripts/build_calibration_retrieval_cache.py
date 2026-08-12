from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

# ruff: noqa: E402

import pandas as pd


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from src.config import ExperimentConfig
from src.pipeline import AdaptiveRecoveryPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Calibration Retrieval Cache（构建校准检索缓存）"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    split_path = config.data_dir / "calibration_split_v4.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    calibration_ids = split["calibration_ids"]
    frame = pd.read_parquet(config.validation_file)
    rows_by_id = {str(row["id"]): row for _, row in frame.iterrows()}
    pipeline = AdaptiveRecoveryPipeline(config, enable_generation=False)

    cache = {}
    rerank_error_examples = 0
    for example_index, example_id in enumerate(calibration_ids, start=1):
        row = rows_by_id[example_id]
        initial, query_vector, retrieval_time_ms = pipeline.retrieve(
            str(row["question"])
        )
        if any(unit.metadata.get("rerank_error") for unit in initial):
            rerank_error_examples += 1
        cache[example_id] = {
            "question": str(row["question"]),
            "initial_evidence": initial,
            "query_vector": query_vector,
            "retrieval_time_ms": retrieval_time_ms,
        }
        if args.progress_every > 0 and example_index % args.progress_every == 0:
            print(
                f"Completed {example_index}/{len(calibration_ids)} "
                "examples（已缓存校准检索样本）",
                flush=True,
            )

    output_path = (
        Path(args.output)
        if args.output
        else config.data_dir / "calibration_retrieval_cache_v4.pkl"
    )
    with output_path.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "dataset_version": split["dataset_version"],
        "artifact": "Calibration Retrieval Cache（校准检索缓存）",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "examples": len(cache),
        "embedding_model": config.embedding_model,
        "rerank_model": config.rerank_model,
        "rerank_error_examples": rerank_error_examples,
        "vector_dimension": config.vector_dimension,
        "cache_file": str(output_path),
        "split_file": str(split_path),
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
