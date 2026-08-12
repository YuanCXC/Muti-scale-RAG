from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import pandas as pd


EXPV4_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPV4_ROOT.parent
sys.path.insert(0, str(EXPV4_ROOT))

from src.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze Final Balanced Evaluation Split（冻结最终平衡评估集）"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--per-type", type=int, default=1000)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "v4" / "final_evaluation_split_2000_v4.json"),
    )
    parser.add_argument(
        "--cache-output",
        default=str(PROJECT_ROOT / "data" / "v4" / "final_evaluation_retrieval_cache_2000_v4.pkl"),
    )
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    split_path = config.work_data_dir / "calibration_split_v4.json"
    cache_path = config.work_data_dir / "evaluation_retrieval_cache_v4.pkl"
    source_split = json.loads(split_path.read_text(encoding="utf-8"))
    evaluation_ids = {str(value) for value in source_split["evaluation_ids"]}
    calibration_ids = {str(value) for value in source_split["calibration_ids"]}

    frame = pd.read_parquet(config.validation_file, columns=["id", "type", "level"])
    frame["id"] = frame["id"].astype(str)
    frame = frame[frame["id"].isin(evaluation_ids)]
    grouped = {
        question_type: frame.loc[frame["type"] == question_type, "id"].tolist()
        for question_type in ("bridge", "comparison")
    }

    rng = random.Random(args.seed)
    bridge_ids = rng.sample(grouped["bridge"], args.per_type)
    comparison_ids = rng.sample(grouped["comparison"], args.per_type)
    final_ids = bridge_ids + comparison_ids
    rng.shuffle(final_ids)

    rows_by_id = frame.set_index("id").to_dict("index")
    records = [
        {
            "id": example_id,
            "type": str(rows_by_id[example_id]["type"]),
            "level": str(rows_by_id[example_id]["level"]),
        }
        for example_id in final_ids
    ]
    output = {
        "dataset_version": "HotpotQA v4 Final 2000（HotpotQA 第四版最终两千样本）",
        "split_name": "Balanced Final Evaluation Split（平衡最终评估集）",
        "source_evaluation_examples": len(evaluation_ids),
        "seed": args.seed,
        "sampling_design": "1000 Bridge + 1000 Comparison（1000 条桥接型 + 1000 条比较型）",
        "total_examples": len(records),
        "type_counts": {"bridge": len(bridge_ids), "comparison": len(comparison_ids)},
        "evaluation_ids": final_ids,
        "examples": records,
        "external_model_calls": 0,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with cache_path.open("rb") as handle:
        source_cache = pickle.load(handle)
    final_cache = {example_id: source_cache[example_id] for example_id in final_ids}
    cache_output_path = Path(args.cache_output)
    with cache_output_path.open("wb") as handle:
        pickle.dump(final_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    selected_ids = set(final_ids)
    report = {
        "examples": len(final_ids),
        "unique_ids": len(selected_ids),
        "bridge": sum(row["type"] == "bridge" for row in records),
        "comparison": sum(row["type"] == "comparison" for row in records),
        "calibration_overlap": len(selected_ids & calibration_ids),
        "cache_examples": len(final_cache),
        "external_model_calls": 0,
        "split_file": str(output_path),
        "cache_file": str(cache_output_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
