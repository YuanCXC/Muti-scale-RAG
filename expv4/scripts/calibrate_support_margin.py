from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from src.config import ExperimentConfig
from src.metrics.multihop_metrics import (
    aggregate_metrics,
    evidence_coverage_scores,
    supporting_fact_scores,
)
from src.pipeline import AdaptiveRecoveryPipeline


DEFAULT_MARGINS = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.30]


def _gold_facts(value: dict) -> list[tuple[str, int]]:
    return [
        (str(title), int(sent_id))
        for title, sent_id in zip(value["title"], value["sent_id"])
    ]


def _bootstrap(
    selected: list[dict],
    comparison: list[dict],
    samples: int,
    seed: int,
) -> dict:
    differences = np.asarray(
        [a["support_f1"] - b["support_f1"] for a, b in zip(selected, comparison)],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        sample = rng.integers(0, len(differences), len(differences))
        means[index] = differences[sample].mean()
    return {
        "mean_support_f1_difference": float(differences.mean()),
        "ci_95_lower": float(np.quantile(means, 0.025)),
        "ci_95_upper": float(np.quantile(means, 0.975)),
        "bootstrap_samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Support Margin Calibration（支撑事实分数容差校准）"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument("--margins", nargs="+", type=float, default=DEFAULT_MARGINS)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()

    margins = sorted(set(args.margins))
    config = ExperimentConfig.load(args.config)
    split_path = config.work_data_dir / "calibration_split_v4.json"
    cache_path = config.work_data_dir / "calibration_retrieval_cache_v4.pkl"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    with cache_path.open("rb") as handle:
        retrieval_cache = pickle.load(handle)

    frame = pd.read_parquet(config.validation_file)
    rows_by_id = {str(row["id"]): row for _, row in frame.iterrows()}
    pipeline = AdaptiveRecoveryPipeline(config, enable_generation=False)
    metrics_by_margin: dict[float, list[dict]] = {margin: [] for margin in margins}
    counts_by_margin: dict[float, list[int]] = {margin: [] for margin in margins}

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output)
        if args.output
        else config.output_dir
        / f"support_margin_calibration_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    )
    with output_path.open("w", encoding="utf-8") as output:
        for example_index, example_id in enumerate(
            split["calibration_ids"], start=1
        ):
            row = rows_by_id[example_id]
            cached = retrieval_cache[example_id]
            retrieved = (
                cached["initial_evidence"],
                cached["query_vector"],
                cached["retrieval_time_ms"],
            )
            result = pipeline.run(str(row["question"]), "adaptive", retrieved)
            gold = _gold_facts(row["supporting_facts"])
            for margin in margins:
                config.support_margin = margin
                predicted = pipeline.support_predictor.predict(
                    result.context_evidence, cached["query_vector"]
                )
                metrics = {
                    **supporting_fact_scores(predicted, gold),
                    **evidence_coverage_scores(predicted, gold, "predicted"),
                }
                metrics_by_margin[margin].append(metrics)
                counts_by_margin[margin].append(len(predicted))
                output.write(
                    json.dumps(
                        {
                            "id": example_id,
                            "support_margin": margin,
                            "gold_supporting_facts": gold,
                            "predicted_supporting_facts": predicted,
                            "metrics": metrics,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if args.progress_every and example_index % args.progress_every == 0:
                print(
                    f"Completed {example_index}/500 examples"
                    "（已完成支撑事实容差校准样本）",
                    flush=True,
                )

    table = []
    for margin in margins:
        metrics = aggregate_metrics(metrics_by_margin[margin])
        table.append(
            {
                "support_margin": margin,
                "examples": len(metrics_by_margin[margin]),
                "support_em": metrics["support_em"],
                "support_precision": metrics["support_precision"],
                "support_recall": metrics["support_recall"],
                "support_f1": metrics["support_f1"],
                "predicted_ccr": metrics["predicted_ccr"],
                "predicted_dccr": metrics["predicted_dccr"],
                "average_predicted_facts": mean(counts_by_margin[margin]),
            }
        )
    selected = max(
        table,
        key=lambda row: (
            row["support_f1"],
            row["predicted_ccr"],
            row["support_precision"],
            -row["average_predicted_facts"],
        ),
    )
    selected_margin = selected["support_margin"]
    comparisons = []
    for margin in margins:
        if margin == selected_margin:
            continue
        comparisons.append(
            {
                "selected_margin": selected_margin,
                "comparison_margin": margin,
                **_bootstrap(
                    metrics_by_margin[selected_margin],
                    metrics_by_margin[margin],
                    args.bootstrap_samples,
                    args.seed,
                ),
            }
        )

    summary = {
        "experiment": "Support Margin Calibration（支撑事实分数容差校准）",
        "dataset_version": "HotpotQA v4（HotpotQA 第四版实验数据）",
        "examples": 500,
        "frozen_recovery_thresholds": {
            "structural_threshold": config.structural_gain_threshold,
            "bridge_threshold": config.bridge_gain_threshold,
        },
        "margins": margins,
        "selection_rule": [
            "Maximize Support F1（最大化支撑事实 F1）",
            "Maximize Predicted CCR（最大化预测完整链召回）",
            "Maximize Support Precision（最大化支撑事实精确率）",
            "Minimize Predicted Facts（最小化预测事实数量）",
        ],
        "selected_margin": selected_margin,
        "selected_result": selected,
        "margin_table": table,
        "paired_bootstrap": comparisons,
        "split_file": str(split_path),
        "retrieval_cache": str(cache_path),
        "prediction_file": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selection_path = config.output_dir / "support_margin_selection_final_v4.json"
    selection_path.write_text(
        json.dumps(
            {
                "selected_margin": selected_margin,
                "selected_result": selected,
                "summary_file": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
