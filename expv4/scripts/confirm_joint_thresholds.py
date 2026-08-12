from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import mean

# ruff: noqa: E402

import numpy as np
import pandas as pd


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from calibrate_bridge_threshold import _bootstrap_interval
from calibrate_structural_threshold import (
    METRIC_LABELS,
    _gold_facts,
    _load_split,
    _metrics_for_result,
)
from src.config import ExperimentConfig
from src.metrics.multihop_metrics import aggregate_metrics
from src.pipeline import AdaptiveRecoveryPipeline


DEFAULT_STRUCTURAL_THRESHOLDS = [0.41, 0.42, 0.43]
DEFAULT_BRIDGE_THRESHOLDS = [0.38, 0.40, 0.42]


def _summarize(
    pair: tuple[float, float],
    metric_rows: list[dict[str, float | None]],
    stat_rows: list[dict],
) -> dict:
    structural_threshold, bridge_threshold = pair
    metrics = aggregate_metrics(metric_rows)
    patterns = Counter(row["activation_pattern"] for row in stat_rows)
    examples = len(metric_rows)
    structural_activation = mean(
        float(row["structural_activated"]) for row in stat_rows
    )
    bridge_activation = mean(float(row["bridge_activated"]) for row in stat_rows)
    return {
        "structural_threshold": structural_threshold,
        "bridge_threshold": bridge_threshold,
        "examples": examples,
        "support_precision": metrics["support_precision"],
        "support_recall": metrics["support_recall"],
        "support_f1": metrics["support_f1"],
        "candidate_support_recall": metrics["candidate_support_recall"],
        "selected_support_recall": metrics["selected_support_recall"],
        "candidate_ccr": metrics["candidate_ccr"],
        "final_ccr": metrics["final_ccr"],
        "candidate_dccr": metrics["candidate_dccr"],
        "final_dccr": metrics["final_dccr"],
        "msfr": metrics["msfr"],
        "msdr": metrics["msdr"],
        "structural_activation_rate": structural_activation,
        "bridge_activation_rate": bridge_activation,
        "operator_activation_rate": structural_activation + bridge_activation,
        "no_recovery_rate": patterns["none"] / examples,
        "structural_only_rate": patterns["structural_only"] / examples,
        "bridge_only_rate": patterns["bridge_only"] / examples,
        "joint_recovery_rate": patterns["structural_bridge"] / examples,
        "average_structural_units": mean(
            row["structural_added_units"] for row in stat_rows
        ),
        "average_bridge_units": mean(row["bridge_added_units"] for row in stat_rows),
        "average_context_tokens": mean(
            row["selected_context_tokens"] for row in stat_rows
        ),
        "average_time_ms": mean(row["time_ms"] for row in stat_rows),
        "activation_counts": dict(sorted(patterns.items())),
    }


def _select_pair(
    summaries: list[dict],
    metric_rows: dict[tuple[float, float], list[dict[str, float | None]]],
    ccr_min: float,
    dccr_min: float,
    f1_margin: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict | None, list[dict]]:
    feasible = [
        row
        for row in summaries
        if row["final_ccr"] >= ccr_min and row["final_dccr"] >= dccr_min
    ]
    if not feasible:
        return None, []

    best_f1 = max(feasible, key=lambda row: row["support_f1"])
    best_pair = (
        best_f1["structural_threshold"],
        best_f1["bridge_threshold"],
    )
    best_values = np.asarray(
        [row["support_f1"] for row in metric_rows[best_pair]], dtype=float
    )
    rng = np.random.default_rng(seed)
    assessments = []
    non_inferior = []
    for row in feasible:
        pair = (row["structural_threshold"], row["bridge_threshold"])
        values = np.asarray(
            [item["support_f1"] for item in metric_rows[pair]], dtype=float
        )
        differences = values - best_values
        lower, upper = _bootstrap_interval(differences, bootstrap_samples, rng)
        accepted = lower >= -f1_margin
        assessments.append(
            {
                "structural_threshold": pair[0],
                "bridge_threshold": pair[1],
                "best_f1_pair": {
                    "structural_threshold": best_pair[0],
                    "bridge_threshold": best_pair[1],
                },
                "mean_support_f1_difference": float(differences.mean()),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "non_inferiority_margin": f1_margin,
                "non_inferior": accepted,
            }
        )
        if accepted:
            non_inferior.append(row)

    return min(
        non_inferior,
        key=lambda row: (
            row["average_context_tokens"],
            row["operator_activation_rate"],
            -row["support_f1"],
            row["structural_threshold"],
            row["bridge_threshold"],
        ),
    ), assessments


def _pairwise_bootstrap(
    selected_pair: tuple[float, float],
    pairs: list[tuple[float, float]],
    rows_by_pair: dict[tuple[float, float], list[dict[str, float | None]]],
    samples: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    comparisons = []
    for comparison_pair in pairs:
        if comparison_pair == selected_pair:
            continue
        for metric in ("support_f1", "final_ccr", "final_dccr"):
            selected_values = np.asarray(
                [row[metric] for row in rows_by_pair[selected_pair]], dtype=float
            )
            comparison_values = np.asarray(
                [row[metric] for row in rows_by_pair[comparison_pair]], dtype=float
            )
            differences = selected_values - comparison_values
            lower, upper = _bootstrap_interval(differences, samples, rng)
            comparisons.append(
                {
                    "selected_pair": {
                        "structural_threshold": selected_pair[0],
                        "bridge_threshold": selected_pair[1],
                    },
                    "comparison_pair": {
                        "structural_threshold": comparison_pair[0],
                        "bridge_threshold": comparison_pair[1],
                    },
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "mean_difference": float(differences.mean()),
                    "ci_95_lower": lower,
                    "ci_95_upper": upper,
                    "bootstrap_samples": samples,
                }
            )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local Joint Threshold Confirmation（局部门控阈值联合确认）"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument(
        "--structural-thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_STRUCTURAL_THRESHOLDS,
    )
    parser.add_argument(
        "--bridge-thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_BRIDGE_THRESHOLDS,
    )
    parser.add_argument("--ccr-min", type=float, default=0.70)
    parser.add_argument("--dccr-min", type=float, default=0.81)
    parser.add_argument("--f1-noninferiority-margin", type=float, default=0.005)
    parser.add_argument("--calibration-size", type=int, default=500)
    parser.add_argument("--run-size", type=int)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--retrieval-cache")
    parser.add_argument("--output")
    args = parser.parse_args()

    structural_thresholds = sorted(set(args.structural_thresholds))
    bridge_thresholds = sorted(set(args.bridge_thresholds))
    pairs = list(product(structural_thresholds, bridge_thresholds))
    config = ExperimentConfig.load(args.config)
    split = _load_split(config, args.calibration_size, args.seed)
    run_ids = split["calibration_ids"]
    if args.run_size is not None:
        if args.run_size <= 0:
            raise ValueError("run-size must be greater than 0")
        run_ids = run_ids[: args.run_size]

    frame = pd.read_parquet(config.validation_file)
    rows_by_id = {str(row["id"]): row for _, row in frame.iterrows()}
    cache_path = (
        Path(args.retrieval_cache)
        if args.retrieval_cache
        else config.work_data_dir / "calibration_retrieval_cache_v4.pkl"
    )
    with cache_path.open("rb") as handle:
        retrieval_cache = pickle.load(handle)
    pipeline = AdaptiveRecoveryPipeline(config, enable_generation=False)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output)
        if args.output
        else config.output_dir / f"joint_threshold_confirmation_{timestamp}.jsonl"
    )
    metrics_by_pair: dict[tuple[float, float], list[dict[str, float | None]]] = {
        pair: [] for pair in pairs
    }
    stats_by_pair: dict[tuple[float, float], list[dict]] = {pair: [] for pair in pairs}

    with output_path.open("w", encoding="utf-8") as output_handle:
        for example_index, example_id in enumerate(run_ids, start=1):
            row = rows_by_id[example_id]
            query = str(row["question"])
            gold_facts = _gold_facts(row["supporting_facts"])
            cached = retrieval_cache[example_id]
            retrieved = (
                cached["initial_evidence"],
                cached["query_vector"],
                cached["retrieval_time_ms"],
            )
            for structural_threshold, bridge_threshold in pairs:
                config.structural_gain_threshold = structural_threshold
                config.bridge_gain_threshold = bridge_threshold
                result = pipeline.run(query, "adaptive", retrieved)
                metrics = _metrics_for_result(result, gold_facts)
                pair = (structural_threshold, bridge_threshold)
                metrics_by_pair[pair].append(metrics)
                stats_by_pair[pair].append(result.stats)
                detail = {
                    "id": example_id,
                    "question": query,
                    "structural_threshold": structural_threshold,
                    "bridge_threshold": bridge_threshold,
                    "gold_supporting_facts": gold_facts,
                    "predicted_supporting_facts": result.supporting_facts,
                    "initial_evidence": [unit.key for unit in result.initial_evidence],
                    "candidate_evidence": [
                        unit.key for unit in result.candidate_evidence
                    ],
                    "context_evidence": [unit.key for unit in result.context_evidence],
                    "metrics": metrics,
                    "stats": result.stats,
                }
                output_handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
            if args.progress_every > 0 and example_index % args.progress_every == 0:
                print(
                    f"Completed {example_index}/{len(run_ids)} "
                    "examples（已完成局部联合确认样本）",
                    flush=True,
                )

    pair_table = [
        _summarize(pair, metrics_by_pair[pair], stats_by_pair[pair]) for pair in pairs
    ]
    selected, non_inferiority = _select_pair(
        pair_table,
        metrics_by_pair,
        args.ccr_min,
        args.dccr_min,
        args.f1_noninferiority_margin,
        args.bootstrap_samples,
        args.seed,
    )
    paired_bootstrap = []
    if selected is not None:
        selected_pair = (
            selected["structural_threshold"],
            selected["bridge_threshold"],
        )
        paired_bootstrap = _pairwise_bootstrap(
            selected_pair,
            pairs,
            metrics_by_pair,
            args.bootstrap_samples,
            args.seed,
        )

    summary_path = output_path.with_suffix(".summary.json")
    csv_path = output_path.with_suffix(".summary.csv")
    summary = {
        "experiment": "Local Joint Threshold Confirmation（局部门控阈值联合确认）",
        "dataset_version": split["dataset_version"],
        "seed": args.seed,
        "calibration_size": args.calibration_size,
        "executed_examples": len(run_ids),
        "structural_thresholds": structural_thresholds,
        "bridge_thresholds": bridge_thresholds,
        "fixed_parameters": {
            "second_hop_threshold": config.second_hop_gain_threshold,
            "selector": "Frozen current Evidence Selector（冻结当前证据选择器）",
            "embedding_model": config.embedding_model,
            "rerank_model": config.rerank_model,
        },
        "selection_rule": {
            "chain_constraints": {
                "final_ccr_min": args.ccr_min,
                "final_dccr_min": args.dccr_min,
            },
            "support_f1_noninferiority_margin": args.f1_noninferiority_margin,
            "lexicographic_objective": [
                "Retain Support F1 non-inferiority（保持支撑事实 F1 非劣效）",
                "Minimize Context Tokens（最小化上下文词元）",
                "Minimize Operator Activation（最小化算子激活）",
            ],
        },
        "selected_pair": {
            "structural_threshold": selected["structural_threshold"],
            "bridge_threshold": selected["bridge_threshold"],
        }
        if selected
        else None,
        "selected_result": selected,
        "metric_labels": METRIC_LABELS,
        "pair_table": pair_table,
        "non_inferiority_assessment": non_inferiority,
        "paired_bootstrap": paired_bootstrap,
        "split_file": str(config.work_data_dir / "calibration_split_v4.json"),
        "retrieval_cache": str(cache_path),
        "prediction_file": str(output_path),
        "csv_file": str(csv_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_rows = [
        {key: value for key, value in row.items() if key != "activation_counts"}
        for row in pair_table
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    if len(run_ids) == args.calibration_size and selected is not None:
        selection_path = config.output_dir / "joint_threshold_selection_final_v4.json"
        selection_path.write_text(
            json.dumps(
                {
                    "dataset_version": split["dataset_version"],
                    "selection": "Joint Threshold Selection（联合阈值选择结果）",
                    "seed": args.seed,
                    "calibration_size": args.calibration_size,
                    "selected_pair": summary["selected_pair"],
                    "second_hop_threshold_fixed_at": config.second_hop_gain_threshold,
                    "selection_rule": summary["selection_rule"],
                    "selected_result": selected,
                    "summary_file": str(summary_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        summary["selection_file"] = str(selection_path)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
