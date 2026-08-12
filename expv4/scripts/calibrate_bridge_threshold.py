from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

# ruff: noqa: E402

import numpy as np
import pandas as pd


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from calibrate_structural_threshold import (
    METRIC_LABELS,
    _gold_facts,
    _load_split,
    _metrics_for_result,
)
from src.config import ExperimentConfig
from src.metrics.multihop_metrics import aggregate_metrics
from src.pipeline import AdaptiveRecoveryPipeline


DEFAULT_THRESHOLDS = [0.32, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]


def _summarize(
    threshold: float,
    metric_rows: list[dict[str, float | None]],
    stat_rows: list[dict],
) -> dict:
    metrics = aggregate_metrics(metric_rows)
    patterns = Counter(row["activation_pattern"] for row in stat_rows)
    examples = len(metric_rows)
    return {
        "bridge_threshold": threshold,
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
        "structural_activation_rate": mean(
            float(row["structural_activated"]) for row in stat_rows
        ),
        "bridge_activation_rate": mean(
            float(row["bridge_activated"]) for row in stat_rows
        ),
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


def _bootstrap_interval(
    differences: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    bootstrap_means = np.empty(samples, dtype=float)
    for sample_index in range(samples):
        indices = rng.integers(0, len(differences), len(differences))
        bootstrap_means[sample_index] = differences[indices].mean()
    return (
        float(np.quantile(bootstrap_means, 0.025)),
        float(np.quantile(bootstrap_means, 0.975)),
    )


def _select_threshold(
    summaries: list[dict],
    metric_rows: dict[float, list[dict[str, float | None]]],
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
    best_threshold = best_f1["bridge_threshold"]
    best_values = np.asarray(
        [row["support_f1"] for row in metric_rows[best_threshold]], dtype=float
    )
    rng = np.random.default_rng(seed)
    assessments = []
    non_inferior = []
    for row in feasible:
        threshold = row["bridge_threshold"]
        values = np.asarray(
            [item["support_f1"] for item in metric_rows[threshold]], dtype=float
        )
        differences = values - best_values
        lower, upper = _bootstrap_interval(differences, bootstrap_samples, rng)
        accepted = lower >= -f1_margin
        assessments.append(
            {
                "threshold": threshold,
                "best_f1_threshold": best_threshold,
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
            row["bridge_activation_rate"],
            -row["support_f1"],
            row["bridge_threshold"],
        ),
    ), assessments


def _paired_bootstrap(
    selected_threshold: float,
    thresholds: list[float],
    rows_by_threshold: dict[float, list[dict[str, float | None]]],
    samples: int,
    seed: int,
) -> list[dict]:
    selected_index = thresholds.index(selected_threshold)
    neighbor_indices = {
        index
        for index in (selected_index - 1, selected_index + 1)
        if 0 <= index < len(thresholds)
    }
    rng = np.random.default_rng(seed)
    comparisons = []
    for neighbor_index in sorted(neighbor_indices):
        neighbor = thresholds[neighbor_index]
        for metric in ("support_f1", "final_ccr", "final_dccr", "msdr"):
            paired_values = [
                (selected_row[metric], neighbor_row[metric])
                for selected_row, neighbor_row in zip(
                    rows_by_threshold[selected_threshold],
                    rows_by_threshold[neighbor],
                )
                if selected_row[metric] is not None and neighbor_row[metric] is not None
            ]
            selected_values = np.asarray(
                [item[0] for item in paired_values], dtype=float
            )
            neighbor_values = np.asarray(
                [item[1] for item in paired_values], dtype=float
            )
            differences = selected_values - neighbor_values
            lower, upper = _bootstrap_interval(differences, samples, rng)
            comparisons.append(
                {
                    "selected_threshold": selected_threshold,
                    "neighbor_threshold": neighbor,
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
        description="Bridge Threshold Calibration（桥接阈值校准）"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS
    )
    parser.add_argument("--structural-threshold", type=float, default=0.42)
    parser.add_argument("--ccr-min", type=float, default=0.70)
    parser.add_argument("--dccr-min", type=float, default=0.81)
    parser.add_argument("--f1-noninferiority-margin", type=float, default=0.005)
    parser.add_argument("--calibration-size", type=int, default=500)
    parser.add_argument("--run-size", type=int)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output")
    parser.add_argument(
        "--retrieval-cache",
        help="Calibration Retrieval Cache（校准检索缓存）",
    )
    parser.add_argument(
        "--input-results",
        help="Reuse an existing JSONL result（复用已有 JSONL 结果，不调用模型）",
    )
    args = parser.parse_args()

    thresholds = sorted(set(args.thresholds))
    config = ExperimentConfig.load(args.config)
    config.structural_gain_threshold = args.structural_threshold
    split = _load_split(config, args.calibration_size, args.seed)
    run_ids = split["calibration_ids"]
    if args.run_size is not None:
        if args.run_size <= 0:
            raise ValueError("run-size must be greater than 0")
        run_ids = run_ids[: args.run_size]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.input_results:
        output_path = Path(args.input_results)
    else:
        output_path = (
            Path(args.output)
            if args.output
            else config.output_dir / f"bridge_threshold_calibration_{timestamp}.jsonl"
        )
    metrics_by_threshold: dict[float, list[dict[str, float | None]]] = {
        threshold: [] for threshold in thresholds
    }
    stats_by_threshold: dict[float, list[dict]] = {
        threshold: [] for threshold in thresholds
    }

    if args.input_results:
        observed_ids = set()
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                detail = json.loads(line)
                threshold = float(detail["bridge_threshold"])
                if threshold not in metrics_by_threshold:
                    continue
                observed_ids.add(str(detail["id"]))
                metrics_by_threshold[threshold].append(detail["metrics"])
                stats_by_threshold[threshold].append(detail["stats"])
        run_ids = sorted(observed_ids)
    else:
        frame = pd.read_parquet(config.validation_file)
        rows_by_id = {str(row["id"]): row for _, row in frame.iterrows()}
        pipeline = AdaptiveRecoveryPipeline(config, enable_generation=False)
        cache_path = (
            Path(args.retrieval_cache)
            if args.retrieval_cache
            else config.work_data_dir / "calibration_retrieval_cache_v4.pkl"
        )
        with cache_path.open("rb") as cache_handle:
            retrieval_cache = pickle.load(cache_handle)
        with output_path.open("w", encoding="utf-8") as handle:
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
                for threshold in thresholds:
                    config.bridge_gain_threshold = threshold
                    result = pipeline.run(query, "adaptive", retrieved)
                    metrics = _metrics_for_result(result, gold_facts)
                    metrics_by_threshold[threshold].append(metrics)
                    stats_by_threshold[threshold].append(result.stats)
                    detail = {
                        "id": example_id,
                        "question": query,
                        "structural_threshold": args.structural_threshold,
                        "bridge_threshold": threshold,
                        "gold_supporting_facts": gold_facts,
                        "predicted_supporting_facts": result.supporting_facts,
                        "initial_evidence": [
                            unit.key for unit in result.initial_evidence
                        ],
                        "candidate_evidence": [
                            unit.key for unit in result.candidate_evidence
                        ],
                        "context_evidence": [
                            unit.key for unit in result.context_evidence
                        ],
                        "metrics": metrics,
                        "stats": result.stats,
                    }
                    handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
                if args.progress_every > 0 and example_index % args.progress_every == 0:
                    print(
                        f"Completed {example_index}/{len(run_ids)} "
                        "examples（已完成桥接阈值校准样本）",
                        flush=True,
                    )

    threshold_table = [
        _summarize(
            threshold,
            metrics_by_threshold[threshold],
            stats_by_threshold[threshold],
        )
        for threshold in thresholds
    ]
    selected, non_inferiority = _select_threshold(
        threshold_table,
        metrics_by_threshold,
        args.ccr_min,
        args.dccr_min,
        args.f1_noninferiority_margin,
        args.bootstrap_samples,
        args.seed,
    )
    bootstrap = []
    if selected is not None:
        bootstrap = _paired_bootstrap(
            selected["bridge_threshold"],
            thresholds,
            metrics_by_threshold,
            args.bootstrap_samples,
            args.seed,
        )

    summary_path = output_path.with_suffix(".summary.json")
    csv_path = output_path.with_suffix(".summary.csv")
    summary = {
        "experiment": "Bridge Threshold Calibration（桥接阈值校准实验）",
        "dataset_version": split["dataset_version"],
        "seed": args.seed,
        "calibration_size": args.calibration_size,
        "executed_examples": len(run_ids),
        "thresholds": thresholds,
        "fixed_parameters": {
            "structural_threshold": args.structural_threshold,
            "second_hop_threshold": config.second_hop_gain_threshold,
            "selector": "Frozen current Evidence Selector（冻结当前证据选择器）",
            "embedding_model": config.embedding_model,
            "rerank_model": config.rerank_model,
            "context_budget": config.context_budget,
            "max_context_units": config.max_context_units,
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
                "Minimize Bridge Activation（最小化桥接恢复激活率）",
            ],
        },
        "selected_threshold": selected["bridge_threshold"] if selected else None,
        "selected_result": selected,
        "metric_labels": METRIC_LABELS,
        "threshold_table": threshold_table,
        "non_inferiority_assessment": non_inferiority,
        "paired_bootstrap": bootstrap,
        "split_file": str(config.work_data_dir / "calibration_split_v4.json"),
        "retrieval_cache": str(
            Path(args.retrieval_cache)
            if args.retrieval_cache
            else config.work_data_dir / "calibration_retrieval_cache_v4.pkl"
        ),
        "prediction_file": str(output_path),
        "csv_file": str(csv_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_rows = [
        {key: value for key, value in row.items() if key != "activation_counts"}
        for row in threshold_table
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    if len(run_ids) == args.calibration_size and selected is not None:
        selection_path = config.output_dir / "bridge_threshold_selection_final_v4.json"
        selection_path.write_text(
            json.dumps(
                {
                    "dataset_version": split["dataset_version"],
                    "selection": "Bridge Threshold Selection（桥接阈值选择结果）",
                    "seed": args.seed,
                    "calibration_size": args.calibration_size,
                    "selected_threshold": selected["bridge_threshold"],
                    "structural_threshold_fixed_at": args.structural_threshold,
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
