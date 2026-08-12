from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from calibrate_gating import (
    METRIC_LABELS,
    STRATEGY_LABELS,
    _gold_facts,
    _summarize_strategy,
)
from src.config import ExperimentConfig
from src.metrics.multihop_metrics import (
    aggregate_metrics,
    evidence_coverage_scores,
    recovery_scores,
    supporting_fact_scores,
)
from src.pipeline import AdaptiveRecoveryPipeline


STRATEGIES = list(STRATEGY_LABELS)
BOOTSTRAP_METRICS = {
    "support_f1": "Supporting Fact F1（支撑事实 F1）",
    "ccr": "Complete Chain Recall（完整证据链召回率）",
    "dccr": "Document-level Complete Chain Recall（文档级完整证据链召回率）",
}


def _metrics_for_result(result, gold_facts: list[tuple[str, int]]) -> dict:
    candidate_facts = [unit.key for unit in result.candidate_evidence]
    context_facts = [unit.key for unit in result.context_evidence]
    return {
        **supporting_fact_scores(result.supporting_facts, gold_facts),
        **evidence_coverage_scores(candidate_facts, gold_facts, "candidate"),
        **evidence_coverage_scores(context_facts, gold_facts, "selected"),
        **recovery_scores(
            [unit.key for unit in result.initial_evidence],
            context_facts,
            gold_facts,
        ),
    }


def _paired_bootstrap(
    metrics_by_strategy: dict[str, list[dict]],
    samples: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    metric_names = list(BOOTSTRAP_METRICS)
    adaptive = np.asarray(
        [
            [float(row[metric]) for metric in metric_names]
            for row in metrics_by_strategy["adaptive"]
        ],
        dtype=float,
    )
    comparisons = []
    for baseline in STRATEGIES:
        if baseline == "adaptive":
            continue
        baseline_values = np.asarray(
            [
                [float(row[metric]) for metric in metric_names]
                for row in metrics_by_strategy[baseline]
            ],
            dtype=float,
        )
        differences = adaptive - baseline_values
        bootstrap_means = np.empty((samples, len(metric_names)), dtype=float)
        for start in range(0, samples, 256):
            stop = min(start + 256, samples)
            indices = rng.integers(
                0,
                len(differences),
                size=(stop - start, len(differences)),
            )
            bootstrap_means[start:stop] = differences[indices].mean(axis=1)
        for metric_index, metric in enumerate(metric_names):
            lower, upper = np.quantile(
                bootstrap_means[:, metric_index], [0.025, 0.975]
            )
            if lower > 0:
                conclusion = "Adaptive Higher（自适应策略更高）"
            elif upper < 0:
                conclusion = "Adaptive Lower（自适应策略更低）"
            else:
                conclusion = "No Clear Difference（无明确差异）"
            comparisons.append(
                {
                    "adaptive_strategy": STRATEGY_LABELS["adaptive"],
                    "baseline_strategy": STRATEGY_LABELS[baseline],
                    "metric": metric,
                    "metric_label": BOOTSTRAP_METRICS[metric],
                    "mean_difference": float(differences[:, metric_index].mean()),
                    "ci_95_lower": float(lower),
                    "ci_95_upper": float(upper),
                    "bootstrap_samples": samples,
                    "conclusion": conclusion,
                }
            )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final Evidence Evaluation（最终证据评估）"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--output")
    parser.add_argument("--split-file")
    parser.add_argument(
        "--split-part",
        choices=("calibration_ids", "evaluation_ids"),
        default="evaluation_ids",
    )
    parser.add_argument("--retrieval-cache")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    split_path = (
        Path(args.split_file)
        if args.split_file
        else config.work_data_dir / "calibration_split_v4.json"
    )
    cache_path = (
        Path(args.retrieval_cache)
        if args.retrieval_cache
        else config.work_data_dir / "evaluation_retrieval_cache_v4.pkl"
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    evaluation_ids = [str(value) for value in split[args.split_part]]
    with cache_path.open("rb") as handle:
        retrieval_cache = pickle.load(handle)
    if set(retrieval_cache) != set(evaluation_ids):
        raise ValueError(
            "Evaluation retrieval cache does not match evaluation split"
            "（评估检索缓存与评估集不一致）"
        )
    if any(
        unit.metadata.get("rerank_error")
        for item in retrieval_cache.values()
        for unit in item["initial_evidence"]
    ):
        raise ValueError(
            "Evaluation retrieval cache contains rerank errors"
            "（评估检索缓存仍包含重排错误）"
        )

    frame = pd.read_parquet(config.validation_file)
    rows_by_id = {str(row["id"]): row for _, row in frame.iterrows()}
    pipeline = AdaptiveRecoveryPipeline(config, enable_generation=False)
    output_path = (
        Path(args.output)
        if args.output
        else config.output_dir / "final_evidence_evaluation_v4.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_by_strategy: dict[str, list[dict]] = {
        strategy: [] for strategy in STRATEGIES
    }
    stats_by_strategy: dict[str, list[dict]] = {
        strategy: [] for strategy in STRATEGIES
    }
    started_at = time.perf_counter()

    with output_path.open("w", encoding="utf-8") as output_handle:
        for example_index, example_id in enumerate(evaluation_ids, start=1):
            row = rows_by_id[example_id]
            query = str(row["question"])
            gold_facts = _gold_facts(row["supporting_facts"])
            cached = retrieval_cache[example_id]
            retrieved = (
                cached["initial_evidence"],
                cached["query_vector"],
                cached["retrieval_time_ms"],
            )
            results = pipeline.run_variants(query, STRATEGIES, retrieved)
            for strategy, result in results.items():
                metrics = _metrics_for_result(result, gold_facts)
                metrics_by_strategy[strategy].append(metrics)
                stats_by_strategy[strategy].append(result.stats)
                detail = {
                    "id": example_id,
                    "question": query,
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy],
                    "gold_supporting_facts": gold_facts,
                    "predicted_supporting_facts": result.supporting_facts,
                    "initial_evidence": [
                        unit.key for unit in result.initial_evidence
                    ],
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
                    f"Completed {example_index}/{len(evaluation_ids)} examples"
                    "（已完成最终证据评估）",
                    flush=True,
                )

    quality_cost_table = []
    for strategy in STRATEGIES:
        strategy_summary = _summarize_strategy(
            strategy,
            metrics_by_strategy[strategy],
            stats_by_strategy[strategy],
        )
        aggregated = aggregate_metrics(metrics_by_strategy[strategy])
        strategy_summary.update(
            {
                "candidate_support_recall": aggregated[
                    "candidate_support_recall"
                ],
                "selected_support_recall": aggregated[
                    "selected_support_recall"
                ],
                "candidate_ccr": aggregated["candidate_ccr"],
                "selected_ccr": aggregated["selected_ccr"],
                "candidate_dccr": aggregated["candidate_dccr"],
                "selected_dccr": aggregated["selected_dccr"],
            }
        )
        strategy_summary["average_recovery_time_ms"] = float(
            np.mean(
                [row["recovery_time_ms"] for row in stats_by_strategy[strategy]]
            )
        )
        quality_cost_table.append(strategy_summary)

    paired_bootstrap = _paired_bootstrap(
        metrics_by_strategy,
        args.bootstrap_samples,
        args.seed,
    )
    elapsed_seconds = time.perf_counter() - started_at
    summary_path = output_path.with_suffix(".summary.json")
    csv_path = output_path.with_suffix(".summary.csv")
    summary = {
        "experiment": "Final Evidence Evaluation（最终证据评估）",
        "dataset_version": split["dataset_version"],
        "split_part": args.split_part,
        "split": "Fixed Evaluation Split（固定评估集）",
        "evaluation_examples": len(evaluation_ids),
        "strategy_runs": len(evaluation_ids) * len(STRATEGIES),
        "strategies": [STRATEGY_LABELS[strategy] for strategy in STRATEGIES],
        "frozen_parameters": {
            "structural_gain_threshold": config.structural_gain_threshold,
            "bridge_gain_threshold": config.bridge_gain_threshold,
            "second_hop_gain_threshold": config.second_hop_gain_threshold,
            "support_margin": config.support_margin,
            "context_budget": config.context_budget,
            "max_context_units": config.max_context_units,
            "max_bridge_hops": config.max_bridge_hops,
        },
        "external_model_calls": 0,
        "metric_labels": METRIC_LABELS,
        "quality_cost_table": quality_cost_table,
        "paired_bootstrap": paired_bootstrap,
        "bootstrap_seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "split_file": str(split_path),
        "retrieval_cache": str(cache_path),
        "prediction_file": str(output_path),
        "csv_file": str(csv_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(quality_cost_table).drop(columns="activation_counts").to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
