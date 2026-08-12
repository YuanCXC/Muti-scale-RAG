from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from calibrate_gating import _gold_facts
from src.config import ExperimentConfig
from src.metrics.multihop_metrics import aggregate_metrics
from src.pipeline import AdaptiveRecoveryPipeline
from run_final_evidence_evaluation import _metrics_for_result


VARIANT_LABELS = {
    "full_adaptive": "Full Adaptive Recovery（完整自适应恢复）",
    "without_structural": "Without Structural Recovery（移除结构恢复）",
    "without_bridge": "Without Bridge Recovery（移除桥接恢复）",
    "one_hop_only": "One-hop Bridge Only（仅保留一跳桥接）",
    "two_hop_extension": "Two-hop Bridge Extension（第二跳桥接扩展）",
    "without_semantic_validation": (
        "Without LLM Semantic Validation（移除大模型语义验证）"
    ),
}

BOOTSTRAP_METRICS = {
    "support_f1": "Supporting Fact F1（支撑事实 F1）",
    "ccr": "Complete Chain Recall（完整证据链召回率）",
    "dccr": "Document-level Complete Chain Recall（文档级完整证据链召回率）",
}


def _summarize(variant: str, metric_rows: list[dict], stat_rows: list[dict]) -> dict:
    metrics = aggregate_metrics(metric_rows)
    patterns = Counter(row["activation_pattern"] for row in stat_rows)
    return {
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "examples": len(metric_rows),
        "support_em": metrics["support_em"],
        "support_precision": metrics["support_precision"],
        "support_recall": metrics["support_recall"],
        "support_f1": metrics["support_f1"],
        "candidate_support_recall": metrics["candidate_support_recall"],
        "selected_support_recall": metrics["selected_support_recall"],
        "candidate_ccr": metrics["candidate_ccr"],
        "ccr": metrics["ccr"],
        "candidate_dccr": metrics["candidate_dccr"],
        "dccr": metrics["dccr"],
        "msfr": metrics["msfr"],
        "msdr": metrics["msdr"],
        "average_context_tokens": mean(
            row["selected_context_tokens"] for row in stat_rows
        ),
        "average_expanded_units": mean(
            row["structural_added_units"] + row["bridge_added_units"]
            for row in stat_rows
        ),
        "average_recovery_time_ms": mean(
            row["recovery_time_ms"] for row in stat_rows
        ),
        "structural_activation_rate": mean(
            float(row["structural_activated"]) for row in stat_rows
        ),
        "bridge_activation_rate": mean(
            float(row["bridge_activated"]) for row in stat_rows
        ),
        "second_hop_activation_rate": mean(
            float(row["second_bridge_hop"]) for row in stat_rows
        ),
        "activation_counts": dict(sorted(patterns.items())),
    }


def _paired_bootstrap(
    rows_by_variant: dict[str, list[dict]],
    samples: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    metric_names = list(BOOTSTRAP_METRICS)
    full = np.asarray(
        [
            [float(row[metric]) for metric in metric_names]
            for row in rows_by_variant["full_adaptive"]
        ],
        dtype=float,
    )
    comparisons = []
    for ablation in VARIANT_LABELS:
        if ablation == "full_adaptive":
            continue
        ablated = np.asarray(
            [
                [float(row[metric]) for metric in metric_names]
                for row in rows_by_variant[ablation]
            ],
            dtype=float,
        )
        differences = full - ablated
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
                conclusion = "Full Method Higher（完整方法更高）"
            elif upper < 0:
                conclusion = "Full Method Lower（完整方法更低）"
            else:
                conclusion = "No Clear Difference（无明确差异）"
            comparisons.append(
                {
                    "ablation": ablation,
                    "ablation_label": VARIANT_LABELS[ablation],
                    "metric": metric,
                    "metric_label": BOOTSTRAP_METRICS[metric],
                    "mean_full_minus_ablation": float(
                        differences[:, metric_index].mean()
                    ),
                    "ci_95_lower": float(lower),
                    "ci_95_upper": float(upper),
                    "bootstrap_samples": samples,
                    "conclusion": conclusion,
                }
            )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen Ablation Study（冻结参数消融实验）"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress-every", type=int, default=500)
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
    raw_bridge_index_path = config.raw_bridge_index_file
    split = json.loads(split_path.read_text(encoding="utf-8"))
    evaluation_ids = [str(value) for value in split[args.split_part]]
    with cache_path.open("rb") as handle:
        retrieval_cache = pickle.load(handle)
    if set(retrieval_cache) != set(evaluation_ids):
        raise ValueError(
            "Evaluation cache does not match fixed split"
            "（评估缓存与固定评估集不一致）"
        )

    frame = pd.read_parquet(config.validation_file)
    rows_by_id = {str(row["id"]): row for _, row in frame.iterrows()}
    full_pipeline = AdaptiveRecoveryPipeline(config, enable_generation=False)
    raw_config = deepcopy(config)
    raw_config.bridge_index_path = str(raw_bridge_index_path)
    raw_pipeline = AdaptiveRecoveryPipeline(raw_config, enable_generation=False)

    structural_threshold = config.structural_gain_threshold
    bridge_threshold = config.bridge_gain_threshold
    max_bridge_hops = config.max_bridge_hops
    variant_settings = {
        "full_adaptive": (structural_threshold, bridge_threshold, max_bridge_hops),
        "without_structural": (float("inf"), bridge_threshold, max_bridge_hops),
        "without_bridge": (structural_threshold, float("inf"), max_bridge_hops),
        "one_hop_only": (structural_threshold, bridge_threshold, 1),
        "two_hop_extension": (structural_threshold, bridge_threshold, 2),
    }

    output_path = (
        Path(args.output)
        if args.output
        else config.output_dir / "frozen_ablation_study_v4.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_by_variant: dict[str, list[dict]] = {
        variant: [] for variant in VARIANT_LABELS
    }
    stats_by_variant: dict[str, list[dict]] = {
        variant: [] for variant in VARIANT_LABELS
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
            results = {}
            for variant, settings in variant_settings.items():
                (
                    config.structural_gain_threshold,
                    config.bridge_gain_threshold,
                    config.max_bridge_hops,
                ) = settings
                results[variant] = full_pipeline.run(query, "adaptive", retrieved)
            results["without_semantic_validation"] = raw_pipeline.run(
                query, "adaptive", retrieved
            )

            for variant, result in results.items():
                metrics = _metrics_for_result(result, gold_facts)
                metrics_by_variant[variant].append(metrics)
                stats_by_variant[variant].append(result.stats)
                output_handle.write(
                    json.dumps(
                        {
                            "id": example_id,
                            "question": query,
                            "variant": variant,
                            "variant_label": VARIANT_LABELS[variant],
                            "gold_supporting_facts": gold_facts,
                            "predicted_supporting_facts": result.supporting_facts,
                            "candidate_evidence": [
                                unit.key for unit in result.candidate_evidence
                            ],
                            "context_evidence": [
                                unit.key for unit in result.context_evidence
                            ],
                            "metrics": metrics,
                            "stats": result.stats,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if args.progress_every > 0 and example_index % args.progress_every == 0:
                print(
                    f"Completed {example_index}/{len(evaluation_ids)} examples"
                    "（已完成冻结消融实验）",
                    flush=True,
                )

    config.structural_gain_threshold = structural_threshold
    config.bridge_gain_threshold = bridge_threshold
    config.max_bridge_hops = max_bridge_hops
    ablation_table = [
        _summarize(
            variant,
            metrics_by_variant[variant],
            stats_by_variant[variant],
        )
        for variant in VARIANT_LABELS
    ]
    paired_bootstrap = _paired_bootstrap(
        metrics_by_variant,
        args.bootstrap_samples,
        args.seed,
    )
    semantic_edges = sum(len(links) for links in full_pipeline.bridge_index.values())
    raw_edges = sum(len(links) for links in raw_pipeline.bridge_index.values())
    summary_path = output_path.with_suffix(".summary.json")
    csv_path = output_path.with_suffix(".summary.csv")
    summary = {
        "experiment": "Frozen Ablation Study（冻结参数消融实验）",
        "dataset_version": split["dataset_version"],
        "split_part": args.split_part,
        "evaluation_examples": len(evaluation_ids),
        "variant_runs": len(evaluation_ids) * len(VARIANT_LABELS),
        "frozen_parameters": {
            "structural_gain_threshold": structural_threshold,
            "bridge_gain_threshold": bridge_threshold,
            "second_hop_gain_threshold": config.second_hop_gain_threshold,
            "support_margin": config.support_margin,
            "context_budget": config.context_budget,
            "max_context_units": config.max_context_units,
            "max_bridge_hops": config.max_bridge_hops,
        },
        "semantic_validation_ablation": {
            "raw_bridge_index": str(raw_bridge_index_path),
            "raw_bridge_edges": raw_edges,
            "semantic_bridge_index": str(config.bridge_index_file),
            "semantic_bridge_edges": semantic_edges,
        },
        "external_model_calls": 0,
        "ablation_table": ablation_table,
        "paired_bootstrap": paired_bootstrap,
        "bootstrap_seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "split_file": str(split_path),
        "retrieval_cache": str(cache_path),
        "prediction_file": str(output_path),
        "csv_file": str(csv_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_rows = [
        {key: value for key, value in row.items() if key != "activation_counts"}
        for row in ablation_table
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
