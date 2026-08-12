from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

# ruff: noqa: E402

import pandas as pd


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from src.config import ExperimentConfig
from src.metrics.multihop_metrics import (
    aggregate_metrics,
    evidence_coverage_scores,
    recovery_scores,
    supporting_fact_scores,
)
from src.pipeline import AdaptiveRecoveryPipeline


STRATEGY_LABELS = {
    "none": "No Recovery（不启用恢复）",
    "always_structural": "Always Structural（始终启用结构恢复）",
    "always_bridge": "Always Bridge（始终启用桥接恢复）",
    "always_both": "Always Both（始终启用两种恢复）",
    "adaptive": "Adaptive Recovery（自适应恢复）",
}

METRIC_LABELS = {
    "support_em": "Supporting Fact EM（支撑事实完全匹配）",
    "support_precision": "Supporting Fact Precision（支撑事实精确率）",
    "support_recall": "Supporting Fact Recall（支撑事实召回率）",
    "support_f1": "Supporting Fact F1（支撑事实 F1）",
    "msfr": "Missing Supporting Fact Recovery（缺失支撑事实恢复率）",
    "msdr": "Missing Supporting Document Recovery（缺失支撑文档恢复率）",
    "ccr": "Complete Chain Recall（完整证据链召回率）",
    "dccr": "Document-level Complete Chain Recall（文档级完整证据链召回率）",
    "initial_sentence_chain_complete": "Initial Sentence Chain Complete（初始句子证据链完整率）",
    "initial_document_chain_complete": "Initial Document Chain Complete（初始文档证据链完整率）",
    "average_context_tokens": "Average Context Tokens（平均上下文词元数）",
    "average_expanded_units": "Average Expanded Units（平均扩展证据单元数）",
    "average_time_ms": "Average Time（平均耗时，毫秒）",
}


def _gold_facts(value: dict) -> list[tuple[str, int]]:
    return [
        (str(title), int(sent_id))
        for title, sent_id in zip(value["title"], value["sent_id"])
    ]


def _write_split(
    frame: pd.DataFrame,
    output_path: Path,
    calibration_size: int,
    seed: int,
) -> dict:
    ids = [str(value) for value in frame["id"]]
    if calibration_size <= 0 or calibration_size >= len(ids):
        raise ValueError(f"calibration-size must be between 1 and {len(ids) - 1}")
    random.Random(seed).shuffle(ids)
    split = {
        "dataset_version": "HotpotQA v4（HotpotQA 第四版实验数据）",
        "split_name": "Calibration Split（校准集划分）",
        "seed": seed,
        "total_examples": len(ids),
        "calibration_size": calibration_size,
        "evaluation_size": len(ids) - calibration_size,
        "calibration_ids": ids[:calibration_size],
        "evaluation_ids": ids[calibration_size:],
    }
    output_path.write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return split


def _average(rows: list[dict], key: str) -> float:
    return mean(float(row[key]) for row in rows) if rows else 0.0


def _summarize_strategy(
    strategy: str,
    metric_rows: list[dict[str, float | None]],
    stat_rows: list[dict],
) -> dict:
    metrics = aggregate_metrics(metric_rows)
    activation_counts = Counter(row["activation_pattern"] for row in stat_rows)
    examples = len(metric_rows)
    return {
        "strategy": strategy,
        "strategy_label": STRATEGY_LABELS[strategy],
        "examples": examples,
        "support_em": metrics.get("support_em", 0.0),
        "support_precision": metrics.get("support_precision", 0.0),
        "support_recall": metrics.get("support_recall", 0.0),
        "support_f1": metrics.get("support_f1", 0.0),
        "msfr": metrics.get("msfr", 0.0),
        "msdr": metrics.get("msdr", 0.0),
        "ccr": metrics.get("ccr", 0.0),
        "dccr": metrics.get("dccr", 0.0),
        "initial_sentence_chain_complete": metrics.get(
            "initial_sentence_chain_complete", 0.0
        ),
        "initial_document_chain_complete": metrics.get(
            "initial_document_chain_complete", 0.0
        ),
        "average_context_tokens": _average(stat_rows, "selected_context_tokens"),
        "average_expanded_units": mean(
            row["structural_added_units"] + row["bridge_added_units"]
            for row in stat_rows
        )
        if stat_rows
        else 0.0,
        "average_time_ms": _average(stat_rows, "time_ms"),
        "structural_activation_rate": mean(
            float(row["structural_activated"]) for row in stat_rows
        )
        if stat_rows
        else 0.0,
        "bridge_activation_rate": mean(
            float(row["bridge_activated"]) for row in stat_rows
        )
        if stat_rows
        else 0.0,
        "second_hop_activation_rate": mean(
            float(row["second_bridge_hop"]) for row in stat_rows
        )
        if stat_rows
        else 0.0,
        "activation_counts": dict(sorted(activation_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate v4 recovery gating（校准第四版恢复门控）"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument("--calibration-size", type=int, default=500)
    parser.add_argument(
        "--run-size",
        type=int,
        help="Only run the first N calibration examples（仅运行前 N 个校准样本）",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGY_LABELS),
        default=list(STRATEGY_LABELS),
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    frame = pd.read_parquet(config.validation_file)
    split_path = config.work_data_dir / "calibration_split_v4.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        split = _write_split(
            frame,
            split_path,
            calibration_size=args.calibration_size,
            seed=args.seed,
        )
    cache_path = config.work_data_dir / "calibration_retrieval_cache_v4.pkl"
    with cache_path.open("rb") as handle:
        retrieval_cache = pickle.load(handle)

    run_ids = split["calibration_ids"]
    if args.run_size is not None:
        if args.run_size <= 0:
            raise ValueError("run-size must be greater than 0")
        run_ids = run_ids[: args.run_size]

    rows_by_id = {str(row["id"]): row for _, row in frame.iterrows()}
    pipeline = AdaptiveRecoveryPipeline(config, enable_generation=False)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output)
        if args.output
        else config.output_dir / f"gating_calibration_{timestamp}.jsonl"
    )

    metrics_by_strategy: dict[str, list[dict[str, float | None]]] = {
        strategy: [] for strategy in args.strategies
    }
    stats_by_strategy: dict[str, list[dict]] = {
        strategy: [] for strategy in args.strategies
    }

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
            results = pipeline.run_variants(query, args.strategies, retrieved)
            for strategy, result in results.items():
                metrics = {
                    **supporting_fact_scores(result.supporting_facts, gold_facts),
                    **evidence_coverage_scores(
                        [unit.key for unit in result.candidate_evidence],
                        gold_facts,
                        "candidate",
                    ),
                    **evidence_coverage_scores(
                        [unit.key for unit in result.context_evidence],
                        gold_facts,
                        "selected",
                    ),
                    **recovery_scores(
                        [unit.key for unit in result.initial_evidence],
                        [unit.key for unit in result.context_evidence],
                        gold_facts,
                    ),
                }
                metrics_by_strategy[strategy].append(metrics)
                stats_by_strategy[strategy].append(result.stats)
                detail = {
                    "id": example_id,
                    "question": query,
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy],
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
                handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
            if args.progress_every > 0 and example_index % args.progress_every == 0:
                print(
                    f"Completed {example_index}/{len(run_ids)} "
                    "examples（已完成校准样本）",
                    flush=True,
                )

    quality_cost_table = [
        _summarize_strategy(
            strategy,
            metrics_by_strategy[strategy],
            stats_by_strategy[strategy],
        )
        for strategy in args.strategies
    ]
    summary_path = output_path.with_suffix(".summary.json")
    csv_path = output_path.with_suffix(".summary.csv")
    summary = {
        "experiment": "Recovery Gating Calibration（恢复门控校准实验）",
        "dataset_version": split["dataset_version"],
        "seed": args.seed,
        "calibration_size": args.calibration_size,
        "executed_examples": len(run_ids),
        "strategies": [STRATEGY_LABELS[item] for item in args.strategies],
        "metric_labels": METRIC_LABELS,
        "quality_cost_table": quality_cost_table,
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
