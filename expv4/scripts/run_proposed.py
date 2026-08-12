from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

# ruff: noqa: E402

import pandas as pd


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from src.config import ExperimentConfig
from src.metrics.multihop_metrics import aggregate_metrics, evaluate_prediction
from src.models import EvidenceUnit
from src.pipeline import AdaptiveRecoveryPipeline


def _gold_facts(value: dict) -> list[tuple[str, int]]:
    return [
        (str(title), int(sent_id))
        for title, sent_id in zip(value["title"], value["sent_id"])
    ]


def _unit_to_dict(unit: EvidenceUnit) -> dict:
    return {
        "title": unit.title,
        "sent_id": unit.sent_id,
        "text": unit.text,
        "score": unit.score,
        "source": unit.source,
        "metadata": unit.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the v4 proposed adaptive recovery method"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    pipeline = AdaptiveRecoveryPipeline(
        config, enable_generation=not args.skip_generation
    )
    frame = pd.read_parquet(config.validation_file)
    subset = frame.iloc[args.start_index : args.start_index + args.sample_size]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output)
        if args.output
        else config.output_dir / f"proposed_{timestamp}.jsonl"
    )

    metric_rows = []
    stat_rows = []
    activation_counts: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as handle:
        for _, row in subset.iterrows():
            result = pipeline.run(str(row["question"]))
            gold_facts = _gold_facts(row["supporting_facts"])
            metrics = evaluate_prediction(
                result.answer,
                result.supporting_facts,
                [unit.key for unit in result.initial_evidence],
                [unit.key for unit in result.context_evidence],
                str(row["answer"]),
                gold_facts,
            )
            metric_rows.append(metrics)
            stat_rows.append(result.stats)
            pattern = result.stats["activation_pattern"]
            activation_counts[pattern] = activation_counts.get(pattern, 0) + 1

            item = {
                "id": str(row["id"]),
                "question": str(row["question"]),
                "gold_answer": str(row["answer"]),
                "predicted_answer": result.answer,
                "gold_supporting_facts": gold_facts,
                "predicted_supporting_facts": result.supporting_facts,
                "initial_evidence": [
                    _unit_to_dict(unit) for unit in result.initial_evidence
                ],
                "context_evidence": [
                    _unit_to_dict(unit) for unit in result.context_evidence
                ],
                "stats": result.stats,
                "metrics": metrics,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "examples": len(metric_rows),
        "activation_counts": activation_counts,
        "metrics": aggregate_metrics(metric_rows),
        "average_context_tokens": mean(
            row["selected_context_tokens"] for row in stat_rows
        )
        if stat_rows
        else 0.0,
        "average_expanded_units": mean(
            row["structural_added_units"] + row["bridge_added_units"]
            for row in stat_rows
        )
        if stat_rows
        else 0.0,
        "average_time_ms": mean(row["time_ms"] for row in stat_rows)
        if stat_rows
        else 0.0,
        "prediction_file": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
