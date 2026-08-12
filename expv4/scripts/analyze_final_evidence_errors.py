from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


EXPV4_ROOT = Path(__file__).resolve().parents[1]

STRATEGIES = [
    "none",
    "always_structural",
    "always_bridge",
    "always_both",
    "adaptive",
]

STRATEGY_LABELS = {
    "none": "No Recovery（无需恢复）",
    "always_structural": "Always Structural（始终结构恢复）",
    "always_bridge": "Always Bridge（始终桥接恢复）",
    "always_both": "Always Both（始终联合恢复）",
    "adaptive": "Adaptive Recovery（自适应恢复）",
}

ROUTE_LABELS = {
    "none": "No Recovery（无需恢复）",
    "structural_only": "Structural Only（仅结构恢复）",
    "bridge_only": "Bridge Only（仅桥接恢复）",
    "structural_bridge": "Joint Recovery（联合恢复）",
}

FAILURE_LABELS = {
    "candidate_missing": "Candidate Missing（候选阶段仍缺证据）",
    "selector_drop": "Selector Drop（选择器丢失已找回证据）",
    "support_prediction_error": "Support Prediction Error（支撑事实预测错误）",
    "support_exact_success": "Exact Support Success（支撑事实完全正确）",
}

SUPPORT_ERROR_LABELS = {
    "overprediction_only": "Overprediction Only（仅误选多余事实）",
    "underprediction_only": "Underprediction Only（仅漏选金标事实）",
    "mixed_prediction_error": "Mixed Prediction Error（同时误选和漏选）",
}

ROUTE_METRICS = [
    "support_em",
    "support_precision",
    "support_recall",
    "support_f1",
    "candidate_support_recall",
    "selected_support_recall",
    "candidate_ccr",
    "selected_ccr",
    "candidate_dccr",
    "selected_dccr",
]


def _failure_stage(adaptive: dict) -> str:
    metrics = adaptive["metrics"]
    if metrics["candidate_ccr"] < 1.0:
        return "candidate_missing"
    if metrics["selected_ccr"] < 1.0:
        return "selector_drop"
    if metrics["support_em"] < 1.0:
        return "support_prediction_error"
    return "support_exact_success"


def _support_error_type(adaptive: dict) -> str:
    gold = {tuple(value) for value in adaptive["gold_supporting_facts"]}
    predicted = {tuple(value) for value in adaptive["predicted_supporting_facts"]}
    false_positives = predicted - gold
    false_negatives = gold - predicted
    if false_positives and not false_negatives:
        return "overprediction_only"
    if false_negatives and not false_positives:
        return "underprediction_only"
    return "mixed_prediction_error"


def _average(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _summarize_route(
    route: str,
    records: list[dict],
    failure_counts: Counter,
    support_error_counts: Counter,
) -> dict:
    examples = len(records)
    summary = {
        "route": route,
        "route_label": ROUTE_LABELS[route],
        "examples": examples,
        "rate": 0.0,
    }
    for metric in ROUTE_METRICS:
        summary[metric] = _average(
            [float(record["metrics"][metric]) for record in records]
        )
    summary.update(
        {
            "average_context_tokens": _average(
                [float(record["stats"]["selected_context_tokens"]) for record in records]
            ),
            "average_structural_added_units": _average(
                [float(record["stats"]["structural_added_units"]) for record in records]
            ),
            "average_bridge_added_units": _average(
                [float(record["stats"]["bridge_added_units"]) for record in records]
            ),
            "second_hop_rate": _average(
                [float(record["stats"]["second_bridge_hop"]) for record in records]
            ),
            "average_gold_supporting_facts": _average(
                [float(len(record["gold_supporting_facts"])) for record in records]
            ),
            "average_predicted_supporting_facts": _average(
                [
                    float(len(record["predicted_supporting_facts"]))
                    for record in records
                ]
            ),
            "failure_counts": {
                stage: failure_counts[(route, stage)] for stage in FAILURE_LABELS
            },
            "failure_rates": {
                stage: failure_counts[(route, stage)] / examples if examples else 0.0
                for stage in FAILURE_LABELS
            },
            "support_prediction_error_counts": {
                error_type: support_error_counts[(route, error_type)]
                for error_type in SUPPORT_ERROR_LABELS
            },
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route-wise Error Analysis（分路由误差分析）"
    )
    parser.add_argument(
        "--input",
        default=str(EXPV4_ROOT / "results" / "final_evidence_evaluation_v4.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(
            EXPV4_ROOT / "results" / "final_evidence_error_analysis_v4.json"
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    route_records: dict[str, list[dict]] = defaultdict(list)
    failure_counts: Counter = Counter()
    support_error_counts: Counter = Counter()
    failure_ids: dict[str, list[str]] = defaultdict(list)
    comparison_values: dict[str, dict[str, list[float]]] = {
        baseline: defaultdict(list)
        for baseline in STRATEGIES
        if baseline != "adaptive"
    }
    comparison_outcomes: dict[str, dict[str, Counter]] = {
        baseline: {
            metric: Counter()
            for metric in ("support_f1", "selected_ccr", "selected_dccr")
        }
        for baseline in STRATEGIES
        if baseline != "adaptive"
    }
    recovery_effect = Counter()
    best_strategy_counts = Counter()
    adaptive_oracle_gaps: list[float] = []
    no_recovery_consistency_errors = 0
    examples = 0
    current_id = None
    group: dict[str, dict] = {}

    def analyze_group(example_id: str, records: dict[str, dict]) -> None:
        nonlocal examples, no_recovery_consistency_errors
        if set(records) != set(STRATEGIES):
            raise ValueError(
                f"Incomplete strategy group for {example_id}"
                "（同一问题的五种策略结果不完整）"
            )
        examples += 1
        adaptive = records["adaptive"]
        route = adaptive["stats"]["activation_pattern"]
        stage = _failure_stage(adaptive)
        route_records[route].append(adaptive)
        failure_counts[(route, stage)] += 1
        failure_counts[("all", stage)] += 1
        failure_ids[stage].append(example_id)
        if stage == "support_prediction_error":
            error_type = _support_error_type(adaptive)
            support_error_counts[(route, error_type)] += 1
            support_error_counts[("all", error_type)] += 1

        none_f1 = float(records["none"]["metrics"]["support_f1"])
        adaptive_f1 = float(adaptive["metrics"]["support_f1"])
        if route != "none":
            if adaptive_f1 > none_f1:
                recovery_effect["beneficial"] += 1
            elif adaptive_f1 < none_f1:
                recovery_effect["harmful"] += 1
            else:
                recovery_effect["neutral"] += 1

        if route == "none":
            for key in (
                "support_em",
                "support_precision",
                "support_recall",
                "support_f1",
                "candidate_ccr",
                "selected_ccr",
                "candidate_dccr",
                "selected_dccr",
            ):
                if adaptive["metrics"][key] != records["none"]["metrics"][key]:
                    no_recovery_consistency_errors += 1
                    break

        strategy_f1 = {
            strategy: float(records[strategy]["metrics"]["support_f1"])
            for strategy in STRATEGIES
        }
        best_f1 = max(strategy_f1.values())
        for strategy, value in strategy_f1.items():
            if value == best_f1:
                best_strategy_counts[strategy] += 1
        adaptive_oracle_gaps.append(best_f1 - adaptive_f1)

        for baseline in comparison_values:
            for metric in ("support_f1", "selected_ccr", "selected_dccr"):
                difference = float(adaptive["metrics"][metric]) - float(
                    records[baseline]["metrics"][metric]
                )
                comparison_values[baseline][metric].append(difference)
                if difference > 0:
                    outcome = "adaptive_higher"
                elif difference < 0:
                    outcome = "adaptive_lower"
                else:
                    outcome = "equal"
                comparison_outcomes[baseline][metric][outcome] += 1

    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            example_id = str(item["id"])
            if current_id is None:
                current_id = example_id
            if example_id != current_id:
                analyze_group(current_id, group)
                current_id = example_id
                group = {}
            group[item["strategy"]] = item
    if current_id is not None:
        analyze_group(current_id, group)

    route_table = []
    for route in ROUTE_LABELS:
        row = _summarize_route(
            route,
            route_records[route],
            failure_counts,
            support_error_counts,
        )
        row["rate"] = row["examples"] / examples if examples else 0.0
        route_table.append(row)

    overall_failure_counts = {
        stage: failure_counts[("all", stage)] for stage in FAILURE_LABELS
    }
    overall_failure_rates = {
        stage: count / examples if examples else 0.0
        for stage, count in overall_failure_counts.items()
    }
    pairwise = []
    for baseline, metric_values in comparison_values.items():
        for metric, values in metric_values.items():
            pairwise.append(
                {
                    "baseline": baseline,
                    "baseline_label": STRATEGY_LABELS[baseline],
                    "metric": metric,
                    "mean_adaptive_difference": _average(values),
                    "outcomes": dict(comparison_outcomes[baseline][metric]),
                }
            )

    activated_examples = sum(recovery_effect.values())
    summary = {
        "analysis": "Route-wise Error Analysis（分路由误差分析）",
        "examples": examples,
        "input_file": str(input_path),
        "failure_stage_definition": FAILURE_LABELS,
        "overall_failure_counts": overall_failure_counts,
        "overall_failure_rates": overall_failure_rates,
        "support_prediction_error_labels": SUPPORT_ERROR_LABELS,
        "support_prediction_error_counts": {
            error_type: support_error_counts[("all", error_type)]
            for error_type in SUPPORT_ERROR_LABELS
        },
        "route_table": route_table,
        "activated_recovery_effect_vs_no_recovery": {
            "examples": activated_examples,
            "counts": dict(recovery_effect),
            "rates": {
                key: recovery_effect[key] / activated_examples
                if activated_examples
                else 0.0
                for key in ("beneficial", "harmful", "neutral")
            },
        },
        "adaptive_support_f1_oracle": {
            "best_strategy_tie_counts": {
                strategy: best_strategy_counts[strategy] for strategy in STRATEGIES
            },
            "adaptive_matches_best_rate": best_strategy_counts["adaptive"] / examples,
            "average_gap_to_per_example_best": _average(adaptive_oracle_gaps),
        },
        "pairwise_per_example_outcomes": pairwise,
        "no_recovery_route_consistency_errors": no_recovery_consistency_errors,
        "failure_example_ids": dict(failure_ids),
        "external_model_calls": 0,
    }
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    route_csv_path = output_path.with_suffix(".routes.csv")
    route_csv_rows = []
    for row in route_table:
        flat = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "failure_counts",
                "failure_rates",
                "support_prediction_error_counts",
            }
        }
        for stage in FAILURE_LABELS:
            flat[f"{stage}_count"] = row["failure_counts"][stage]
            flat[f"{stage}_rate"] = row["failure_rates"][stage]
        for error_type in SUPPORT_ERROR_LABELS:
            flat[f"{error_type}_count"] = row[
                "support_prediction_error_counts"
            ][error_type]
        route_csv_rows.append(flat)
    with route_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(route_csv_rows[0]))
        writer.writeheader()
        writer.writerows(route_csv_rows)

    print(
        json.dumps(
            {
                "analysis": summary["analysis"],
                "examples": examples,
                "overall_failure_counts": overall_failure_counts,
                "overall_failure_rates": overall_failure_rates,
                "support_prediction_error_counts": summary[
                    "support_prediction_error_counts"
                ],
                "route_table": route_table,
                "activated_recovery_effect_vs_no_recovery": summary[
                    "activated_recovery_effect_vs_no_recovery"
                ],
                "adaptive_support_f1_oracle": summary[
                    "adaptive_support_f1_oracle"
                ],
                "no_recovery_route_consistency_errors": (
                    no_recovery_consistency_errors
                ),
                "output_file": str(output_path),
                "route_csv_file": str(route_csv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
