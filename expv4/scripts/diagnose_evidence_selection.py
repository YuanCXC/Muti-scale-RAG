from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

# ruff: noqa: E402

import numpy as np
import pandas as pd


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from calibrate_bridge_threshold import _bootstrap_interval
from src.config import ExperimentConfig
from src.metrics.multihop_metrics import supporting_fact_scores


DIAGNOSIS_LABELS = {
    "initial_complete_preserved": "Initial Complete Preserved（初始完整证据被保留）",
    "initial_context_selection_loss": "Initial Evidence Selection Loss（初始完整证据被上下文选择丢失）",
    "initial_support_prediction_loss": "Initial Support Prediction Loss（初始完整证据被支撑事实预测丢失）",
    "document_recovery_failure": "Document Recovery Failure（文档恢复失败）",
    "target_sentence_localization_failure": "Target Sentence Localization Failure（目标支持句定位失败）",
    "context_selection_loss": "Context Selection Loss（上下文选择丢失）",
    "support_prediction_loss": "Support Prediction Loss（支撑事实预测丢失）",
    "recovered_complete": "Recovered Chain Preserved（恢复后的完整证据链被保留）",
}


def _facts(value) -> list[tuple[str, int]]:
    return [(str(title), int(sent_id)) for title, sent_id in value]


def _gold_facts(value: dict) -> list[tuple[str, int]]:
    return [
        (str(title), int(sent_id))
        for title, sent_id in zip(value["title"], value["sent_id"])
    ]


def _coverage(
    evidence_facts: list[tuple[str, int]], gold_facts: list[tuple[str, int]]
) -> dict[str, float]:
    evidence = set(evidence_facts)
    gold = set(gold_facts)
    evidence_documents = {title for title, _ in evidence}
    gold_documents = {title for title, _ in gold}
    return {
        "support_recall": len(evidence & gold) / len(gold) if gold else 0.0,
        "ccr": float(gold <= evidence),
        "dccr": float(gold_documents <= evidence_documents),
    }


def _diagnose_case(
    initial: dict[str, float],
    candidate: dict[str, float],
    selected: dict[str, float],
    predicted: dict[str, float],
) -> str:
    if initial["ccr"]:
        if not selected["ccr"]:
            return "initial_context_selection_loss"
        if not predicted["ccr"]:
            return "initial_support_prediction_loss"
        return "initial_complete_preserved"
    if not candidate["dccr"]:
        return "document_recovery_failure"
    if not candidate["ccr"]:
        return "target_sentence_localization_failure"
    if not selected["ccr"]:
        return "context_selection_loss"
    if not predicted["ccr"]:
        return "support_prediction_loss"
    return "recovered_complete"


def _aggregate_group(name: str, rows: list[dict]) -> dict:
    return {
        "group": name,
        "examples": len(rows),
        "initial_support_recall": mean(row["initial_support_recall"] for row in rows),
        "candidate_support_recall": mean(
            row["candidate_support_recall"] for row in rows
        ),
        "selected_support_recall": mean(row["selected_support_recall"] for row in rows),
        "predicted_support_recall": mean(
            row["predicted_support_recall"] for row in rows
        ),
        "predicted_support_precision": mean(
            row["predicted_support_precision"] for row in rows
        ),
        "predicted_support_f1": mean(row["predicted_support_f1"] for row in rows),
        "initial_ccr": mean(row["initial_ccr"] for row in rows),
        "candidate_ccr": mean(row["candidate_ccr"] for row in rows),
        "selected_ccr": mean(row["selected_ccr"] for row in rows),
        "predicted_ccr": mean(row["predicted_ccr"] for row in rows),
        "initial_dccr": mean(row["initial_dccr"] for row in rows),
        "candidate_dccr": mean(row["candidate_dccr"] for row in rows),
        "selected_dccr": mean(row["selected_dccr"] for row in rows),
        "predicted_dccr": mean(row["predicted_dccr"] for row in rows),
        "average_context_tokens": mean(row["context_tokens"] for row in rows),
    }


def _bootstrap_transitions(rows: list[dict], samples: int, seed: int) -> list[dict]:
    transitions = [
        ("Candidate minus Initial（候选池减初始证据）", "candidate", "initial"),
        ("Selected minus Candidate（最终上下文减候选池）", "selected", "candidate"),
        (
            "Predicted minus Selected（预测支撑事实减最终上下文）",
            "predicted",
            "selected",
        ),
    ]
    rng = np.random.default_rng(seed)
    results = []
    for transition_label, after, before in transitions:
        for metric in ("support_recall", "ccr", "dccr"):
            differences = np.asarray(
                [row[f"{after}_{metric}"] - row[f"{before}_{metric}"] for row in rows],
                dtype=float,
            )
            lower, upper = _bootstrap_interval(differences, samples, rng)
            results.append(
                {
                    "transition": transition_label,
                    "metric": metric,
                    "mean_difference": float(differences.mean()),
                    "ci_95_lower": lower,
                    "ci_95_upper": upper,
                    "bootstrap_samples": samples,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evidence Selection Diagnosis（证据选择诊断）"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument("--input")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    selection_path = config.output_dir / "joint_threshold_selection_final_v4.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_pair = selection["selected_pair"]
    if args.input:
        input_path = Path(args.input)
    else:
        candidates = sorted(
            config.output_dir.glob("joint_threshold_confirmation_*.jsonl"),
            key=lambda path: path.stat().st_mtime,
        )
        input_path = candidates[-1]

    frame = pd.read_parquet(config.validation_file)
    metadata_by_id = {
        str(row["id"]): {
            "question_type": str(row["type"]),
            "level": str(row["level"]),
            "gold_facts": _gold_facts(row["supporting_facts"]),
        }
        for _, row in frame.iterrows()
    }

    case_rows = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if float(item["structural_threshold"]) != float(
                selected_pair["structural_threshold"]
            ) or float(item["bridge_threshold"]) != float(
                selected_pair["bridge_threshold"]
            ):
                continue
            example_id = str(item["id"])
            metadata = metadata_by_id[example_id]
            gold_facts = metadata["gold_facts"]
            initial_facts = _facts(item["initial_evidence"])
            candidate_facts = _facts(item["candidate_evidence"])
            selected_facts = _facts(item["context_evidence"])
            predicted_facts = _facts(item["predicted_supporting_facts"])
            initial = _coverage(initial_facts, gold_facts)
            candidate = _coverage(candidate_facts, gold_facts)
            selected = _coverage(selected_facts, gold_facts)
            predicted = _coverage(predicted_facts, gold_facts)
            support = supporting_fact_scores(predicted_facts, gold_facts)
            diagnosis = _diagnose_case(initial, candidate, selected, predicted)
            case_rows.append(
                {
                    "id": example_id,
                    "question": item["question"],
                    "question_type": metadata["question_type"],
                    "level": metadata["level"],
                    "activation_pattern": item["stats"]["activation_pattern"],
                    "diagnosis": diagnosis,
                    "diagnosis_label": DIAGNOSIS_LABELS[diagnosis],
                    "initial_support_recall": initial["support_recall"],
                    "candidate_support_recall": candidate["support_recall"],
                    "selected_support_recall": selected["support_recall"],
                    "predicted_support_recall": predicted["support_recall"],
                    "predicted_support_precision": support["support_precision"],
                    "predicted_support_f1": support["support_f1"],
                    "initial_ccr": initial["ccr"],
                    "candidate_ccr": candidate["ccr"],
                    "selected_ccr": selected["ccr"],
                    "predicted_ccr": predicted["ccr"],
                    "initial_dccr": initial["dccr"],
                    "candidate_dccr": candidate["dccr"],
                    "selected_dccr": selected["dccr"],
                    "predicted_dccr": predicted["dccr"],
                    "context_tokens": item["stats"]["selected_context_tokens"],
                }
            )

    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    grouped_rows["Overall（总体）"] = case_rows
    for row in case_rows:
        grouped_rows[f"Activation: {row['activation_pattern']}（按激活模式）"].append(
            row
        )
        grouped_rows[f"Question Type: {row['question_type']}（按问题类型）"].append(row)
        grouped_rows[f"Level: {row['level']}（按难度）"].append(row)
    group_table = [_aggregate_group(name, rows) for name, rows in grouped_rows.items()]

    diagnosis_counts = Counter(row["diagnosis"] for row in case_rows)
    diagnosis_table = []
    for diagnosis, count in diagnosis_counts.most_common():
        examples = [row["id"] for row in case_rows if row["diagnosis"] == diagnosis]
        diagnosis_table.append(
            {
                "diagnosis": diagnosis,
                "diagnosis_label": DIAGNOSIS_LABELS[diagnosis],
                "examples": count,
                "rate": count / len(case_rows),
                "example_ids": examples[:10],
            }
        )

    overall = group_table[0]
    diagnostic_gaps = {
        "target_sentence_localization_gap": overall["candidate_dccr"]
        - overall["candidate_ccr"],
        "context_selection_ccr_loss": overall["candidate_ccr"]
        - overall["selected_ccr"],
        "support_prediction_ccr_loss": overall["selected_ccr"]
        - overall["predicted_ccr"],
        "context_selection_dccr_loss": overall["candidate_dccr"]
        - overall["selected_dccr"],
        "support_prediction_dccr_loss": overall["selected_dccr"]
        - overall["predicted_dccr"],
    }
    primary_bottleneck = max(diagnostic_gaps, key=diagnostic_gaps.get)
    transitions = _bootstrap_transitions(case_rows, args.bootstrap_samples, args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else config.output_dir / f"evidence_selection_diagnosis_{timestamp}"
    )
    summary_path = prefix.with_suffix(".summary.json")
    group_csv_path = prefix.with_suffix(".groups.csv")
    case_csv_path = prefix.with_suffix(".cases.csv")
    summary = {
        "experiment": "Evidence Selection Diagnosis（证据选择诊断）",
        "examples": len(case_rows),
        "frozen_thresholds": selected_pair,
        "input_file": str(input_path),
        "stage_table": [
            {
                "stage": "Initial Evidence（初始证据）",
                "support_recall": overall["initial_support_recall"],
                "ccr": overall["initial_ccr"],
                "dccr": overall["initial_dccr"],
            },
            {
                "stage": "Candidate Evidence（候选证据）",
                "support_recall": overall["candidate_support_recall"],
                "ccr": overall["candidate_ccr"],
                "dccr": overall["candidate_dccr"],
            },
            {
                "stage": "Selected Context（选中上下文）",
                "support_recall": overall["selected_support_recall"],
                "ccr": overall["selected_ccr"],
                "dccr": overall["selected_dccr"],
            },
            {
                "stage": "Predicted Supporting Facts（预测支撑事实）",
                "support_recall": overall["predicted_support_recall"],
                "ccr": overall["predicted_ccr"],
                "dccr": overall["predicted_dccr"],
            },
        ],
        "predicted_support_precision": overall["predicted_support_precision"],
        "predicted_support_f1": overall["predicted_support_f1"],
        "diagnostic_gaps": diagnostic_gaps,
        "primary_bottleneck": primary_bottleneck,
        "diagnosis_table": diagnosis_table,
        "group_table": group_table,
        "paired_bootstrap_transitions": transitions,
        "group_csv_file": str(group_csv_path),
        "case_csv_file": str(case_csv_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with group_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_table[0]))
        writer.writeheader()
        writer.writerows(group_table)
    with case_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
