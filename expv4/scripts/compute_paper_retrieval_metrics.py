from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


EXPV4_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPV4_ROOT))

from src.metrics.paper_metrics import title_ranking_scores, unique_titles


STRATEGY_LABELS = {
    "none": "No Recovery（无需恢复）",
    "always_structural": "Always Structural（始终结构恢复）",
    "always_bridge": "Always Bridge（始终桥接恢复）",
    "always_both": "Always Both（始终联合恢复）",
    "adaptive": "Adaptive Recovery（自适应恢复）",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper Retrieval Metrics（论文检索指标）"
    )
    parser.add_argument(
        "--input",
        default=str(EXPV4_ROOT / "results" / "final_evidence_evaluation_v4.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(
            EXPV4_ROOT / "results" / "paper_retrieval_metrics_v4.summary.json"
        ),
    )
    args = parser.parse_args()

    rows_by_strategy: dict[str, list[dict]] = defaultdict(list)
    with Path(args.input).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            strategy = item["strategy"]
            gold_titles = unique_titles(
                [tuple(value) for value in item["gold_supporting_facts"]]
            )
            context_titles = unique_titles(
                [tuple(value) for value in item["context_evidence"]]
            )
            scores = title_ranking_scores(context_titles, gold_titles)
            bridge_titles = {
                str(chain["target_title"])
                for chain in item["stats"].get("bridge_chains", [])
            }
            rows_by_strategy[strategy].append(
                {
                    **scores,
                    "avg_token": float(item["stats"]["selected_context_tokens"]),
                    "time_ms": float(item["stats"]["time_ms"]),
                    "recovery_time_ms": float(item["stats"]["recovery_time_ms"]),
                    "ext_node": float(len(bridge_titles)),
                }
            )

    table = []
    for strategy in STRATEGY_LABELS:
        rows = rows_by_strategy[strategy]
        table.append(
            {
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS[strategy],
                "examples": len(rows),
                "recall": mean(row["title_recall"] for row in rows),
                "precision": mean(row["title_precision"] for row in rows),
                "mrr": mean(row["title_mrr"] for row in rows),
                "ndcg": mean(row["title_ndcg"] for row in rows),
                "map": mean(row["title_average_precision"] for row in rows),
                "avg_token": mean(row["avg_token"] for row in rows),
                "time_ms": mean(row["time_ms"] for row in rows),
                "recovery_time_ms": mean(
                    row["recovery_time_ms"] for row in rows
                ),
                "ext_node": mean(row["ext_node"] for row in rows),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    summary = {
        "experiment": "Paper Retrieval Metrics（论文检索指标）",
        "metric_level": "Title Level（标题层级）",
        "metric_definitions": {
            "recall": "Gold Supporting Title Recall（金标准支持标题召回率）",
            "precision": "Gold Supporting Title Precision（金标准支持标题精确率）",
            "mrr": "Mean Reciprocal Rank（首个支持标题平均倒数排名）",
            "ndcg": "Normalized Discounted Cumulative Gain（归一化折损累计增益）",
            "map": "Mean Average Precision（平均精度均值）",
            "avg_token": "Average Context Tokens（平均上下文词元）",
            "time_ms": "Cached Retrieval plus Recovery Time（缓存检索耗时加恢复耗时）",
            "ext_node": "Unique Bridge Target Titles（唯一桥接目标标题数）",
        },
        "external_model_calls": 0,
        "table": table,
        "source_file": str(Path(args.input)),
        "csv_file": str(csv_path),
    }
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
