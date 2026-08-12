from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPV4 = ROOT / "expv4"
RETRIEVAL_DIR = EXPV4 / "results" / "final_2000_v4" / "retrieval"
GENERATION_DIR = EXPV4 / "results" / "final_2000_v4" / "generation"

METRICS = (
    "answer_em",
    "answer_f1",
    "accuracy",
    "faithfulness",
    "answer_relevance",
    "context_relevance",
)

LABELS = {
    "semantic_rag": "SemanticRAG（语义检索增强生成）",
    "rerank_rag": "Rerank RAG（重排序检索增强生成）",
    "graph_rag": "GraphRAG（图检索增强生成）",
    "kg_rag": "KG-RAG（知识图谱检索增强生成）",
    "macrag": "MacRAG（多尺度自适应检索增强生成）",
    "ours": "Ours（本文方法）",
    "none": "No Recovery（无需恢复）",
    "always_structural": "Always Structural（始终结构细化）",
    "always_bridge": "Always Bridge（始终桥接补全）",
    "always_both": "Always Both（始终联合恢复）",
    "without_structural": "w/o Structural（移除结构细化）",
    "without_bridge": "w/o Bridge Completion（移除桥接补全）",
    "without_budget": "w/o Budget（移除预算选择）",
    "raw_bridge": "Raw Bridge（未经大模型语义验证的桥接关系）",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean_ci(values: list[float]) -> tuple[float, float]:
    average = mean(values)
    if len(values) < 2:
        return average, 0.0
    return average, 1.96 * stdev(values) / math.sqrt(len(values))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty group")
    result: dict[str, Any] = {
        "examples": len(rows),
        "refusal_rate": mean(float(row["is_refusal"]) for row in rows),
    }
    refusal_values = [float(row["is_refusal"]) for row in rows]
    _, result["refusal_rate_ci95"] = mean_ci(refusal_values)
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        result[metric], result[f"{metric}_ci95"] = mean_ci(values)
    return result


def natural_weighted(rows: list[dict[str, Any]], bridge_weight: float) -> dict[str, float]:
    grouped = {
        question_type: [row for row in rows if row["question_type"] == question_type]
        for question_type in ("bridge", "comparison")
    }
    result = {}
    for metric in (*METRICS, "is_refusal"):
        bridge = mean(float(row[metric]) for row in grouped["bridge"])
        comparison = mean(float(row[metric]) for row in grouped["comparison"])
        name = "refusal_rate" if metric == "is_refusal" else metric
        result[f"natural_weighted_{name}"] = (
            bridge_weight * bridge + (1.0 - bridge_weight) * comparison
        )
    return result


def generation_row(
    method: str,
    rows_by_method: dict[str, list[dict[str, Any]]],
    bridge_weight: float,
    label: str | None = None,
) -> dict[str, Any]:
    rows = rows_by_method[method]
    return {
        "method": method,
        "label": label or LABELS[method],
        **aggregate(rows),
        **natural_weighted(rows, bridge_weight),
    }


def write_table(name: str, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    json_path = GENERATION_DIR / f"{name}.json"
    csv_path = GENERATION_DIR / f"{name}.csv"
    payload = {**metadata, "table": rows, "csv_file": str(csv_path)}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def retrieval_rows(filename: str, identity: str) -> dict[str, dict[str, Any]]:
    payload = json.loads((RETRIEVAL_DIR / filename).read_text(encoding="utf-8"))
    return {str(row[identity]): row for row in payload["table"]}


def main() -> None:
    answers_path = GENERATION_DIR / "answers_final_v4.jsonl"
    evaluations_path = GENERATION_DIR / "semantic_evaluations_final_v4.jsonl"
    answers = load_jsonl(answers_path)
    evaluations = load_jsonl(evaluations_path)
    answer_map = {(row["id"], row["method"]): row for row in answers}
    evaluation_map = {(row["id"], row["method"]): row for row in evaluations}
    if len(answer_map) != 28000 or len(evaluation_map) != 28000:
        raise RuntimeError(
            f"Expected 28,000 answers/evaluations, found {len(answer_map)}/{len(evaluation_map)}"
        )
    if set(answer_map) != set(evaluation_map):
        raise RuntimeError("Answer and evaluation key sets differ")
    if any(row["status"] != "success" for row in answers + evaluations):
        raise RuntimeError("Final files contain failed rows")

    rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluations:
        rows_by_method[row["method"]].append(row)
    method_counts = Counter(row["method"] for row in evaluations)
    if len(method_counts) != 14 or set(method_counts.values()) != {2000}:
        raise RuntimeError(f"Invalid final method counts: {dict(method_counts)}")

    split = json.loads(
        (ROOT / "data" / "v4" / "final_evaluation_split_2000_v4.json").read_text(
            encoding="utf-8"
        )
    )
    source_counts = split.get("source_type_counts", {"bridge": 5517, "comparison": 1388})
    bridge_weight = source_counts["bridge"] / sum(source_counts.values())
    metadata = {
        "dataset": "HotpotQA v4 Final 2000（HotpotQA 第四版最终两千样本）",
        "sampling": "1000 Bridge + 1000 Comparison（桥接型与比较型各一千条）",
        "generation_model": "glm-4.7-flash",
        "thinking_mode": "disabled",
        "temperature": 0.0,
        "evaluation_protocol": "v4.2-complete-evidence-chain",
        "recursive_bridge_completion_included": False,
        "natural_bridge_weight": bridge_weight,
    }

    external_methods = (
        "semantic_rag",
        "rerank_rag",
        "graph_rag",
        "kg_rag",
        "macrag",
        "ours",
    )
    table3 = [generation_row(method, rows_by_method, bridge_weight) for method in external_methods]
    write_table("table3_external_generation_quality_v4", table3, metadata)

    fixed_retrieval = retrieval_rows("table4_fixed_recovery_retrieval_v4.json", "strategy")
    table4 = []
    for method in ("none", "always_structural", "always_bridge", "always_both", "ours"):
        retrieval_key = "adaptive" if method == "ours" else method
        label = "Adaptive Recovery（自适应恢复）" if method == "ours" else LABELS[method]
        table4.append(
            {
                **fixed_retrieval[retrieval_key],
                **generation_row(method, rows_by_method, bridge_weight, label),
            }
        )
    write_table("table4_fixed_recovery_full_v4", table4, metadata)

    core_retrieval = retrieval_rows("table5_core_ablation_retrieval_v4.json", "variant")
    core_specs = (
        ("ours", "full", "Full（完整方法）"),
        ("without_structural", "without_structural", LABELS["without_structural"]),
        ("without_bridge", "without_bridge", LABELS["without_bridge"]),
        ("always_both", "without_adaptive", "w/o Adaptive Gating（移除自适应门控）"),
        ("without_budget", "without_budget", LABELS["without_budget"]),
    )
    table5 = []
    for method, variant, label in core_specs:
        table5.append(
            {
                **core_retrieval[variant],
                **generation_row(method, rows_by_method, bridge_weight, label),
                "variant": variant,
            }
        )
    write_table("table5_core_ablation_full_v4", table5, metadata)

    question_retrieval = retrieval_rows("table6_question_type_retrieval_v4.json", "question_type")
    ours_rows = rows_by_method["ours"]
    table6 = []
    for question_type, label in (
        ("bridge", "Bridge（桥接型）"),
        ("comparison", "Comparison（比较型）"),
    ):
        group = [row for row in ours_rows if row["question_type"] == question_type]
        table6.append({**question_retrieval[question_type], "label": label, **aggregate(group)})
    balanced = {"question_type": "balanced_overall", "label": "Balanced Overall（平衡总体）", **aggregate(ours_rows)}
    natural = {
        "question_type": "natural_weighted_overall",
        "label": "Natural-weighted Overall（原始分布加权总体）",
        "examples": 2000,
        **natural_weighted(ours_rows, bridge_weight),
    }
    table6.extend([balanced, natural])
    write_table("table6_question_type_full_v4", table6, metadata)

    activation_by_id: dict[str, str] = {}
    context_path = RETRIEVAL_DIR / "retrieval_contexts_14_final_variants_v4.jsonl"
    with context_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("method") == "ours":
                activation_by_id[row["id"]] = row["stats"]["activation_pattern"]
    gating_retrieval = retrieval_rows("table7_gating_behavior_retrieval_v4.json", "activation_pattern")
    gating_labels = {
        "none": "(0,0) No Recovery（无需恢复）",
        "structural_only": "(1,0) Structural Refinement（结构细化）",
        "bridge_only": "(0,1) Bridge Completion（桥接补全）",
        "structural_bridge": "(1,1) Joint Recovery（联合恢复）",
    }
    table7 = []
    for pattern in ("none", "structural_only", "bridge_only", "structural_bridge"):
        group = [row for row in ours_rows if activation_by_id[row["id"]] == pattern]
        table7.append(
            {
                **gating_retrieval[pattern],
                "label": gating_labels[pattern],
                "trigger_rate": len(group) / 2000,
                **aggregate(group),
            }
        )
    write_table("table7_gating_behavior_full_v4", table7, metadata)

    bridge_retrieval = retrieval_rows(
        "table8_bridge_semantic_validation_retrieval_v4.json", "variant"
    )
    semantic_specs = (
        ("ours", "full_adaptive", "Validated Bridge（经过语义验证的桥接关系）"),
        ("raw_bridge", "without_semantic_validation", LABELS["raw_bridge"]),
    )
    table8 = []
    for method, variant, label in semantic_specs:
        all_rows = rows_by_method[method]
        for scope in ("overall", "bridge", "comparison"):
            group = all_rows if scope == "overall" else [
                row for row in all_rows if row["question_type"] == scope
            ]
            retrieval = bridge_retrieval[variant] if scope == "overall" else {}
            table8.append(
                {
                    **retrieval,
                    "method": method,
                    "variant": variant,
                    "scope": scope,
                    "label": label,
                    **aggregate(group),
                }
            )
    write_table("table8_bridge_semantic_validation_full_v4", table8, metadata)

    efficiency_retrieval = retrieval_rows("table9_efficiency_retrieval_v4.json", "method")
    table9 = []
    for method in external_methods:
        method_answers = [row for row in answers if row["method"] == method]
        method_evaluations = rows_by_method[method]
        table9.append(
            {
                **efficiency_retrieval[method],
                "method": method,
                "label": LABELS[method],
                "answer_prompt_tokens": mean(row["prompt_tokens"] for row in method_answers),
                "answer_completion_tokens": mean(row["completion_tokens"] for row in method_answers),
                "answer_latency_seconds": mean(row["latency_seconds"] for row in method_answers),
                "evaluation_tokens": mean(row["total_tokens"] for row in method_evaluations),
                "evaluation_latency_seconds": mean(
                    row["latency_seconds"] for row in method_evaluations
                ),
            }
        )
    write_table("table9_efficiency_full_v4", table9, metadata)

    report = {
        "answers": len(answer_map),
        "evaluations": len(evaluation_map),
        "methods": dict(sorted(method_counts.items())),
        "tables": [
            "table3_external_generation_quality_v4",
            "table4_fixed_recovery_full_v4",
            "table5_core_ablation_full_v4",
            "table6_question_type_full_v4",
            "table7_gating_behavior_full_v4",
            "table8_bridge_semantic_validation_full_v4",
            "table9_efficiency_full_v4",
        ],
        "recursive_bridge_completion_included": False,
        "status": "complete",
    }
    (GENERATION_DIR / "generation_tables_report_v4.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
