from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

import pandas as pd
import numpy as np


EXPV4_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPV4_ROOT.parent
sys.path.insert(0, str(EXPV4_ROOT))

from src.config import ExperimentConfig
from src.metrics.paper_metrics import title_ranking_scores, unique_titles
from src.models import EvidenceUnit
from src.retrieval.offline_baselines import OfflineBaselines
from src.selection.evidence_selection import ContextSelector


EXTERNAL_LABELS = {
    "semantic_rag": "SemanticRAG（语义检索增强生成）",
    "rerank_rag": "Rerank RAG（重排序检索增强生成）",
    "graph_rag": "GraphRAG（图检索增强生成）",
    "kg_rag": "KG-RAG（知识图谱检索增强生成）",
    "macrag": "MacRAG（多尺度自适应检索增强生成）",
    "ours": "Ours（本文方法）",
}
FIXED_LABELS = {
    "none": "No Recovery（无需恢复）",
    "always_structural": "Always Structural（始终结构恢复）",
    "always_bridge": "Always Bridge（始终桥接恢复）",
    "always_both": "Always Both（始终联合恢复）",
    "adaptive": "Adaptive/Ours（自适应恢复/本文方法）",
}
CORE_LABELS = {
    "full": "Full/Ours（完整方法/本文方法）",
    "without_structural": "w/o Structural（移除结构恢复）",
    "without_bridge": "w/o Semantic/Bridge（移除语义桥接）",
    "without_adaptive": "w/o Adaptive（移除自适应门控）",
    "without_budget": "w/o Budget（移除证据预算）",
}
SUPPLEMENT_LABELS = {
    "full_adaptive": "Full（完整方法）",
    "two_hop_extension": "Two-hop Extension（第二跳扩展）",
    "without_semantic_validation": "Raw Bridge（未验证桥接索引）",
}


def load_selected_rows(path: Path, selected_ids: set[str], key: str) -> dict[str, dict[str, dict]]:
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            example_id = str(item["id"])
            if example_id in selected_ids:
                rows[str(item[key])][example_id] = item
    return dict(rows)


def evidence_dict(unit: EvidenceUnit) -> dict:
    return {
        "title": unit.title,
        "sent_id": unit.sent_id,
        "text": unit.text,
        "score": unit.score,
        "source": unit.source,
        "vector_id": unit.vector_id,
        "metadata": unit.metadata,
    }


def gold_facts(value: dict) -> list[tuple[str, int]]:
    return [
        (str(title), int(sent_id))
        for title, sent_id in zip(value["title"], value["sent_id"])
    ]


def row_metrics(context: list[EvidenceUnit], gold_facts: list[list | tuple]) -> dict[str, float]:
    gold_titles = unique_titles([tuple(value) for value in gold_facts])
    predicted_titles = unique_titles([unit.key for unit in context])
    return title_ranking_scores(predicted_titles, gold_titles)


def ci95(values: list[float]) -> float:
    return 1.96 * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def aggregate(rows: list[dict]) -> dict:
    metric_names = [
        "title_recall",
        "title_precision",
        "title_mrr",
        "title_ndcg",
        "title_average_precision",
    ]
    output = {"examples": len(rows)}
    for metric in metric_names:
        values = [float(row["metrics"][metric]) for row in rows]
        output[metric] = mean(values)
        output[f"{metric}_ci95"] = ci95(values)
    output.update(
        {
            "avg_token": mean(float(row["stats"]["selected_context_tokens"]) for row in rows),
            "time_ms": mean(float(row["stats"]["time_ms"]) for row in rows),
            "ext_node": mean(float(row["stats"].get("extended_nodes", 0)) for row in rows),
        }
    )
    return output


def summaries_by_group(rows: list[dict], labels: dict[str, str], group_key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    table = []
    for key, label in labels.items():
        values = grouped[key]
        overall = aggregate(values)
        bridge = aggregate([row for row in values if row["question_type"] == "bridge"])
        comparison = aggregate(
            [row for row in values if row["question_type"] == "comparison"]
        )
        bridge_weight = 5517 / 6905
        comparison_weight = 1388 / 6905
        natural = {
            metric: bridge_weight * bridge[metric] + comparison_weight * comparison[metric]
            for metric in (
                "title_recall",
                "title_precision",
                "title_mrr",
                "title_ndcg",
                "title_average_precision",
                "avg_token",
                "time_ms",
                "ext_node",
            )
        }
        table.append(
            {
                group_key: key,
                "label": label,
                **overall,
                "bridge_title_recall": bridge["title_recall"],
                "comparison_title_recall": comparison["title_recall"],
                "natural_weighted_title_recall": natural["title_recall"],
                "natural_weighted_title_precision": natural["title_precision"],
                "natural_weighted_mrr": natural["title_mrr"],
                "natural_weighted_ndcg": natural["title_ndcg"],
                "natural_weighted_map": natural["title_average_precision"],
            }
        )
    return table


def write_table(output_dir: Path, name: str, table: list[dict], metadata: dict) -> None:
    json_path = output_dir / f"{name}.json"
    csv_path = output_dir / f"{name}.csv"
    payload = {**metadata, "table": table, "csv_file": str(csv_path)}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final Retrieval-only Experiments（最终仅检索实验）"
    )
    parser.add_argument(
        "--config", default=str(EXPV4_ROOT / "configs" / "proposed.json")
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--output-dir", default=str(EXPV4_ROOT / "results" / "final_2000_v4" / "retrieval")
    )
    parser.add_argument("--fixed-source")
    parser.add_argument("--ablation-source")
    args = parser.parse_args()

    config = ExperimentConfig.load(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_report_path = output_dir / "retrieval_experiment_report_v4.json"
    previous_reranking_requests = 0
    if previous_report_path.exists():
        previous_report = json.loads(previous_report_path.read_text(encoding="utf-8"))
        previous_reranking_requests = int(
            previous_report.get("reranking_requests_including_retries", 0)
        )
    split_path = PROJECT_ROOT / "data" / "v4" / "final_evaluation_split_2000_v4.json"
    cache_path = PROJECT_ROOT / "data" / "v4" / "final_evaluation_retrieval_cache_2000_v4.pkl"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    example_ids = [str(value) for value in split["evaluation_ids"]]
    selected_ids = set(example_ids)
    example_meta = {str(row["id"]): row for row in split["examples"]}
    with cache_path.open("rb") as handle:
        retrieval_cache = pickle.load(handle)

    frame = pd.read_parquet(
        config.validation_file,
        columns=["id", "question", "supporting_facts"],
    )
    frame["id"] = frame["id"].astype(str)
    rows_by_id = {
        str(row["id"]): row for _, row in frame[frame["id"].isin(selected_ids)].iterrows()
    }
    fixed_source = load_selected_rows(
        Path(args.fixed_source)
        if args.fixed_source
        else EXPV4_ROOT / "results" / "final_evidence_evaluation_v4.jsonl",
        selected_ids,
        "strategy",
    )
    ablation_source = load_selected_rows(
        Path(args.ablation_source)
        if args.ablation_source
        else EXPV4_ROOT / "results" / "frozen_ablation_study_v4.jsonl",
        selected_ids,
        "variant",
    )

    baselines = OfflineBaselines(config)
    record_by_key = {
        (record.title, record.sent_id): record for record in baselines.sentence_index.records
    }

    query_matrix = np.ascontiguousarray(
        np.stack([retrieval_cache[example_id]["query_vector"] for example_id in example_ids]),
        dtype=np.float32,
    )
    sentence_scores, sentence_ids = baselines.sentence_index.index.search(
        query_matrix, 10
    )
    sentence_candidates_by_id: dict[str, list[EvidenceUnit]] = {}
    for example_id, scores, vector_ids in zip(
        example_ids, sentence_scores, sentence_ids
    ):
        units = []
        for score, vector_id in zip(scores, vector_ids):
            if vector_id < 0:
                continue
            record = baselines.sentence_index.records[int(vector_id)]
            units.append(
                EvidenceUnit(
                    title=record.title,
                    sent_id=record.sent_id,
                    text=record.text,
                    score=float(score),
                    source="sentence_vector",
                    vector_id=record.vector_id,
                )
            )
        sentence_candidates_by_id[example_id] = units
    paragraph_batches = baselines.paragraph_index.batch_search(query_matrix, 10)
    paragraph_candidates_by_id = dict(zip(example_ids, paragraph_batches))

    def units_from_keys(keys: list[list | tuple], source: str) -> list[EvidenceUnit]:
        units = []
        for title, sent_id in keys:
            record = record_by_key[(str(title), int(sent_id))]
            units.append(
                EvidenceUnit(
                    title=record.title,
                    sent_id=record.sent_id,
                    text=record.text,
                    score=0.0,
                    source=source,
                    vector_id=record.vector_id,
                )
            )
        return units

    def make_row(
        example_id: str,
        group_key: str,
        group_value: str,
        label: str,
        context: list[EvidenceUnit],
        stats: dict,
    ) -> dict:
        example_gold_facts = gold_facts(rows_by_id[example_id]["supporting_facts"])
        normalized_stats = {
            **stats,
            "selected_context_tokens": sum(unit.token_count for unit in context),
            "selected_context_units": len(context),
            "time_ms": float(stats.get("time_ms", 0.0)),
            "extended_nodes": int(stats.get("extended_nodes", 0)),
        }
        return {
            "id": example_id,
            "question": str(rows_by_id[example_id]["question"]),
            "question_type": str(example_meta[example_id]["type"]),
            "level": str(example_meta[example_id]["level"]),
            group_key: group_value,
            "label": label,
            "gold_supporting_facts": example_gold_facts,
            "context_evidence": [evidence_dict(unit) for unit in context],
            "metrics": row_metrics(context, example_gold_facts),
            "stats": normalized_stats,
        }

    external_cache_path = output_dir / "external_retrieval_rows_v4.jsonl"
    external_rows: list[dict] = []
    if external_cache_path.exists():
        with external_cache_path.open(encoding="utf-8") as handle:
            external_rows = [json.loads(line) for line in handle if line.strip()]
        external_rows = [
            row for row in external_rows if str(row["method"]) != "ours"
        ]
        for example_id in example_ids:
            ours_source = fixed_source["adaptive"][example_id]
            ours_context = units_from_keys(ours_source["context_evidence"], "ours")
            ours_stats = {
                **ours_source["stats"],
                "extended_nodes": ours_source["stats"]["structural_added_units"]
                + ours_source["stats"]["bridge_added_units"],
            }
            external_rows.append(
                make_row(
                    example_id,
                    "method",
                    "ours",
                    EXTERNAL_LABELS["ours"],
                    ours_context,
                    ours_stats,
                )
            )
        with external_cache_path.open("w", encoding="utf-8") as handle:
            for row in external_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    cached_methods: dict[str, set[str]] = defaultdict(set)
    for row in external_rows:
        cached_methods[str(row["id"])].add(str(row["method"]))
    pending_ids = [
        example_id
        for example_id in example_ids
        if cached_methods[example_id] != set(EXTERNAL_LABELS)
    ]

    def run_external(example_id: str) -> list[dict]:
        source = rows_by_id[example_id]
        query = str(source["question"])
        cached = retrieval_cache[example_id]
        query_vector = cached["query_vector"]
        sentence_candidates = sentence_candidates_by_id[example_id]
        paragraph_candidates = paragraph_candidates_by_id[example_id]
        results = {
            "semantic_rag": baselines.semantic_rag(query_vector, sentence_candidates),
            "rerank_rag": baselines.rerank_rag(query, query_vector, sentence_candidates),
            "graph_rag": baselines.graph_rag(query, query_vector, sentence_candidates),
            "kg_rag": baselines.kg_rag(query, query_vector, cached["initial_evidence"]),
            "macrag": baselines.macrag(
                query,
                query_vector,
                sentence_candidates,
                paragraph_candidates,
            ),
        }
        output = [
            make_row(
                example_id,
                "method",
                method,
                EXTERNAL_LABELS[method],
                result.context_evidence,
                result.stats,
            )
            for method, result in results.items()
        ]
        ours_source = fixed_source["adaptive"][example_id]
        ours_context = units_from_keys(ours_source["context_evidence"], "ours")
        ours_stats = {
            **ours_source["stats"],
            "extended_nodes": ours_source["stats"]["structural_added_units"]
            + ours_source["stats"]["bridge_added_units"],
        }
        output.append(
            make_row(
                example_id, "method", "ours", EXTERNAL_LABELS["ours"], ours_context, ours_stats
            )
        )
        return output

    with external_cache_path.open("a", encoding="utf-8") as cache_handle:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(run_external, example_id): example_id for example_id in pending_ids}
            for completed, future in enumerate(as_completed(futures), start=1):
                result_rows = future.result()
                external_rows.extend(result_rows)
                for row in result_rows:
                    cache_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                cache_handle.flush()
                if args.progress_every and completed % args.progress_every == 0:
                    print(
                        f"Completed external retrieval {completed}/{len(pending_ids)}"
                        "（已完成外部方法检索）",
                        flush=True,
                    )

    order = {example_id: index for index, example_id in enumerate(example_ids)}
    external_rows.sort(key=lambda row: (order[row["id"]], list(EXTERNAL_LABELS).index(row["method"])))

    fixed_rows: list[dict] = []
    for strategy, label in FIXED_LABELS.items():
        for example_id in example_ids:
            source = fixed_source[strategy][example_id]
            context = units_from_keys(source["context_evidence"], strategy)
            stats = {
                **source["stats"],
                "extended_nodes": source["stats"]["structural_added_units"]
                + source["stats"]["bridge_added_units"],
            }
            fixed_rows.append(
                make_row(example_id, "strategy", strategy, label, context, stats)
            )

    core_rows: list[dict] = []
    source_map = {
        "full": ("ablation", "full_adaptive"),
        "without_structural": ("ablation", "without_structural"),
        "without_bridge": ("ablation", "without_bridge"),
        "without_adaptive": ("fixed", "always_both"),
    }
    for variant, (source_kind, source_name) in source_map.items():
        source_rows = ablation_source[source_name] if source_kind == "ablation" else fixed_source[source_name]
        for example_id in example_ids:
            source = source_rows[example_id]
            context = units_from_keys(source["context_evidence"], variant)
            stats = {
                **source["stats"],
                "extended_nodes": source["stats"].get("structural_added_units", 0)
                + source["stats"].get("bridge_added_units", 0),
            }
            core_rows.append(
                make_row(example_id, "variant", variant, CORE_LABELS[variant], context, stats)
            )

    no_budget_config = deepcopy(config)
    no_budget_config.context_budget = 10**9
    no_budget_config.max_context_units = 10**6
    no_budget_selector = ContextSelector(no_budget_config, baselines.sentence_index)
    for example_id in example_ids:
        source = fixed_source["adaptive"][example_id]
        candidates = units_from_keys(source["candidate_evidence"], "without_budget")
        context = no_budget_selector.select(candidates, retrieval_cache[example_id]["query_vector"])
        stats = {
            **source["stats"],
            "extended_nodes": source["stats"]["structural_added_units"]
            + source["stats"]["bridge_added_units"],
        }
        core_rows.append(
            make_row(
                example_id,
                "variant",
                "without_budget",
                CORE_LABELS["without_budget"],
                context,
                stats,
            )
        )

    supplement_rows: list[dict] = []
    for variant, label in SUPPLEMENT_LABELS.items():
        for example_id in example_ids:
            source = ablation_source[variant][example_id]
            context = units_from_keys(source["context_evidence"], variant)
            stats = {
                **source["stats"],
                "extended_nodes": source["stats"].get("structural_added_units", 0)
                + source["stats"].get("bridge_added_units", 0),
            }
            supplement_rows.append(
                make_row(example_id, "variant", variant, label, context, stats)
            )

    contexts_13 = []
    contexts_13.extend(external_rows)
    contexts_13.extend(row for row in fixed_rows if row["strategy"] != "adaptive")
    contexts_13.extend(
        row
        for row in core_rows
        if row["variant"] in {"without_structural", "without_bridge", "without_budget"}
    )
    contexts_path = output_dir / "retrieval_contexts_13_variants_v4.jsonl"
    with contexts_path.open("w", encoding="utf-8") as handle:
        for row in contexts_13:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    external_table = summaries_by_group(external_rows, EXTERNAL_LABELS, "method")
    fixed_table = summaries_by_group(fixed_rows, FIXED_LABELS, "strategy")
    core_table = summaries_by_group(core_rows, CORE_LABELS, "variant")
    supplement_table = summaries_by_group(
        supplement_rows, SUPPLEMENT_LABELS, "variant"
    )
    common_metadata = {
        "dataset": "HotpotQA v4 Final 2000（HotpotQA 第四版最终两千样本）",
        "sampling": "1000 Bridge + 1000 Comparison（两类各一千条）",
        "embedding_model": config.embedding_model,
        "reranking_model": config.rerank_model,
        "max_bridge_hops": config.max_bridge_hops,
        "large_language_model_calls": 0,
        "timing_note": (
            "Diagnostic only（仅作诊断）: external baselines reuse frozen query "
            "vectors while Ours inherits cached end-to-end initial retrieval time; "
            "time_ms is not a fair cross-method latency comparison."
        ),
    }
    cumulative_reranking_requests = previous_reranking_requests + baselines.rerank_calls
    write_table(
        output_dir,
        "table2_external_retrieval_comparison_v4",
        external_table,
        {**common_metadata, "reranking_requests": cumulative_reranking_requests},
    )
    write_table(
        output_dir,
        "table4_fixed_recovery_retrieval_v4",
        fixed_table,
        common_metadata,
    )
    write_table(
        output_dir,
        "table5_core_ablation_retrieval_v4",
        core_table,
        common_metadata,
    )
    write_table(
        output_dir,
        "table8_multihop_semantic_retrieval_v4",
        supplement_table,
        common_metadata,
    )

    ours_rows = [row for row in external_rows if row["method"] == "ours"]
    question_type_table = []
    for question_type in ("bridge", "comparison"):
        question_type_table.append(
            {
                "question_type": question_type,
                "label": f"{question_type.title()}（{'桥接型' if question_type == 'bridge' else '比较型'}）",
                **aggregate([row for row in ours_rows if row["question_type"] == question_type]),
            }
        )
    write_table(
        output_dir,
        "table6_question_type_retrieval_v4",
        question_type_table,
        common_metadata,
    )

    adaptive_rows = {row["id"]: row for row in fixed_rows if row["strategy"] == "adaptive"}
    gate_groups: dict[str, list[dict]] = defaultdict(list)
    for example_id in example_ids:
        pattern = str(fixed_source["adaptive"][example_id]["stats"]["activation_pattern"])
        gate_groups[pattern].append(adaptive_rows[example_id])
    gate_labels = {
        "none": "(0,0) No Recovery（无需恢复）",
        "structural_only": "(1,0) Structural Recovery（结构恢复）",
        "bridge_only": "(0,1) Bridge Recovery（桥接恢复）",
        "structural_bridge": "(1,1) Joint Recovery（联合恢复）",
    }
    gate_table = []
    for pattern, label in gate_labels.items():
        values = gate_groups[pattern]
        gate_table.append(
            {
                "activation_pattern": pattern,
                "label": label,
                "trigger_rate": len(values) / len(example_ids),
                **aggregate(values),
            }
        )
    write_table(
        output_dir,
        "table7_gating_behavior_retrieval_v4",
        gate_table,
        common_metadata,
    )

    efficiency_table = [
        {
            "method": row["method"],
            "label": row["label"],
            "examples": row["examples"],
            "avg_token": row["avg_token"],
            "time_ms": row["time_ms"],
            "ext_node": row["ext_node"],
        }
        for row in external_table
    ]
    write_table(
        output_dir,
        "table9_efficiency_retrieval_v4",
        efficiency_table,
        common_metadata,
    )

    report = {
        "examples": len(example_ids),
        "external_context_rows": len(external_rows),
        "fixed_strategy_rows": len(fixed_rows),
        "core_ablation_rows": len(core_rows),
        "supplement_rows": len(supplement_rows),
        "unique_paid_context_rows": len(contexts_13),
        "expected_unique_paid_context_rows": 13 * len(example_ids),
        "embedding_requests": 0,
        "reranking_requests_including_retries": cumulative_reranking_requests,
        "large_language_model_calls": 0,
        "output_dir": str(output_dir),
    }
    (output_dir / "retrieval_experiment_report_v4.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
