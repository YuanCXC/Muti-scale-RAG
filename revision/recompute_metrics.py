# -*- coding: utf-8 -*-
"""Recompute revision metrics from experiments_v1 and new_experiments_v2.

Comparison tables are sourced from experiments_v1. Main, fixed-scale,
ablation, complexity, and efficiency tables are sourced from new_experiments_v2.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = PROJECT_ROOT / "experiments_v1"
V2_RUN = PROJECT_ROOT / "new_experiments_v2" / "results" / "paper_hotpotqa_20260531_091117"
OUTPUT_ROOT = PROJECT_ROOT / "revision" / "recomputed_metrics"


@dataclass(frozen=True)
class V1ExperimentSource:
    experiment: str
    method: str
    path: Path
    has_llm_judge: bool
    sample_limit: Optional[int] = None


V1_COMPARISON_SOURCES = [
    V1ExperimentSource(
        experiment="exp1_coarse_vector_retrieval",
        method="Coarse vector retrieval",
        path=V1_ROOT / "exp1_coarse_vector_retrieval" / "experiment_details_20260328_191104.json",
        has_llm_judge=False,
        sample_limit=1000,
    ),
    V1ExperimentSource(
        experiment="exp2_fine_grained_vector_retrieval",
        method="Fine-grained vector retrieval",
        path=V1_ROOT / "exp2_fine_grained_vector_retrieval" / "semantic_evaluation_20260531_221910.json",
        has_llm_judge=True,
    ),
    V1ExperimentSource(
        experiment="exp3_unified_chunking",
        method="Unified chunking",
        path=V1_ROOT / "exp3_unified_chunking" / "semantic_evaluation_20260531_222130.json",
        has_llm_judge=True,
    ),
    V1ExperimentSource(
        experiment="exp4_1hop_expansion",
        method="1-hop expansion",
        path=V1_ROOT / "exp4_1hop_expansion" / "semantic_evaluation_20260531_222339.json",
        has_llm_judge=True,
    ),
    V1ExperimentSource(
        experiment="exp5_2hop_expansion",
        method="2-hop expansion",
        path=V1_ROOT / "exp5_2hop_expansion" / "semantic_evaluation_20260531_222603.json",
        has_llm_judge=True,
    ),
]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def mean(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if is_number(v)]
    return sum(nums) / len(nums) if nums else None


def round4(value: Optional[float]) -> Optional[float]:
    return round(float(value), 4) if is_number(value) else None


def mean_record(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return {key: round4(mean(row.get(key) for row in rows)) for key in keys}


def nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def flatten_v1_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = row.get("metrics") or {}
    generation = row.get("generation_metrics") or {}
    semantic = row.get("semantic_metrics") or {}
    stats = row.get("stats") or {}

    flat: Dict[str, Any] = {
        "Title Recall": row.get("title_recall"),
        "Title Precision": row.get("title_precision"),
        "MRR": metrics.get("mrr"),
        "NDCG": metrics.get("ndcg"),
        "MAP": metrics.get("map_score"),
        "Hit Rate": metrics.get("hit_rate"),
        "Answer EM": generation.get("exact_match"),
        "Answer F1": generation.get("f1_score"),
        "Semantic Similarity": generation.get("semantic_similarity"),
        "correctness": semantic.get("correctness"),
        "faithfulness": semantic.get("faithfulness"),
        "context_relevance": semantic.get("context_relevance"),
        "Graph Coverage": row.get("graph_coverage"),
        "Expanded Entities": row.get("expanded_entity_count"),
        "Expanded Edges": row.get("expanded_edge_count"),
        "Avg Evidence Score": row.get("avg_evidence_score"),
        "Min Evidence Score": row.get("min_evidence_score"),
        "Update Trigger Rate": row.get("update_trigger_rate"),
    }

    for k, value in (metrics.get("recall_at_k") or {}).items():
        flat[f"Recall@{k}"] = value
    for k, value in (metrics.get("precision_at_k") or {}).items():
        flat[f"Precision@{k}"] = value

    latency = stats.get("latency") if isinstance(stats, Mapping) else None
    if isinstance(latency, Mapping):
        for key, value in latency.items():
            flat[f"Latency {key}"] = value

    return flat


def flatten_v2_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    retrieval = row.get("retrieval_metrics") or {}
    semantic = row.get("semantic_metrics") or {}
    stats = row.get("stats") or {}
    return {
        "Recall": retrieval.get("recall"),
        "Precision": retrieval.get("precision"),
        "MRR": retrieval.get("mrr"),
        "NDCG": retrieval.get("ndcg"),
        "MAP": retrieval.get("map_score"),
        "Avg Len": retrieval.get("avg_len"),
        "Time/ms": retrieval.get("time_ms"),
        "Expanded Nodes": retrieval.get("expanded_nodes"),
        "correctness": semantic.get("correctness"),
        "faithfulness": semantic.get("faithfulness"),
        "answer_relevance": semantic.get("answer_relevance"),
        "context_relevance": semantic.get("context_relevance"),
        "Complexity Score": row.get("complexity_score"),
        "Route Graph Trigger": 1.0 if row.get("route") == "graph_expansion" else 0.0,
        "Route Parent": 1.0 if row.get("route") in {"parent", "parent_all"} else 0.0,
        "Route Fine": 1.0 if row.get("route") == "fine_only" else 0.0,
        "Route": stats.get("route") if isinstance(stats, Mapping) else row.get("route"),
    }


def load_v1_rows(source: V1ExperimentSource) -> List[Dict[str, Any]]:
    with open(source.path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("results") if isinstance(data, Mapping) and "results" in data else data
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported v1 result shape: {source.path}")
    if source.sample_limit:
        rows = rows[: source.sample_limit]
    return rows


def recompute_v1_comparison() -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for source in V1_COMPARISON_SOURCES:
        rows = load_v1_rows(source)
        flat_rows = [flatten_v1_row(row) for row in rows]
        record = {
            "Section": "comparison_v1",
            "Experiment": source.experiment,
            "Method": source.method,
            "Samples": len(rows),
            "Source": str(source.path.relative_to(PROJECT_ROOT)),
            "Has LLM Judge": source.has_llm_judge,
        }
        record.update(mean_record(flat_rows))
        records.append(record)
    return pd.DataFrame(records)


def load_v2_details() -> Dict[str, Any]:
    with open(V2_RUN / "details.json", "r", encoding="utf-8") as f:
        return json.load(f)


def recompute_v2_group(details: Mapping[str, Any], key: str, section: str, methods: Optional[Sequence[str]] = None) -> pd.DataFrame:
    group = details[key]
    if methods is None:
        methods = list(group.keys())
    records: List[Dict[str, Any]] = []
    for method in methods:
        rows = group[method]
        flat_rows = [flatten_v2_row(row) for row in rows]
        record = {
            "Section": section,
            "Experiment": key,
            "Method": method,
            "Samples": len(rows),
            "Source": str((V2_RUN / "details.json").relative_to(PROJECT_ROOT)),
            "Has LLM Judge": True,
        }
        record.update(mean_record(flat_rows))
        records.append(record)
    return pd.DataFrame(records)


def recompute_v2_tables(details: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    return {
        "main_v2_proposed": recompute_v2_group(details, "method_rows", "main_v2", ["Proposed"]),
        "fixed_scale_v2": recompute_v2_group(details, "fixed_rows", "fixed_scale_v2"),
        "ablation_v2": recompute_v2_group(details, "ablation_rows", "ablation_v2"),
        "complexity_v2": pd.DataFrame(details["complexity_rows"]),
        "efficiency_v2": pd.DataFrame(details["efficiency_rows"]),
    }


def write_outputs(output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}

    comparison = recompute_v1_comparison()
    comparison_path = output_dir / "comparison_v1_all_metrics.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    outputs["comparison_v1_all_metrics"] = str(comparison_path)

    details = load_v2_details()
    v2_tables = recompute_v2_tables(details)
    summary_frames = [comparison]
    for name, df in v2_tables.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        outputs[name] = str(path)
        if "Method" in df.columns and "Samples" in df.columns:
            summary_frames.append(df)

    summary = pd.concat(summary_frames, ignore_index=True, sort=False)
    summary_path = output_dir / "all_selected_experiments_all_metrics.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    outputs["all_selected_experiments_all_metrics"] = str(summary_path)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "policy": {
                    "comparison": "experiments_v1",
                    "main_fixed_ablation_complexity_efficiency": "new_experiments_v2",
                    "v1_exp1_note": "No 1000-sample LLM judge file exists; first 1000 rows are summarized with available retrieval/generation metrics only.",
                },
                "outputs": outputs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    outputs["manifest"] = str(output_dir / "manifest.json")
    return outputs


def main() -> int:
    output_dir = OUTPUT_ROOT / datetime.now().strftime("metrics_%Y%m%d_%H%M%S")
    outputs = write_outputs(output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
