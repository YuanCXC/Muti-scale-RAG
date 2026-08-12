from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPV4 = ROOT / "expv4"
sys.path.insert(0, str(EXPV4))

from src.config import ExperimentConfig
from src.indexing.sentence_index import SentenceIndex
from src.metrics.paper_metrics import title_ranking_scores, unique_titles
from src.models import EvidenceUnit
from src.retrieval.standard_external.graph_methods import (
    GraphRAGLocalAdapter,
    HippoRAG2Adapter,
    KG2RAGAdapter,
    MacRAGAdapter,
    RelationGraph,
)
from src.retrieval.standard_external.protocol import EvidenceBudget, StandardCorpus, Timer


METHODS = {
    "hybrid_rerank": "Hybrid + Rerank（混合检索加重排）",
    "graphrag": "GraphRAG（图检索增强生成）",
    "kg2rag": "KG²RAG / KG-RAG（知识图谱检索增强生成）",
    "macrag": "MacRAG（多尺度自适应检索增强生成）",
    "hipporag2": "HippoRAG 2（图记忆检索增强生成）",
    "ours": "Ours（本文方法）",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified-resource external retrieval comparison")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def load_graph(path: Path) -> tuple[RelationGraph, int]:
    graph = RelationGraph()
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("semantic_verdict") != "supported":
                continue
            if item.get("included_in_final_index") is False:
                continue
            graph.add(
                str(item["source_title"]),
                str(item["target_title"]),
                float(item.get("semantic_bridge_prior", 0.0)),
            )
            count += 1
    return graph, count


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def facts(value: dict) -> list[tuple[str, int]]:
    return [(str(title), int(sent_id)) for title, sent_id in zip(value["title"], value["sent_id"])]


def retrieval_metrics(evidence: list[EvidenceUnit], gold: list[tuple[str, int]]) -> dict[str, float]:
    return title_ranking_scores(unique_titles([unit.key for unit in evidence]), unique_titles(gold))


def aggregate(rows: list[dict]) -> dict:
    names = ("title_recall", "title_precision", "title_mrr", "title_ndcg", "title_average_precision")
    result = {"examples": len(rows)}
    for name in names:
        values = [float(row["metrics"][name]) for row in rows]
        result[name] = mean(values)
        result[f"{name}_ci95"] = 1.96 * stdev(values) / math.sqrt(len(values))
    result["avg_token"] = mean(row["stats"]["selected_context_tokens"] for row in rows)
    result["retrieval_time_ms"] = mean(row["stats"]["retrieval_time_ms"] for row in rows)
    result["online_llm_calls"] = sum(row["stats"].get("online_llm_calls", 0) for row in rows)
    return result


def make_row(
    example_id: str,
    question: str,
    meta: dict,
    method: str,
    evidence: list[EvidenceUnit],
    gold: list[tuple[str, int]],
    seconds: float,
    metadata: dict,
) -> dict:
    return {
        "id": example_id,
        "question": question,
        "question_type": str(meta["type"]),
        "level": str(meta["level"]),
        "method": method,
        "label": METHODS[method],
        "gold_supporting_facts": gold,
        "context_evidence": [evidence_dict(unit) for unit in evidence],
        "metrics": retrieval_metrics(evidence, gold),
        "stats": {
            "selected_context_tokens": sum(unit.token_count for unit in evidence),
            "selected_context_units": len(evidence),
            "retrieval_time_ms": seconds * 1000.0,
            "online_llm_calls": 0,
            **metadata,
        },
    }


def main() -> None:
    args = arguments()
    config = ExperimentConfig.load(EXPV4 / "configs" / "proposed.json")
    output_dir = EXPV4 / "results" / "final_2000_v4" / "unified_external"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "retrieval_contexts_unified_external_v4.jsonl"

    split = json.loads((ROOT / "data" / "v4" / "final_evaluation_split_2000_v4.json").read_text(encoding="utf-8"))
    ids = [str(value) for value in split["evaluation_ids"]]
    meta_by_id = {str(row["id"]): row for row in split["examples"]}
    selected = set(ids)
    frame = pd.read_parquet(config.validation_file, columns=["id", "question", "supporting_facts"])
    frame["id"] = frame["id"].astype(str)
    source_by_id = {str(row["id"]): row for _, row in frame[frame["id"].isin(selected)].iterrows()}

    with (ROOT / "data" / "v4" / "final_evaluation_retrieval_cache_2000_v4.pkl").open("rb") as handle:
        cache = pickle.load(handle)
    corpus = StandardCorpus.from_official_sentences(config.sentence_file)
    graph, validated_edges = load_graph(config.bridge_relation_dataset_file)
    budget = EvidenceBudget(config.context_budget, config.max_context_units)
    adapters = {
        "graphrag": GraphRAGLocalAdapter(corpus, graph, budget),
        "kg2rag": KG2RAGAdapter(corpus, graph, budget),
        "macrag": MacRAGAdapter(corpus, budget),
        "hipporag2": HippoRAG2Adapter(corpus, graph, budget),
    }

    ours_by_id: dict[str, dict] = {}
    source_contexts = EXPV4 / "results" / "final_2000_v4" / "retrieval" / "retrieval_contexts_14_final_variants_v4.jsonl"
    for row in load_jsonl(source_contexts):
        if row.get("method") == "ours":
            ours_by_id[str(row["id"])] = row
    if len(ours_by_id) != 2000:
        raise RuntimeError(f"Expected 2,000 Ours contexts, found {len(ours_by_id)}")

    existing = {(str(row["id"]), str(row["method"])): row for row in load_jsonl(rows_path)}
    with rows_path.open("a", encoding="utf-8") as output:
        for index, example_id in enumerate(ids, start=1):
            source = source_by_id[example_id]
            question = str(source["question"])
            gold = facts(source["supporting_facts"])
            initial = list(cache[example_id]["initial_evidence"])
            initial = budget.apply(initial)
            title_scores: dict[str, float] = {}
            for unit in initial:
                title_scores[unit.title] = max(title_scores.get(unit.title, -math.inf), float(unit.score))
            seeds = sorted(title_scores.items(), key=lambda item: item[1], reverse=True)

            generated: dict[str, tuple[list[EvidenceUnit], float, dict]] = {
                "hybrid_rerank": (
                    initial,
                    float(cache[example_id]["retrieval_time_ms"]) / 1000.0,
                    {
                        "resource": "frozen_v4_initial_retrieval_cache",
                        "embedding_model": config.embedding_model,
                        "reranking_model": config.rerank_model,
                    },
                )
            }
            for method, adapter in adapters.items():
                result = adapter.retrieve(initial, question) if method == "macrag" else adapter.retrieve(seeds, question)
                generated[method] = (result.evidence, result.retrieval_seconds, result.metadata)

            ours = ours_by_id[example_id]
            ours_evidence = [EvidenceUnit(**unit) for unit in ours["context_evidence"]]
            generated["ours"] = (
                ours_evidence,
                float(ours["stats"]["time_ms"]) / 1000.0,
                {"resource": "frozen_v4_formal_retrieval_result", "activation_pattern": ours["stats"]["activation_pattern"]},
            )

            for method, (evidence, seconds, metadata) in generated.items():
                key = (example_id, method)
                if key in existing:
                    continue
                row = make_row(example_id, question, meta_by_id[example_id], method, evidence, gold, seconds, metadata)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                existing[key] = row
            if args.progress_every and index % args.progress_every == 0:
                print(f"retrieval: {index}/{len(ids)}", flush=True)

    rows = list(existing.values())
    counts = Counter(row["method"] for row in rows)
    if counts != Counter({method: 2000 for method in METHODS}):
        raise RuntimeError(f"Invalid method distribution: {dict(counts)}")
    table = [{"method": method, "label": label, **aggregate([row for row in rows if row["method"] == method])} for method, label in METHODS.items()]
    metadata = {
        "dataset": "HotpotQA v4 Final 2000（桥接型和比较型各 1,000 条）",
        "comparison_protocol": "Unified-resource reproduction（统一资源复现）",
        "shared_corpus_documents": len(corpus.documents),
        "shared_validated_relation_edges": validated_edges,
        "context_budget_tokens": config.context_budget,
        "max_context_units": config.max_context_units,
        "online_large_language_model_calls": 0,
        "note": "Baselines reproduce core online mechanisms over identical V4 corpus, embeddings, reranking seeds, and validated relation graph; values are not copied from original papers.",
        "table": table,
    }
    json_path = output_dir / "table2_unified_external_retrieval_v4.json"
    csv_path = output_dir / "table2_unified_external_retrieval_v4.csv"
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(table, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
