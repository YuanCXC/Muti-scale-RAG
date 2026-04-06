# -*- coding: utf-8 -*-
"""重新计算实验指标

从已有的实验详情 JSON 文件重新计算所有指标。
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics.retrieval_metrics import RetrievalMetrics
from src.evaluation.metrics.generation_metrics import GenerationMetrics
from src.llms.embedding_client import EmbeddingClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


def recalculate_metrics(
    details_path: str,
    output_dir: str,
    top_k_values: Optional[List[int]] = None,
    recalculate_generation: bool = False,
):
    """重新计算实验指标
    
    Args:
        details_path: 实验详情 JSON 文件路径
        output_dir: 输出目录
        top_k_values: 评估的 K 值列表
        recalculate_generation: 是否重新计算生成指标（需要 embedding）
    """
    top_k_values = top_k_values or [1, 3, 5, 10, 20, 50]
    
    details_path = Path(details_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"加载实验详情: {details_path}")
    with open(details_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    
    logger.info(f"加载完成: {len(all_results)} 条记录")
    
    retrieval_metrics = RetrievalMetrics(k_values=top_k_values)
    generation_metrics = GenerationMetrics()
    embedding_client = None
    
    if recalculate_generation:
        logger.info("加载嵌入客户端...")
        embedding_client = EmbeddingClient()
    
    aggregated_metrics = {
        "recall_at_k": {k: [] for k in top_k_values},
        "precision_at_k": {k: [] for k in top_k_values},
        "mrr": [],
        "ndcg": [],
        "map_score": [],
        "hit_rate": [],
        "title_recall": [],
        "title_precision": [],
        "exact_match": [],
        "f1_score": [],
        "semantic_similarity": [],
    }
    
    updated_results = []
    
    for i, result in enumerate(all_results):
        retrieved_titles = result.get("retrieved_titles", [])
        relevant_titles = result.get("relevant_titles", [])
        generated_answer = result.get("generated_answer", "")
        ground_truth = result.get("answer", "")
        
        new_metrics = retrieval_metrics.compute(retrieved_titles, relevant_titles)
        
        retrieved_set = set(retrieved_titles)
        relevant_set = set(relevant_titles)
        title_recall = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0
        title_precision = len(retrieved_set & relevant_set) / len(retrieved_set) if retrieved_set else 0
        
        if recalculate_generation and embedding_client:
            gen_result = generation_metrics.compute(
                predicted=generated_answer,
                ground_truth=ground_truth,
                compute_semantic=True,
                embedding_client=embedding_client,
            )
        else:
            gen_result = result.get("generation_metrics", {})
        
        for k in top_k_values:
            aggregated_metrics["recall_at_k"][k].append(
                new_metrics.recall_at_k.get(k, 0)
            )
            aggregated_metrics["precision_at_k"][k].append(
                new_metrics.precision_at_k.get(k, 0)
            )
        
        aggregated_metrics["mrr"].append(new_metrics.mrr)
        aggregated_metrics["ndcg"].append(new_metrics.ndcg)
        aggregated_metrics["map_score"].append(new_metrics.map_score)
        aggregated_metrics["hit_rate"].append(new_metrics.hit_rate)
        aggregated_metrics["title_recall"].append(title_recall)
        aggregated_metrics["title_precision"].append(title_precision)
        
        if isinstance(gen_result, dict):
            aggregated_metrics["exact_match"].append(gen_result.get("exact_match", 0))
            aggregated_metrics["f1_score"].append(gen_result.get("f1_score", 0))
            aggregated_metrics["semantic_similarity"].append(gen_result.get("semantic_similarity", 0))
        elif gen_result is not None:
            aggregated_metrics["exact_match"].append(gen_result.exact_match)
            aggregated_metrics["f1_score"].append(gen_result.f1_score)
            aggregated_metrics["semantic_similarity"].append(gen_result.semantic_similarity)
        else:
            aggregated_metrics["exact_match"].append(0)
            aggregated_metrics["f1_score"].append(0)
            aggregated_metrics["semantic_similarity"].append(0)
        
        updated_results.append({
            **result,
            "metrics": new_metrics.to_dict(),
            "title_recall": title_recall,
            "title_precision": title_precision,
        })
        
        if (i + 1) % 1000 == 0:
            logger.info(f"已处理 {i + 1}/{len(all_results)} 条记录")
    
    final_metrics = {
        "recall_at_k": {
            k: np.mean(v) for k, v in aggregated_metrics["recall_at_k"].items()
        },
        "precision_at_k": {
            k: np.mean(v) for k, v in aggregated_metrics["precision_at_k"].items()
        },
        "mrr": np.mean(aggregated_metrics["mrr"]),
        "ndcg": np.mean(aggregated_metrics["ndcg"]),
        "map_score": np.mean(aggregated_metrics["map_score"]),
        "hit_rate": np.mean(aggregated_metrics["hit_rate"]),
        "title_recall": np.mean(aggregated_metrics["title_recall"]),
        "title_precision": np.mean(aggregated_metrics["title_precision"]),
        "exact_match": np.mean(aggregated_metrics["exact_match"]),
        "f1_score": np.mean(aggregated_metrics["f1_score"]),
        "semantic_similarity": np.nanmean(aggregated_metrics["semantic_similarity"]),
    }
    
    baseline_recall = 0.1
    recall_improvement = {
        k: (v - baseline_recall) / baseline_recall * 100 if baseline_recall > 0 else 0
        for k, v in final_metrics["recall_at_k"].items()
    }
    final_metrics["recall_improvement"] = recall_improvement
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary = {
        "experiment_name": "recalculated_metrics",
        "timestamp": datetime.now().isoformat(),
        "source_file": str(details_path),
        "total_samples": len(all_results),
        "top_k_values": top_k_values,
        "metrics": final_metrics,
    }
    
    summary_path = output_dir / f"recalculated_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"汇总结果已保存: {summary_path}")
    
    details_path = output_dir / f"recalculated_details_{timestamp}.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(updated_results, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果已保存: {details_path}")
    
    metrics_rows = [
        {
            "Metric": "Recall@K",
            **{f"K={k}": v for k, v in final_metrics["recall_at_k"].items()},
        },
        {
            "Metric": "Precision@K",
            **{f"K={k}": v for k, v in final_metrics["precision_at_k"].items()},
        },
        {
            "Metric": "Recall提升率(%)",
            **{f"K={k}": v for k, v in final_metrics["recall_improvement"].items()},
        },
        {
            "Metric": "MRR",
            "Value": final_metrics["mrr"],
        },
        {
            "Metric": "NDCG",
            "Value": final_metrics["ndcg"],
        },
        {
            "Metric": "MAP",
            "Value": final_metrics["map_score"],
        },
        {
            "Metric": "Hit Rate",
            "Value": final_metrics["hit_rate"],
        },
        {
            "Metric": "Title Recall",
            "Value": final_metrics["title_recall"],
        },
        {
            "Metric": "Title Precision",
            "Value": final_metrics["title_precision"],
        },
        {
            "Metric": "Exact Match",
            "Value": final_metrics["exact_match"],
        },
        {
            "Metric": "F1 Score",
            "Value": final_metrics["f1_score"],
        },
        {
            "Metric": "Semantic Similarity",
            "Value": final_metrics["semantic_similarity"],
        },
    ]
    
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / f"recalculated_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"指标表格已保存: {metrics_path}")
    
    print("\n" + "=" * 60)
    print("重新计算结果")
    print("=" * 60)
    
    print(f"\n【数据统计】")
    print(f"  总样本数: {len(all_results)}")
    print(f"  K 值列表: {top_k_values}")
    
    print("\n【Recall@K】")
    for k, v in final_metrics["recall_at_k"].items():
        print(f"  Recall@{k}: {v:.4f}")
    
    print("\n【Precision@K】")
    for k, v in final_metrics["precision_at_k"].items():
        print(f"  Precision@{k}: {v:.4f}")
    
    print("\n【综合指标】")
    print(f"  MRR: {final_metrics['mrr']:.4f}")
    print(f"  NDCG: {final_metrics['ndcg']:.4f}")
    print(f"  MAP: {final_metrics['map_score']:.4f}")
    print(f"  Hit Rate: {final_metrics['hit_rate']:.4f}")
    
    print("\n【标题级别指标】")
    print(f"  Title Recall: {final_metrics['title_recall']:.4f}")
    print(f"  Title Precision: {final_metrics['title_precision']:.4f}")
    
    print("\n【生成指标】")
    print(f"  Exact Match: {final_metrics['exact_match']:.4f}")
    print(f"  F1 Score: {final_metrics['f1_score']:.4f}")
    print(f"  Semantic Similarity: {final_metrics['semantic_similarity']:.4f}")
    
    print("\n" + "=" * 60)
    
    return summary


if __name__ == "__main__":
    details_path = "e:/Code_Personal/Subject/test02/experiments/exp5_2hop_expansion/experiment_details_20260329_120921.json"
    output_dir = "e:/Code_Personal/Subject/test02/experiments/exp5_2hop_expansion"
    
    recalculate_metrics(
        details_path=details_path,
        output_dir=output_dir,
        top_k_values=[1, 3, 5, 7, 10, 20, 50],
        recalculate_generation=False,
    )
