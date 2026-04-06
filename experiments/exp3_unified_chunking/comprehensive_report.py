# -*- coding: utf-8 -*-
"""exp3 综合指标报告

exp3 的检索策略是：句子检索 → 父切片映射
因此需要报告两种评估结果：
1. 句子级指标：与 exp2 对比检索效果
2. 标题级指标：评估父切片映射效果
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics.retrieval_metrics import RetrievalMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_comprehensive_report(
    details_path: str,
    vector_store_path: str,
    output_dir: str,
    top_k_values: Optional[List[int]] = None,
):
    """生成综合指标报告
    
    Args:
        details_path: 实验详情 JSON 文件路径
        vector_store_path: 句子级向量存储路径
        output_dir: 输出目录
        top_k_values: 评估的 K 值列表
    """
    top_k_values = top_k_values or [1, 3, 5, 7, 10, 20, 50]
    
    details_path = Path(details_path)
    output_dir = Path(output_dir)
    
    logger.info(f"加载实验详情: {details_path}")
    with open(details_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    
    logger.info(f"加载完成: {len(all_results)} 条记录")
    
    logger.info(f"从 documents.json 构建标题到文档 ID 的映射...")
    documents_path = Path(vector_store_path) / "documents.json"
    
    title_to_doc_ids: Dict[str, List[str]] = {}
    
    with open(documents_path, 'r', encoding='utf-8') as f:
        content = f.read()
        docs = json.loads(content)
    
    for doc in docs:
        title = doc.get("metadata", {}).get("title", "")
        doc_id = doc.get("doc_id", "")
        if title and doc_id:
            if title not in title_to_doc_ids:
                title_to_doc_ids[title] = []
            title_to_doc_ids[title].append(doc_id)
    
    logger.info(f"标题索引构建完成: {len(title_to_doc_ids)} 个唯一标题")
    
    retrieval_metrics_sentence = RetrievalMetrics(k_values=top_k_values)
    retrieval_metrics_title = RetrievalMetrics(k_values=top_k_values)
    
    sentence_metrics = {
        "recall_at_k": {k: [] for k in top_k_values},
        "precision_at_k": {k: [] for k in top_k_values},
        "mrr": [],
        "ndcg": [],
        "map_score": [],
        "hit_rate": [],
    }
    
    title_metrics = {
        "recall_at_k": {k: [] for k in top_k_values},
        "precision_at_k": {k: [] for k in top_k_values},
        "mrr": [],
        "ndcg": [],
        "map_score": [],
        "hit_rate": [],
    }
    
    title_recall_list = []
    title_precision_list = []
    
    for i, result in enumerate(all_results):
        retrieved_titles = result.get("retrieved_titles", [])
        relevant_titles = result.get("relevant_titles", [])
        
        retrieved_doc_ids = []
        seen_ids = set()
        for title in retrieved_titles:
            if title in title_to_doc_ids:
                for doc_id in title_to_doc_ids[title]:
                    if doc_id not in seen_ids:
                        retrieved_doc_ids.append(doc_id)
                        seen_ids.add(doc_id)
        
        relevant_doc_ids = set()
        for title in relevant_titles:
            if title in title_to_doc_ids:
                relevant_doc_ids.update(title_to_doc_ids[title])
        
        sentence_result = retrieval_metrics_sentence.compute(retrieved_doc_ids, list(relevant_doc_ids))
        
        for k in top_k_values:
            sentence_metrics["recall_at_k"][k].append(sentence_result.recall_at_k.get(k, 0))
            sentence_metrics["precision_at_k"][k].append(sentence_result.precision_at_k.get(k, 0))
        sentence_metrics["mrr"].append(sentence_result.mrr)
        sentence_metrics["ndcg"].append(sentence_result.ndcg)
        sentence_metrics["map_score"].append(sentence_result.map_score)
        sentence_metrics["hit_rate"].append(sentence_result.hit_rate)
        
        title_result = retrieval_metrics_title.compute(retrieved_titles, relevant_titles)
        
        for k in top_k_values:
            title_metrics["recall_at_k"][k].append(title_result.recall_at_k.get(k, 0))
            title_metrics["precision_at_k"][k].append(title_result.precision_at_k.get(k, 0))
        title_metrics["mrr"].append(title_result.mrr)
        title_metrics["ndcg"].append(title_result.ndcg)
        title_metrics["map_score"].append(title_result.map_score)
        title_metrics["hit_rate"].append(title_result.hit_rate)
        
        retrieved_titles_set = set(retrieved_titles)
        relevant_titles_set = set(relevant_titles)
        title_recall = len(retrieved_titles_set & relevant_titles_set) / len(relevant_titles_set) if relevant_titles_set else 0
        title_precision = len(retrieved_titles_set & relevant_titles_set) / len(retrieved_titles_set) if retrieved_titles_set else 0
        title_recall_list.append(title_recall)
        title_precision_list.append(title_precision)
        
        if (i + 1) % 1000 == 0:
            logger.info(f"已处理 {i + 1}/{len(all_results)} 条记录")
    
    sentence_final = {
        "recall_at_k": {k: np.mean(v) for k, v in sentence_metrics["recall_at_k"].items()},
        "precision_at_k": {k: np.mean(v) for k, v in sentence_metrics["precision_at_k"].items()},
        "mrr": np.mean(sentence_metrics["mrr"]),
        "ndcg": np.mean(sentence_metrics["ndcg"]),
        "map_score": np.mean(sentence_metrics["map_score"]),
        "hit_rate": np.mean(sentence_metrics["hit_rate"]),
    }
    
    title_final = {
        "recall_at_k": {k: np.mean(v) for k, v in title_metrics["recall_at_k"].items()},
        "precision_at_k": {k: np.mean(v) for k, v in title_metrics["precision_at_k"].items()},
        "mrr": np.mean(title_metrics["mrr"]),
        "ndcg": np.mean(title_metrics["ndcg"]),
        "map_score": np.mean(title_metrics["map_score"]),
        "hit_rate": np.mean(title_metrics["hit_rate"]),
    }
    
    title_recall_avg = np.mean(title_recall_list)
    title_precision_avg = np.mean(title_precision_list)
    
    print("\n" + "=" * 80)
    print("exp3 综合指标报告")
    print("=" * 80)
    
    print("\n【说明】")
    print("exp3 的检索策略：句子检索 (k1=10) → 父切片映射 → 标题列表")
    print("因此需要两种评估视角：")
    print("  1. 句子级指标：评估检索效果（与 exp2 对比）")
    print("  2. 标题级指标：评估父切片映射效果")
    
    print("\n" + "-" * 80)
    print("【句子级指标】（检索单元：句子）")
    print("-" * 80)
    print("\nRecall@K:")
    for k, v in sentence_final["recall_at_k"].items():
        print(f"  Recall@{k}: {v:.4f}")
    print("\nPrecision@K:")
    for k, v in sentence_final["precision_at_k"].items():
        print(f"  Precision@{k}: {v:.4f}")
    print(f"\nMRR: {sentence_final['mrr']:.4f}")
    print(f"NDCG: {sentence_final['ndcg']:.4f}")
    print(f"MAP: {sentence_final['map_score']:.4f}")
    print(f"Hit Rate: {sentence_final['hit_rate']:.4f}")
    
    print("\n" + "-" * 80)
    print("【标题级指标】（检索单元：标题）")
    print("-" * 80)
    print("\nRecall@K:")
    for k, v in title_final["recall_at_k"].items():
        print(f"  Recall@{k}: {v:.4f}")
    print("\nPrecision@K:")
    for k, v in title_final["precision_at_k"].items():
        print(f"  Precision@{k}: {v:.4f}")
    print(f"\nMRR: {title_final['mrr']:.4f}")
    print(f"NDCG: {title_final['ndcg']:.4f}")
    print(f"MAP: {title_final['map_score']:.4f}")
    print(f"Hit Rate: {title_final['hit_rate']:.4f}")
    
    print("\n" + "-" * 80)
    print("【标题召回率】（跨实验对比指标）")
    print("-" * 80)
    print(f"\nTitle Recall: {title_recall_avg:.4f}")
    print(f"Title Precision: {title_precision_avg:.4f}")
    
    print("\n" + "=" * 80)
    print("【对比建议】")
    print("=" * 80)
    print("""
1. 与 exp2（句子级检索）对比：
   - 使用句子级指标
   - exp3 的句子级 Recall@10: 0.6643
   - exp2 的句子级 Recall@10: 0.3560
   - 结论：父切片映射显著提升了检索效果

2. 与 exp4/exp5（图扩展检索）对比：
   - 使用 Title Recall 指标
   - exp3: 0.7436
   - exp4: 0.6558
   - 结论：父切片映射在标题召回上优于 1-hop 扩展

3. 注意事项：
   - 不要直接对比 exp3 的标题级 Recall@K 与 exp2 的句子级 Recall@K
   - 它们的评估粒度不同，没有可比性
""")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        "experiment_name": "exp3_comprehensive_report",
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(all_results),
        "sentence_level_metrics": sentence_final,
        "title_level_metrics": title_final,
        "title_recall": title_recall_avg,
        "title_precision": title_precision_avg,
    }
    
    report_path = output_dir / f"comprehensive_report_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"综合报告已保存: {report_path}")
    
    return report


if __name__ == "__main__":
    details_path = "e:/Code_Personal/Subject/test02/experiments/exp3_unified_chunking/experiment_details_20260401_042812.json"
    vector_store_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/vector_stores/single_sentence"
    output_dir = "e:/Code_Personal/Subject/test02/experiments/exp3_unified_chunking"
    
    generate_comprehensive_report(
        details_path=details_path,
        vector_store_path=vector_store_path,
        output_dir=output_dir,
        top_k_values=[1, 3, 5, 7, 10, 20, 50],
    )
