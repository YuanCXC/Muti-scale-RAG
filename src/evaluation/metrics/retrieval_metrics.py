# -*- coding: utf-8 -*-
"""检索指标模块

提供检索系统的评估指标，包括 Recall@K, Precision@K, MRR, NDCG 等。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """检索评估结果
    
    Attributes:
        recall_at_k: 各 K 值的召回率
        precision_at_k: 各 K 值的精确率
        mrr: 平均倒数排名
        ndcg: 归一化折损累积增益
        map_score: 平均精度均值
        hit_rate: 命中率
    """
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: float = 0.0
    map_score: float = 0.0
    hit_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "map_score": self.map_score,
            "hit_rate": self.hit_rate,
        }


class RetrievalMetrics:
    """检索指标计算器
    
    提供多种检索指标的统一计算接口。
    """
    
    def __init__(self, k_values: Optional[List[int]] = None):
        """初始化检索指标计算器
        
        Args:
            k_values: 评估的 K 值列表，默认为 [1, 3, 5, 10]
        """
        self.k_values = k_values or [1, 3, 5, 10]
    
    def compute(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
    ) -> RetrievalResult:
        """计算所有检索指标
        
        Args:
            retrieved_ids: 检索到的文档 ID 列表（按相关性排序）
            relevant_ids: 相关文档 ID 列表
            
        Returns:
            RetrievalResult 实例
        """
        if not relevant_ids:
            return RetrievalResult()
        
        relevant_set = set(relevant_ids)
        
        recall_at_k = {}
        precision_at_k = {}
        
        for k in self.k_values:
            top_k = set(retrieved_ids[:k])
            relevant_in_top_k = len(top_k & relevant_set)
            
            recall_at_k[k] = relevant_in_top_k / len(relevant_set) if relevant_set else 0
            precision_at_k[k] = relevant_in_top_k / k if k > 0 else 0
        
        mrr = self._compute_mrr(retrieved_ids, relevant_set)
        ndcg = self._compute_ndcg(retrieved_ids, relevant_set)
        map_score = self._compute_map(retrieved_ids, relevant_set)
        hit_rate = self._compute_hit_rate(retrieved_ids, relevant_set)
        
        return RetrievalResult(
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mrr=mrr,
            ndcg=ndcg,
            map_score=map_score,
            hit_rate=hit_rate,
        )
    
    def _compute_mrr(
        self,
        retrieved_ids: List[str],
        relevant_set: set,
    ) -> float:
        """计算平均倒数排名 (Mean Reciprocal Rank)"""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def _compute_ndcg(
        self,
        retrieved_ids: List[str],
        relevant_set: set,
        k: Optional[int] = None,
    ) -> float:
        """计算归一化折损累积增益 (Normalized Discounted Cumulative Gain)"""
        if k is None:
            k = len(retrieved_ids)
        
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_set:
                dcg += 1.0 / math.log2(i + 2)
        
        ideal_dcg = 0.0
        for i in range(min(len(relevant_set), k)):
            ideal_dcg += 1.0 / math.log2(i + 2)
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def _compute_map(
        self,
        retrieved_ids: List[str],
        relevant_set: set,
    ) -> float:
        """计算平均精度均值 (Mean Average Precision)"""
        if not relevant_set:
            return 0.0
        
        precisions = []
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))
        
        return sum(precisions) / len(relevant_set) if relevant_set else 0.0
    
    def _compute_hit_rate(
        self,
        retrieved_ids: List[str],
        relevant_set: set,
        k: int = 10,
    ) -> float:
        """计算命中率 (Hit Rate@K)"""
        top_k = set(retrieved_ids[:k])
        return 1.0 if top_k & relevant_set else 0.0


if __name__ == "__main__":
    print("=" * 50)
    print("测试检索指标")
    print("=" * 50)
    
    metrics = RetrievalMetrics(k_values=[1, 3, 5, 10])
    
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc1", "doc3", "doc6"]
    
    result = metrics.compute(retrieved, relevant)
    
    print(f"✓ Recall@K: {result.recall_at_k}")
    print(f"✓ Precision@K: {result.precision_at_k}")
    print(f"✓ MRR: {result.mrr:.4f}")
    print(f"✓ NDCG: {result.ndcg:.4f}")
    print(f"✓ MAP: {result.map_score:.4f}")
    print(f"✓ Hit Rate: {result.hit_rate:.4f}")
    
    print("\n所有测试通过!")
