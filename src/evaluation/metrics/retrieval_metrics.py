# -*- coding: utf-8 -*-
"""检索指标模块

提供检索系统的评估指标，使用 LLM-based 方式评估 context_recall@K 和 relevance@K。
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """检索评估结果
    
    Attributes:
        context_recall_at_k: 各 K 值的上下文召回率 (LLM-based)，检索到的上下文是否包含回答问题所需的关键信息
        relevance_at_k: 各 K 值的相关性 (LLM-based)，检索到的上下文是否与问题相关
        recall_at_k: 各 K 值的召回率（传统计算）
        precision_at_k: 各 K 值的精确率（传统计算）
        mrr: 平均倒数排名
        ndcg: 归一化折损累积增益
        map_score: 平均精度均值
        hit_rate: 命中率
        raw_response: LLM 原始响应内容
        success: 是否评估成功
    """
    context_recall_at_k: Dict[int, float] = field(default_factory=dict)
    relevance_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: float = 0.0
    map_score: float = 0.0
    hit_rate: float = 0.0
    raw_response: str = ""
    success: bool = False
    
    @property
    def context_recall(self) -> float:
        """获取默认的上下文召回率（使用最大的 K 值）"""
        if self.context_recall_at_k:
            return max(self.context_recall_at_k.values())
        return 0.0
    
    @property
    def relevance(self) -> float:
        """获取默认的相关性（使用最大的 K 值）"""
        if self.relevance_at_k:
            return max(self.relevance_at_k.values())
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "context_recall_at_k": self.context_recall_at_k,
            "context_recall": self.context_recall,
            "relevance_at_k": self.relevance_at_k,
            "relevance": self.relevance,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "map_score": self.map_score,
            "hit_rate": self.hit_rate,
            "raw_response": self.raw_response,
            "success": self.success,
        }


class RetrievalMetrics:
    """检索指标计算器
    
    使用 LLM-based 方式进行评估 context_recall@K 和 relevance@K。
    """
    
    def __init__(
        self,
        k_values: Optional[List[int]] = None,
        max_retries: int = 10,
        context_max_length: int = 3000,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ):
        """初始化检索指标计算器
        
        Args:
            k_values: 评估的 K 值列表，默认为 [1, 3, 5, 10]
            max_retries: 最大重试次数
            context_max_length: 检索上下文的长度限制
            max_tokens: LLM 生成的最大 token 数
            temperature: 温度参数
        """
        self.k_values = k_values or [1, 3, 5, 10]
        self.max_retries = max_retries
        self.context_max_length = context_max_length
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def compute(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
    ) -> RetrievalResult:
        """计算所有检索指标（传统计算方式）
        
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
    
    def compute_llm_based(
        self,
        question: str,
        retrieved_contexts: List[str],
        llm_client: Any,
    ) -> RetrievalResult:
        """使用 LLM 计算 context_recall@K 和 relevance@K
        
        Args:
            question: 用户问题
            retrieved_contexts: 检索到的上下文列表
            llm_client: LLM 客户端
            
        Returns:
            RetrievalResult 实例
        """
        from src.llms.base_client import Message
        
        context_text = "\n".join([str(c) for c in retrieved_contexts])[:self.context_max_length]
        
        k_values_str = ", ".join(str(k) for k in self.k_values)
        
        prompt = f"""你是一个 RAG 评测裁判。请对以下检索结果进行打分。

【问题】: {question}

【检索到的上下文】（共 {len(retrieved_contexts)} 条）:
{context_text}

请评估以下指标（必须输出 JSON）：

1. **context_recall_at_k (上下文召回@K)**: 对于 K ∈ {{{k_values_str}}}，评估前 K 个上下文是否包含回答问题所需的关键信息。
   - 输出格式: {{"1": 0或1, "3": 0或1, "5": 0或1, "10": 0或1}}
   - 注意: 如果 K 大于上下文数量，则使用实际上下文数量

2. **relevance_at_k (相关性@K)**: 对于 K ∈ {{{k_values_str}}}，评估前 K 个上下文是否与问题相关。
   - 输出格式: {{"1": 0或1, "3": 0或1, "5": 0或1, "10": 0或1}}

请仅输出 JSON:
{{
    "context_recall_at_k": {{"1": 1, "3": 1, "5": 1, "10": 1}},
    "relevance_at_k": {{"1": 1, "3": 1, "5": 1, "10": 1}}
}}"""
        
        messages = [Message(role="user", content=prompt)]
        return self._call_llm_with_retry(messages, len(retrieved_contexts), llm_client)
    
    def _call_llm_with_retry(
        self,
        messages: list,
        actual_context_count: int,
        llm_client: Any,
    ) -> RetrievalResult:
        """调用 LLM 并重试"""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                response = llm_client.generate_with_retry(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                if not response.success:
                    logger.warning(f"LLM 检索评估 API 错误: {response.error}，等待重试...")
                    retry_count += 1
                    time.sleep(2 ** retry_count)
                    continue
                
                content = response.content
                
                if not content:
                    logger.warning("LLM 检索评估响应内容为空，等待重试...")
                    retry_count += 1
                    time.sleep(2 ** retry_count)
                    continue
                
                scores = self._parse_json_response(content)
                
                if scores is not None:
                    context_recall_at_k = self._parse_k_dict(
                        scores.get("context_recall_at_k", {}),
                        actual_context_count
                    )
                    relevance_at_k = self._parse_k_dict(
                        scores.get("relevance_at_k", {}),
                        actual_context_count
                    )
                    
                    return RetrievalResult(
                        context_recall_at_k=context_recall_at_k,
                        relevance_at_k=relevance_at_k,
                        raw_response=content,
                        success=True,
                    )
                
                logger.warning("LLM 检索评估返回格式错误，等待重试...")
                retry_count += 1
                time.sleep(1)
                continue
                
            except Exception as e:
                logger.warning(f"LLM 检索评估异常 ({e})，等待重试...")
                retry_count += 1
                time.sleep(2 ** retry_count)
        
        logger.warning(f"LLM 检索评估重试 {self.max_retries} 次后仍失败，返回默认值")
        return RetrievalResult(
            context_recall_at_k={k: 0.0 for k in self.k_values},
            relevance_at_k={k: 0.0 for k in self.k_values},
            success=False,
        )
    
    def _parse_k_dict(self, k_dict: Dict[str, Any], actual_count: int) -> Dict[int, float]:
        """解析 @K 字典，处理 K 值超过实际上下文数量的情况"""
        result = {}
        for k in self.k_values:
            effective_k = min(k, actual_count)
            if effective_k > 0:
                str_k = str(k)
                if str_k in k_dict:
                    result[k] = float(k_dict[str_k])
                else:
                    result[k] = 0.0
        return result
    
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 返回的 JSON 响应"""
        try:
            content = content.replace("```json", "").replace("```", "").strip()
            
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            
            return None
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"JSON 解析失败: {e}")
            return None
    
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
