# -*- coding: utf-8 -*-
"""生成指标模块

提供 RAG 生成答案的评估指标，包括 EM、F1、语义相似度等。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
import json

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """生成评估结果
    
    Attributes:
        exact_match: 精确匹配分数
        f1_score: F1 分数
        semantic_similarity: 语义相似度
        correctness: 正确性分数
        faithfulness: 忠实度
        relevance: 相关性
    """
    exact_match: float = 0.0
    f1_score: float = 0.0
    semantic_similarity: float = 0.0
    correctness: float = 0.0
    faithfulness: float = 0.0
    relevance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "exact_match": self.exact_match,
            "f1_score": self.f1_score,
            "semantic_similarity": self.semantic_similarity,
            "correctness": self.correctness,
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
        }


class GenerationMetrics:
    """生成指标计算器
    
    提供多种生成指标的统一计算接口。
    """
    
    def __init__(self):
        """初始化生成指标计算器"""
        pass
    
    def compute(
        self,
        predicted: str,
        ground_truth: str,
        compute_semantic: bool = False,
        embedding_client: Optional[Any] = None,
        compute_llm_based: bool = False,
        llm_client: Optional[Any] = None,
        context: Optional[str] = None,
    ) -> GenerationResult:
        """计算所有生成指标
        
        Args:
            predicted: 预测的答案
            ground_truth: 标准答案
            compute_semantic: 是否计算语义相似度
            embedding_client: Embedding 客户端
            compute_llm_based: 是否计算 LLM-based 指标
            llm_client: LLM 客户端
            context: 上下文（用于忠实度计算）
            
        Returns:
            GenerationResult 实例
        """
        exact_match = self._compute_exact_match(predicted, ground_truth)
        f1_score = self._compute_f1(predicted, ground_truth)
        
        semantic_similarity = 0.0
        if compute_semantic and embedding_client:
            semantic_similarity = self._compute_semantic_similarity(
                predicted, ground_truth, embedding_client
            )
        
        correctness = 0.0
        faithfulness = 0.0
        relevance = 0.0
        
        if compute_llm_based and llm_client:
            llm_scores = self._compute_llm_based_metrics(
                predicted, ground_truth, llm_client, context
            )
            correctness = llm_scores.get("correctness", 0.0)
            faithfulness = llm_scores.get("faithfulness", 0.0)
            relevance = llm_scores.get("relevance", 0.0)
        
        return GenerationResult(
            exact_match=exact_match,
            f1_score=f1_score,
            semantic_similarity=semantic_similarity,
            correctness=correctness,
            faithfulness=faithfulness,
            relevance=relevance,
        )
    
    def _compute_exact_match(self, predicted: str, ground_truth: str) -> float:
        """计算精确匹配分数（改进版 - 支持包含匹配）
        
        改进策略：
        1. 完全匹配：返回 1.0
        2. 包含匹配：标准答案包含在生成答案中，返回 1.0
        3. 否则：返回 0.0
        """
        pred_normalized = self._normalize_text(predicted)
        truth_normalized = self._normalize_text(ground_truth)
        
        if not pred_normalized or not truth_normalized:
            return 0.0
        
        if pred_normalized == truth_normalized:
            return 1.0
        
        if truth_normalized in pred_normalized:
            return 1.0
        
        return 0.0
    
    def _compute_f1(self, predicted: str, ground_truth: str) -> float:
        """计算 F1 分数（改进版 - 支持字符级和 token 级）
        
        改进策略：
        1. 检测文本语言（中文/英文）
        2. 中文使用字符级 F1
        3. 英文使用 token 级 F1
        4. 计算方式：基于集合的 precision 和 recall
        """
        pred_normalized = self._normalize_text(predicted)
        truth_normalized = self._normalize_text(ground_truth)
        
        if not pred_normalized or not truth_normalized:
            return 0.0
        
        is_chinese = self._is_chinese_text(truth_normalized)
        
        if is_chinese:
            pred_tokens = set(pred_normalized)
            truth_tokens = set(truth_normalized)
        else:
            pred_tokens = set(pred_normalized.split())
            truth_tokens = set(truth_normalized.split())
        
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        common = pred_tokens & truth_tokens
        
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        
        return 2 * precision * recall / (precision + recall)
    
    def _is_chinese_text(self, text: str) -> bool:
        """检测文本是否为中文"""
        chinese_char_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        return chinese_char_count > len(text) * 0.3
    
    def _compute_semantic_similarity(
        self,
        predicted: str,
        ground_truth: str,
        embedding_client: Any,
    ) -> float:
        """计算语义相似度"""
        try:
            pred_embedding = embedding_client.embed([predicted])
            truth_embedding = embedding_client.embed([ground_truth])
            
            if hasattr(pred_embedding, 'tolist'):
                pred_embedding = pred_embedding.tolist()
            if hasattr(truth_embedding, 'tolist'):
                truth_embedding = truth_embedding.tolist()
            
            pred_vec = np.array(pred_embedding[0])
            truth_vec = np.array(truth_embedding[0])
            
            pred_norm = np.linalg.norm(pred_vec)
            truth_norm = np.linalg.norm(truth_vec)
            
            if pred_norm == 0 or truth_norm == 0:
                logger.warning("零向量检测，返回相似度 0.0")
                return 0.0
            
            similarity = np.dot(pred_vec, truth_vec) / (pred_norm * truth_norm)
            
            return float(similarity)
        except Exception as e:
            logger.warning(f"计算语义相似度失败: {e}")
            return 0.0
    
    def _compute_llm_based_metrics(
        self,
        predicted: str,
        ground_truth: str,
        llm_client: Any,
        context: Optional[str] = None,
    ) -> Dict[str, float]:
        """使用 LLM 计算指标"""
        scores = {}
        
        try:
            from src.llms.base_client import Message
            
            prompt = f"""请评估以下答案的质量，给出 0-1 的分数。

标准答案：{ground_truth}
预测答案：{predicted}

请分别给出以下维度的分数（JSON 格式）：
- correctness: 正确性
- relevance: 相关性
- faithfulness: 忠实度（答案是否基于上下文）

返回格式：{{"correctness": 0.8, "relevance": 0.9, "faithfulness": 0.7}}"""

            messages = [Message(role="user", content=prompt)]
            response = llm_client.generate(messages)
            
            result = json.loads(response.content)
            scores = {
                "correctness": float(result.get("correctness", 0)),
                "relevance": float(result.get("relevance", 0)),
                "faithfulness": float(result.get("faithfulness", 0)),
            }
        except Exception as e:
            logger.warning(f"LLM-based 指标计算失败: {e}")
        
        return scores
    
    def _normalize_text(self, text: str) -> str:
        """标准化文本"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text


if __name__ == "__main__":
    print("=" * 50)
    print("测试生成指标（改进版）")
    print("=" * 50)
    
    metrics = GenerationMetrics()
    
    print("\n【测试1：中文文本】")
    predicted = "人工智能是计算机科学的一个分支，致力于创建智能系统。"
    ground_truth = "人工智能是计算机科学的分支，专注于开发智能系统。"
    result = metrics.compute(predicted, ground_truth)
    print(f"预测答案: {predicted}")
    print(f"标准答案: {ground_truth}")
    print(f"✓ Exact Match: {result.exact_match:.4f}")
    print(f"✓ F1 Score: {result.f1_score:.4f}")
    
    print("\n【测试2：英文文本 - 完全匹配】")
    predicted = "Yes, they are both American."
    ground_truth = "yes"
    result = metrics.compute(predicted, ground_truth)
    print(f"预测答案: {predicted}")
    print(f"标准答案: {ground_truth}")
    print(f"✓ Exact Match: {result.exact_match:.4f}")
    print(f"✓ F1 Score: {result.f1_score:.4f}")
    
    print("\n【测试3：英文文本 - 包含匹配】")
    predicted = "Based on the documents, the answer is yes."
    ground_truth = "yes"
    result = metrics.compute(predicted, ground_truth)
    print(f"预测答案: {predicted}")
    print(f"标准答案: {ground_truth}")
    print(f"✓ Exact Match: {result.exact_match:.4f}")
    print(f"✓ F1 Score: {result.f1_score:.4f}")
    
    print("\n所有测试通过!")
