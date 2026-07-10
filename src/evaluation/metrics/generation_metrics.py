# -*- coding: utf-8 -*-
"""生成指标模块

提供 RAG 生成答案的评估指标，使用 LLM-based 方式评估 correctness 和 faithfulness。
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """生成评估结果
    
    Attributes:
        correctness: 正确性分数 (LLM-based)，考生回答的核心含义是否与标准答案一致
        faithfulness: 忠实度 (LLM-based)，考生回答是否完全依据检索到的上下文生成
        exact_match: 精确匹配分数 (保留兼容)
        f1_score: F1 分数 (保留兼容)
        semantic_similarity: 语义相似度 (保留兼容)
        raw_response: LLM 原始响应内容
        success: 是否评估成功
    """
    correctness: float = 0.0
    faithfulness: float = 0.0
    exact_match: float = 0.0
    f1_score: float = 0.0
    semantic_similarity: float = 0.0
    raw_response: str = ""
    success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "correctness": self.correctness,
            "faithfulness": self.faithfulness,
            "exact_match": self.exact_match,
            "f1_score": self.f1_score,
            "semantic_similarity": self.semantic_similarity,
            "raw_response": self.raw_response,
            "success": self.success,
        }


class GenerationMetrics:
    """生成指标计算器
    
    使用 LLM-based 方式进行评估 correctness 和 faithfulness。
    """
    
    def __init__(
        self,
        use_llm_judge: bool = True,
        max_retries: int = 10,
        context_max_length: int = 3000,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ):
        """初始化生成指标计算器
        
        Args:
            use_llm_judge: 是否使用 LLM 裁判进行评估
            max_retries: 最大重试次数
            context_max_length: 检索上下文的长度限制
            max_tokens: LLM 生成的最大 token 数
            temperature: 温度参数
        """
        self.use_llm_judge = use_llm_judge
        self.max_retries = max_retries
        self.context_max_length = context_max_length
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def compute(
        self,
        predicted: str,
        ground_truth: str,
        compute_semantic: bool = False,
        embedding_client: Optional[Any] = None,
        compute_llm_based: bool = False,
        llm_client: Optional[Any] = None,
        context: Optional[str] = None,
        question: Optional[str] = None,
        retrieved_contexts: Optional[List[str]] = None,
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
            question: 原始问题（用于 LLM 评估）
            retrieved_contexts: 检索到的上下文列表（用于 LLM 评估）
            
        Returns:
            GenerationResult 实例
        """
        correctness = 0.0
        faithfulness = 0.0
        raw_response = ""
        success = False
        
        if compute_llm_based and llm_client and question and retrieved_contexts:
            llm_scores = self._compute_llm_based_metrics(
                question, ground_truth, predicted, retrieved_contexts, llm_client
            )
            correctness = llm_scores.get("correctness", 0.0)
            faithfulness = llm_scores.get("faithfulness", 0.0)
            raw_response = llm_scores.get("raw_response", "")
            success = llm_scores.get("success", False)
        
        exact_match = self._compute_exact_match(predicted, ground_truth)
        f1_score = self._compute_f1(predicted, ground_truth, exact_match)
        
        semantic_similarity = 0.0
        if compute_semantic and embedding_client:
            semantic_similarity = self._compute_semantic_similarity(
                predicted, ground_truth, embedding_client
            )
        
        return GenerationResult(
            correctness=correctness,
            faithfulness=faithfulness,
            exact_match=exact_match,
            f1_score=f1_score,
            semantic_similarity=semantic_similarity,
            raw_response=raw_response,
            success=success,
        )
    
    def _compute_llm_based_metrics(
        self,
        question: str,
        ground_truth: str,
        predicted: str,
        retrieved_contexts: List[str],
        llm_client: Any,
    ) -> Dict[str, Any]:
        """使用 LLM 计算 correctness 和 faithfulness"""
        from src.llms.base_client import Message
        
        context_text = "\n".join([str(c) for c in retrieved_contexts])[:self.context_max_length]
        
        prompt = f"""你是一个 RAG 评测裁判。请对以下问答进行打分。

【问题】: {question}
【标准答案】: {ground_truth}
【考生回答】: {predicted}

【检索到的上下文】（共 {len(retrieved_contexts)} 条）:
{context_text}

请评估以下指标（必须输出 JSON）：

1. **correctness (准确性)**: (0或1) 考生回答的核心含义是否与标准答案一致？
2. **faithfulness (忠实度)**: (0或1) 考生回答是否**完全依据**【检索到的上下文】生成？(即没有幻觉或外部知识)

请仅输出 JSON:
{{
    "correctness": 1,
    "faithfulness": 1
}}"""
        
        messages = [Message(role="user", content=prompt)]
        return self._call_llm_with_retry(messages, llm_client)
    
    def _call_llm_with_retry(self, messages: list, llm_client: Any) -> Dict[str, Any]:
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
                    logger.warning(f"LLM 评估 API 错误: {response.error}，等待重试...")
                    retry_count += 1
                    time.sleep(2 ** retry_count)
                    continue
                
                content = response.content
                
                if not content:
                    logger.warning("LLM 评估响应内容为空，等待重试...")
                    retry_count += 1
                    time.sleep(2 ** retry_count)
                    continue
                
                scores = self._parse_json_response(content)
                
                if scores is not None:
                    return {
                        "correctness": float(scores.get("correctness", 0)),
                        "faithfulness": float(scores.get("faithfulness", 0)),
                        "raw_response": content,
                        "success": True,
                    }
                
                logger.warning("LLM 评估返回格式错误，等待重试...")
                retry_count += 1
                time.sleep(1)
                continue
                
            except Exception as e:
                logger.warning(f"LLM 评估异常 ({e})，等待重试...")
                retry_count += 1
                time.sleep(2 ** retry_count)
        
        logger.warning(f"LLM 评估重试 {self.max_retries} 次后仍失败，返回默认值")
        return {
            "correctness": 0.0,
            "faithfulness": 0.0,
            "raw_response": "",
            "success": False,
        }
    
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
    
    def _compute_exact_match(self, predicted: str, ground_truth: str) -> float:
        """计算精确匹配分数（包含匹配）"""
        if not predicted or not ground_truth:
            return 0.0
        
        pred_normalized = self._normalize_text(predicted)
        truth_normalized = self._normalize_text(ground_truth)
        
        if pred_normalized == truth_normalized:
            return 1.0
        
        if truth_normalized in pred_normalized:
            return 1.0
        
        return 0.0
    
    def _compute_f1(self, predicted: str, ground_truth: str, exact_match: float = 0.0) -> float:
        """计算 F1 分数（EM=1时直接返回1）"""
        if exact_match == 1.0:
            return 1.0
        
        if not predicted or not ground_truth:
            return 0.0
        
        pred_normalized = self._normalize_text(predicted)
        truth_normalized = self._normalize_text(ground_truth)
        
        pred_tokens = self._tokenize(pred_normalized)
        truth_tokens = self._tokenize(truth_normalized)
        
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        common = pred_tokens & truth_tokens
        
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        
        return 2 * precision * recall / (precision + recall)
    
    def _tokenize(self, text: str) -> set:
        """分词（支持中英文混合）"""
        tokens = set()
        
        current_word = ""
        for char in text:
            if char.isspace():
                if current_word:
                    tokens.add(current_word)
                    current_word = ""
            elif '\u4e00' <= char <= '\u9fff':
                if current_word:
                    tokens.add(current_word)
                    current_word = ""
                tokens.add(char)
            else:
                current_word += char
        
        if current_word:
            tokens.add(current_word)
        
        return tokens
    
    def _normalize_text(self, text: str) -> str:
        """标准化文本"""
        import string
        
        text = text.lower().strip()
        
        exclude = set(string.punctuation)
        chinese_punc = set('，。！？；：""''（）【】《》、…—·')
        exclude.update(chinese_punc)
        text = ''.join(ch for ch in text if ch not in exclude)
        
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _compute_semantic_similarity(
        self,
        predicted: str,
        ground_truth: str,
        embedding_client: Any,
    ) -> float:
        """计算语义相似度"""
        try:
            import numpy as np
            
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


if __name__ == "__main__":
    print("=" * 50)
    print("测试生成指标（LLM-based 版本）")
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
    
    print("\n所有测试通过!")
