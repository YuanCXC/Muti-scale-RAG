# -*- coding: utf-8 -*-
"""混合检索器实现

整合向量检索和关键词检索，使用 RRF 融合算法。
"""

from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

from src.retrievers.base_retriever import RetrieverBase, SearchResult
from src.retrievers.keyword_retriever import KeywordRetriever
from src.retrievers.vector_retriever import VectorRetriever

logger = get_logger(__name__)


class HybridRetriever(RetrieverBase):
    """混合检索器
    
    整合向量检索和关键词检索，使用 RRF (Reciprocal Rank Fusion) 融合算法。
    
    Attributes:
        vector_retriever: 向量检索器实例
        keyword_retriever: 关键词检索器实例
        vector_weight: 向量检索权重
        keyword_weight: 关键词检索权重
        rrf_k: RRF 算法的 k 参数
    """
    
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        rrf_k: int = 60,
        **kwargs: Any,
    ):
        """初始化混合检索器
        
        Args:
            vector_retriever: 向量检索器实例
            keyword_retriever: 关键词检索器实例
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            rrf_k: RRF 算法的 k 参数，通常为 60
            **kwargs: 额外参数
        """
        super().__init__(**kwargs)
        
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        
        # 验证权重
        if abs(vector_weight + keyword_weight - 1.0) > 0.01:
            logger.warning(
                f"权重总和不为 1.0: vector={vector_weight}, "
                f"keyword={keyword_weight}"
            )
        
        logger.info(
            f"初始化混合检索器: top_k={self.top_k}, "
            f"vector_weight={vector_weight}, keyword_weight={keyword_weight}, "
            f"rrf_k={rrf_k}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        vector_top_k: Optional[int] = None,
        keyword_top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """执行混合检索
        
        Args:
            query: 查询文本
            top_k: 最终返回结果数量（可选）
            vector_top_k: 向量检索返回数量（可选）
            keyword_top_k: 关键词检索返回数量（可选）
            **kwargs: 额外参数
            
        Returns:
            融合后的检索结果列表，按相关性分数降序排列
        """
        actual_top_k = self._get_top_k(top_k)
        
        # 默认每个检索器返回更多结果用于融合
        v_top_k = vector_top_k or min(actual_top_k * 2, 50)
        k_top_k = keyword_top_k or min(actual_top_k * 2, 50)
        
        # 并行执行两种检索
        vector_results = self.vector_retriever.retrieve(
            query=query,
            top_k=v_top_k,
            **kwargs,
        )
        
        keyword_results = self.keyword_retriever.retrieve(
            query=query,
            top_k=k_top_k,
        )
        
        # 使用 RRF 算法融合结果
        fused_results = self._rrf_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
        )
        
        # 截断到 top_k
        fused_results = fused_results[:actual_top_k]
        
        # 根据阈值过滤
        fused_results = self._filter_by_threshold(fused_results)
        
        logger.info(
            f"混合检索完成: query='{query[:50]}...', "
            f"vector_results={len(vector_results)}, "
            f"keyword_results={len(keyword_results)}, "
            f"fused_results={len(fused_results)}"
        )
        
        return fused_results
    
    def _rrf_fusion(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
    ) -> List[SearchResult]:
        """使用 RRF (Reciprocal Rank Fusion) 算法融合结果
        
        RRF 公式: score(d) = sum(1 / (k + rank(d)))
        
        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            
        Returns:
            融合后的结果列表
        """
        # 收集所有文档
        doc_scores: Dict[str, Dict[str, Any]] = {}
        
        # 处理向量检索结果
        for rank, result in enumerate(vector_results, start=1):
            if result.doc_id not in doc_scores:
                doc_scores[result.doc_id] = {
                    "content": result.content,
                    "metadata": result.metadata,
                    "vector_rank": rank,
                    "keyword_rank": float('inf'),
                    "vector_score": result.score,
                    "keyword_score": 0.0,
                }
            else:
                doc_scores[result.doc_id]["vector_rank"] = rank
                doc_scores[result.doc_id]["vector_score"] = result.score
        
        # 处理关键词检索结果
        for rank, result in enumerate(keyword_results, start=1):
            if result.doc_id not in doc_scores:
                doc_scores[result.doc_id] = {
                    "content": result.content,
                    "metadata": result.metadata,
                    "vector_rank": float('inf'),
                    "keyword_rank": rank,
                    "vector_score": 0.0,
                    "keyword_score": result.score,
                }
            else:
                doc_scores[result.doc_id]["keyword_rank"] = rank
                doc_scores[result.doc_id]["keyword_score"] = result.score
        
        # 计算 RRF 分数
        fused_results = []
        
        for doc_id, info in doc_scores.items():
            # 向量检索贡献
            vector_rrf = 0.0
            if info["vector_rank"] != float('inf'):
                vector_rrf = self.vector_weight / (self.rrf_k + info["vector_rank"])
            
            # 关键词检索贡献
            keyword_rrf = 0.0
            if info["keyword_rank"] != float('inf'):
                keyword_rrf = self.keyword_weight / (self.rrf_k + info["keyword_rank"])
            
            # 总分
            total_score = vector_rrf + keyword_rrf
            
            result = SearchResult(
                doc_id=doc_id,
                content=info["content"],
                score=total_score,
                metadata={
                    **info["metadata"],
                    "vector_rank": info["vector_rank"] if info["vector_rank"] != float('inf') else None,
                    "keyword_rank": info["keyword_rank"] if info["keyword_rank"] != float('inf') else None,
                    "vector_score": info["vector_score"],
                    "keyword_score": info["keyword_score"],
                },
            )
            fused_results.append(result)
        
        # 按分数降序排序
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        return fused_results
    
    def _weighted_fusion(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
    ) -> List[SearchResult]:
        """使用加权分数融合结果
        
        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            
        Returns:
            融合后的结果列表
        """
        # 收集所有文档
        doc_scores: Dict[str, Dict[str, Any]] = {}
        
        # 处理向量检索结果
        for result in vector_results:
            if result.doc_id not in doc_scores:
                doc_scores[result.doc_id] = {
                    "content": result.content,
                    "metadata": result.metadata,
                    "vector_score": result.score,
                    "keyword_score": 0.0,
                }
            else:
                doc_scores[result.doc_id]["vector_score"] = result.score
        
        # 处理关键词检索结果
        for result in keyword_results:
            if result.doc_id not in doc_scores:
                doc_scores[result.doc_id] = {
                    "content": result.content,
                    "metadata": result.metadata,
                    "vector_score": 0.0,
                    "keyword_score": result.score,
                }
            else:
                doc_scores[result.doc_id]["keyword_score"] = result.score
        
        # 归一化分数并计算加权总分
        # 注意：这里假设分数已经归一化到 [0, 1] 范围
        fused_results = []
        
        for doc_id, info in doc_scores.items():
            total_score = (
                self.vector_weight * info["vector_score"] +
                self.keyword_weight * info["keyword_score"]
            )
            
            result = SearchResult(
                doc_id=doc_id,
                content=info["content"],
                score=total_score,
                metadata={
                    **info["metadata"],
                    "vector_score": info["vector_score"],
                    "keyword_score": info["keyword_score"],
                },
            )
            fused_results.append(result)
        
        # 按分数降序排序
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        return fused_results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "type": "hybrid",
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "rrf_k": self.rrf_k,
            "vector_retriever": self.vector_retriever.get_stats(),
            "keyword_retriever": self.keyword_retriever.get_stats(),
        }


if __name__ == "__main__":
    import numpy as np
    from src.storage.vector_store import FAISSVectorStore, IndexType
    from src.retrievers.vector_retriever import VectorRetriever
    from src.retrievers.keyword_retriever import KeywordRetriever
    
    print("=" * 50)
    print("测试 Hybrid Retriever 模块")
    print("=" * 50)
    
    # Mock LLM Client for testing
    class MockLLMClient:
        def embed(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            return [np.random.randn(64).astype(np.float32).tolist() for _ in texts]
    
    mock_llm = MockLLMClient()
    
    # 测试文档
    docs = [
        "人工智能是计算机科学的一个分支。",
        "机器学习是人工智能的核心技术。",
        "深度学习使用多层神经网络进行学习。",
        "自然语言处理处理人类语言。",
    ]
    vectors = np.random.randn(4, 64).astype(np.float32)
    
    # 测试 1: 创建向量检索器
    vector_store = FAISSVectorStore(dimension=64, metric="cosine")
    vector_retriever = VectorRetriever(
        vector_store=vector_store,
        llm_client=mock_llm,
        top_k=3
    )
    
    # 构造文档格式
    doc_list = [
        {"doc_id": "d1", "content": docs[0], "metadata": {}},
        {"doc_id": "d2", "content": docs[1], "metadata": {}},
        {"doc_id": "d3", "content": docs[2], "metadata": {}},
        {"doc_id": "d4", "content": docs[3], "metadata": {}},
    ]
    vector_retriever.add_documents(doc_list, vectors.tolist())
    print(f"✓ 创建向量检索器: docs={vector_store.count()}")
    
    # 测试 2: 创建关键词检索器
    keyword_retriever = KeywordRetriever(top_k=3)
    keyword_retriever.add_documents(doc_list)
    print(f"✓ 创建关键词检索器: docs={len(keyword_retriever.documents)}")
    
    # 测试 3: 创建混合检索器
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        top_k=3,
        vector_weight=0.6,
        keyword_weight=0.4
    )
    print(f"✓ 创建混合检索器: weights=({hybrid_retriever.vector_weight}, {hybrid_retriever.keyword_weight})")
    
    # 测试 4: 混合检索
    results = hybrid_retriever.retrieve("人工智能技术", top_k=3)
    print(f"✓ 混合检索: 返回 {len(results)} 个结果")
    
    # 测试 5: 检索结果
    for i, result in enumerate(results[:2]):
        print(f"  - 结果 {i+1}: score={result.score:.4f}")
        print(f"    来源: {result.metadata.get('source_type', 'unknown')}")
    
    # 测试 6: 统计信息
    stats = hybrid_retriever.get_stats()
    print(f"✓ 统计信息: type={stats['type']}, rrf_k={stats['rrf_k']}")
    
    print("\n所有测试通过!")
