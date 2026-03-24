# -*- coding: utf-8 -*-
"""Hybrid RAG Chain 实现

结合向量检索和关键词检索的混合 RAG 流程。
"""

from typing import Any, List, Optional

from src.llms.base_client import LLMResponse, Message
from src.retrievers.base_retriever import RetrieverBase, SearchResult
from src.retrievers.vector_retriever import VectorRetriever
from src.retrievers.keyword_retriever import KeywordRetriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.reranker import Reranker
from src.storage.vector_store.faiss_store import FAISSVectorStore
from src.llms import create_client, create_embedding_client
from src.utils.config import get_config
from src.utils.logger import get_logger

from src.chains.base_chain import ChainResult, RAGChainBase

logger = get_logger(__name__)


class HybridRAGChain(RAGChainBase):
    """Hybrid RAG Chain
    
    结合向量检索和关键词检索，使用 RRF 融合算法。
    支持可选的重排序步骤。
    
    Attributes:
        vector_store: 向量存储实例
        keyword_retriever: 关键词检索器
        reranker: 重排序器（可选）
        vector_weight: 向量检索权重
        keyword_weight: 关键词检索权重
    """
    
    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        keyword_retriever: Optional[KeywordRetriever] = None,
        reranker: Optional[Reranker] = None,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        use_reranker: bool = True,
        **kwargs: Any,
    ):
        """初始化 Hybrid RAG Chain
        
        Args:
            vector_store: 向量存储实例（可选）
            vector_retriever: 向量检索器实例（可选）
            keyword_retriever: 关键词检索器实例（可选）
            reranker: 重排序器实例（可选）
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            use_reranker: 是否使用重排序
            **kwargs: 额外参数
        """
        config = get_config()
        
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.use_reranker = use_reranker
        
        if vector_retriever is not None:
            self.vector_store = vector_retriever.vector_store
            embedding_client = vector_retriever.llm_client
        elif vector_store is not None:
            self.vector_store = vector_store
            embedding_client = create_embedding_client()
            vector_retriever = VectorRetriever(
                vector_store=vector_store,
                llm_client=embedding_client,
            )
        else:
            self.vector_store = FAISSVectorStore(
                dimension=config.vector_dim,
                metric="cosine",
            )
            embedding_client = create_embedding_client()
            vector_retriever = VectorRetriever(
                vector_store=self.vector_store,
                llm_client=embedding_client,
            )
        
        if keyword_retriever is None:
            keyword_retriever = KeywordRetriever(top_k=config.top_k)
        
        self.keyword_retriever = keyword_retriever
        
        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )
        
        if use_reranker and reranker is None:
            try:
                reranker = Reranker()
            except Exception as e:
                logger.warning(f"无法初始化重排序器: {e}")
                self.use_reranker = False
                reranker = None
        
        self.reranker = reranker
        
        super().__init__(retriever=hybrid_retriever, **kwargs)
        
        logger.info(
            f"初始化 HybridRAGChain: "
            f"vector_weight={vector_weight}, keyword_weight={keyword_weight}, "
            f"use_reranker={self.use_reranker}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """执行混合检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 额外参数
            
        Returns:
            检索结果列表
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized")
        
        actual_top_k = top_k or self.config.top_k
        
        retrieval_top_k = actual_top_k * 2 if self.use_reranker else actual_top_k
        
        results = self.retriever.retrieve(query, top_k=retrieval_top_k, **kwargs)
        
        if self.use_reranker and self.reranker is not None:
            results = self.reranker.rerank(
                query=query,
                results=results,
                top_k=actual_top_k,
            )
            logger.info(f"重排序完成: results={len(results)}")
        
        logger.info(
            f"HybridRAG 检索完成: query='{query[:30]}...', "
            f"results={len(results)}"
        )
        
        return results
    
    def generate(
        self,
        query: str,
        context: List[SearchResult],
        **kwargs: Any,
    ) -> LLMResponse:
        """生成答案
        
        Args:
            query: 查询文本
            context: 检索上下文
            **kwargs: 额外参数
            
        Returns:
            LLM 响应
        """
        context_str = self._build_context(context)
        
        system_prompt = """你是一个专业的问答助手。请根据提供的参考文档回答用户问题。
这些文档是通过混合检索（语义检索+关键词检索）获得的，可能包含不同角度的相关信息。
要求：
1. 综合所有相关信息，给出准确、全面的回答
2. 如果信息有冲突，请指出并给出最合理的解释
3. 引用信息时请注明来源文档编号
4. 如果参考文档中没有相关信息，请明确说明"""
        
        messages = self._build_prompt(query, context_str, system_prompt)
        
        response = self.llm_client.chat(messages)
        
        logger.info(
            f"HybridRAG 生成完成: "
            f"tokens={response.usage.get('total_tokens', 'N/A')}"
        )
        
        return response
    
    def add_documents(
        self,
        documents: List[dict],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """添加文档到向量存储和关键词检索器
        
        Args:
            documents: 文档列表
            embeddings: 预计算的嵌入向量（可选）
            
        Returns:
            添加的向量ID列表
        """
        hybrid_retriever = self.retriever
        if isinstance(hybrid_retriever, HybridRetriever):
            vector_ids = hybrid_retriever.vector_retriever.add_documents(
                documents, embeddings
            )
            hybrid_retriever.keyword_retriever.add_documents(documents)
            return vector_ids
        else:
            raise ValueError("Retriever is not a HybridRetriever")
    
    def get_stats(self) -> dict:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "chain_type": "hybrid",
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "use_reranker": self.use_reranker,
            "vector_store": {
                "dimension": self.vector_store.dimension,
                "count": self.vector_store.count(),
            },
            "llm_model": self.config.llm_model,
            "embedding_model": self.config.embedding_model,
        }


if __name__ == "__main__":
    import numpy as np
    
    print("=" * 50)
    print("测试 Hybrid RAG Chain")
    print("=" * 50)
    
    class MockLLMClient:
        def __init__(self):
            self.calls = 0
        
        def embed(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            return [np.random.randn(64).astype(np.float32).tolist() for _ in texts]
        
        def chat(self, messages):
            self.calls += 1
            from src.llms.base_client import LLMResponse
            return LLMResponse(
                content="这是一个模拟的回答。",
                usage={"total_tokens": 100},
            )
    
    mock_llm = MockLLMClient()
    
    vector_store = FAISSVectorStore(dimension=64, metric="cosine")
    
    chain = HybridRAGChain(
        vector_store=vector_store,
        llm_client=mock_llm,
        use_reranker=False,
    )
    print(f"✓ 创建 HybridRAGChain")
    
    docs = [
        {"doc_id": "d1", "content": "人工智能是计算机科学的一个分支。", "metadata": {"source": "wiki"}},
        {"doc_id": "d2", "content": "机器学习是人工智能的核心技术。", "metadata": {"source": "wiki"}},
        {"doc_id": "d3", "content": "深度学习使用多层神经网络。", "metadata": {"source": "wiki"}},
        {"doc_id": "d4", "content": "自然语言处理是人工智能的重要应用。", "metadata": {"source": "wiki"}},
    ]
    chain.add_documents(docs)
    print(f"✓ 添加文档: vector_count={vector_store.count()}, keyword_count={len(chain.keyword_retriever.documents)}")
    
    result = chain.run("人工智能有哪些技术？")
    print(f"✓ 执行 RAG: answer='{result.answer[:50]}...'")
    print(f"  - 检索结果数: {len(result.retrieval_results)}")
    print(f"  - 来源数: {len(result.sources)}")
    
    stats = chain.get_stats()
    print(f"✓ 统计信息: {stats}")
    
    print("\n所有测试通过!")
