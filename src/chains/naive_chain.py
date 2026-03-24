# -*- coding: utf-8 -*-
"""Naive RAG Chain 实现

简单的向量检索 + LLM 生成流程。
"""

from typing import Any, List, Optional

from src.llms.base_client import LLMResponse, Message
from src.retrievers.base_retriever import RetrieverBase, SearchResult
from src.retrievers.vector_retriever import VectorRetriever
from src.storage.vector_store.faiss_store import FAISSVectorStore
from src.llms import create_client, create_embedding_client
from src.utils.config import get_config
from src.utils.logger import get_logger

from src.chains.base_chain import ChainResult, RAGChainBase

logger = get_logger(__name__)


class NaiveRAGChain(RAGChainBase):
    """Naive RAG Chain
    
    最简单的 RAG 实现：向量检索 + LLM 生成。
    
    Attributes:
        vector_store: 向量存储实例
        embedding_client: 嵌入客户端
    """
    
    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        retriever: Optional[VectorRetriever] = None,
        **kwargs: Any,
    ):
        """初始化 Naive RAG Chain
        
        Args:
            vector_store: 向量存储实例（可选）
            retriever: 向量检索器实例（可选）
            **kwargs: 额外参数
        """
        config = get_config()
        
        if retriever is not None:
            self.vector_store = retriever.vector_store
            self.embedding_client = retriever.llm_client
        elif vector_store is not None:
            self.vector_store = vector_store
            self.embedding_client = create_embedding_client()
            retriever = VectorRetriever(
                vector_store=vector_store,
                llm_client=self.embedding_client,
            )
        else:
            self.vector_store = FAISSVectorStore(
                dimension=config.vector_dim,
                metric="cosine",
            )
            self.embedding_client = create_embedding_client()
            retriever = VectorRetriever(
                vector_store=self.vector_store,
                llm_client=self.embedding_client,
            )
        
        super().__init__(retriever=retriever, **kwargs)
        
        logger.info(
            f"初始化 NaiveRAGChain: "
            f"vector_dim={self.vector_store.dimension}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """执行向量检索
        
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
        
        results = self.retriever.retrieve(query, top_k=actual_top_k, **kwargs)
        
        logger.info(
            f"NaiveRAG 检索完成: query='{query[:30]}...', "
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
        
        messages = self._build_prompt(query, context_str)
        
        response = self.llm_client.chat(messages)
        
        logger.info(
            f"NaiveRAG 生成完成: "
            f"tokens={response.usage.get('total_tokens', 'N/A')}"
        )
        
        return response
    
    def add_documents(
        self,
        documents: List[dict],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """添加文档到向量存储
        
        Args:
            documents: 文档列表
            embeddings: 预计算的嵌入向量（可选）
            
        Returns:
            添加的向量ID列表
        """
        if isinstance(self.retriever, VectorRetriever):
            return self.retriever.add_documents(documents, embeddings)
        else:
            raise ValueError("Retriever does not support add_documents")
    
    def get_stats(self) -> dict:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "chain_type": "naive",
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
    print("测试 Naive RAG Chain")
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
    
    chain = NaiveRAGChain(
        vector_store=vector_store,
        llm_client=mock_llm,
    )
    print(f"✓ 创建 NaiveRAGChain")
    
    docs = [
        {"doc_id": "d1", "content": "人工智能是计算机科学的一个分支。", "metadata": {"source": "wiki"}},
        {"doc_id": "d2", "content": "机器学习是人工智能的核心技术。", "metadata": {"source": "wiki"}},
        {"doc_id": "d3", "content": "深度学习使用多层神经网络。", "metadata": {"source": "wiki"}},
    ]
    chain.add_documents(docs)
    print(f"✓ 添加文档: count={vector_store.count()}")
    
    result = chain.run("什么是人工智能？")
    print(f"✓ 执行 RAG: answer='{result.answer[:50]}...'")
    print(f"  - 检索结果数: {len(result.retrieval_results)}")
    print(f"  - 来源数: {len(result.sources)}")
    
    stats = chain.get_stats()
    print(f"✓ 统计信息: {stats}")
    
    print("\n所有测试通过!")
