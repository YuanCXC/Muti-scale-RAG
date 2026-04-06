# -*- coding: utf-8 -*-
"""向量检索器实现

使用 FAISS 向量库进行语义检索。
"""

from typing import Any, Dict, List, Optional

import numpy as np

from src.llms.base_client import BaseLLMClient
from src.storage.vector_store.base_store import VectorMetadata
from src.storage.vector_store.faiss_store import FAISSVectorStore
from src.utils.logger import get_logger

from src.retrievers.base_retriever import RetrieverBase, SearchResult

logger = get_logger(__name__)


class VectorRetriever(RetrieverBase):
    """向量检索器
    
    使用 FAISS 向量库进行语义检索，支持查询向量化和相似度搜索。
    
    Attributes:
        vector_store: FAISS 向量存储实例
        llm_client: LLM 客户端（用于生成嵌入向量）
        embedding_model: 嵌入模型名称
    """
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        llm_client: BaseLLMClient,
        embedding_model: Optional[str] = None,
        **kwargs: Any,
    ):
        """初始化向量检索器
        
        Args:
            vector_store: FAISS 向量存储实例
            llm_client: LLM 客户端（需支持 embed 方法）
            embedding_model: 嵌入模型名称（可选）
            **kwargs: 额外参数
        """
        super().__init__(**kwargs)
        
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        
        logger.info(
            f"初始化向量检索器: top_k={self.top_k}, "
            f"score_threshold={self.score_threshold}"
        )
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """获取查询文本的嵌入向量
        
        Args:
            query: 查询文本
            
        Returns:
            嵌入向量
        """
        embeddings = self.llm_client.embed([query])
        
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Failed to generate embedding for query")
        
        return np.array(embeddings[0], dtype=np.float32)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """执行向量检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量（可选）
            filter_dict: 过滤条件（可选）
            **kwargs: 额外参数
            
        Returns:
            检索结果列表，按相似度分数降序排列
        """
        actual_top_k = self._get_top_k(top_k)
        
        # 获取查询向量
        query_vector = self._get_query_embedding(query)
        
        # 执行向量搜索
        vector_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=actual_top_k,
            filter_dict=filter_dict,
        )
        
        # 转换为 SearchResult 格式
        results = []
        for vr in vector_results:
            result = SearchResult(
                doc_id=vr.metadata.doc_id,
                content=vr.metadata.content,
                score=vr.score,
                metadata={
                    "chunk_id": vr.metadata.chunk_id,
                    "source": vr.metadata.source,
                    "page": vr.metadata.page,
                    "position": vr.metadata.position,
                    **vr.metadata.extra,
                },
            )
            results.append(result)
        
        # 根据阈值过滤
        results = self._filter_by_threshold(results)
        
        logger.info(
            f"向量检索完成: query='{query[:50]}...', "
            f"results={len(results)}"
        )
        
        return results
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """添加文档到向量存储
        
        Args:
            documents: 文档列表，每个文档包含 doc_id, content, metadata
            embeddings: 预计算的嵌入向量（可选）
            
        Returns:
            添加的向量ID列表
        """
        # 如果没有提供嵌入向量，则生成
        if embeddings is None:
            texts = [doc["content"] for doc in documents]
            embeddings = self.llm_client.embed(texts)
        
        # 准备向量数组
        vectors = np.array(embeddings, dtype=np.float32)
        
        # 准备元数据
        metadata_list = []
        ids = []
        
        for doc in documents:
            meta = doc.get("metadata", {})
            
            vector_meta = VectorMetadata(
                doc_id=doc["doc_id"],
                chunk_id=meta.get("chunk_id", doc["doc_id"]),
                content=doc["content"],
                source=meta.get("source", "unknown"),
                page=meta.get("page"),
                position=meta.get("position"),
                extra=meta.get("extra", {}),
            )
            metadata_list.append(vector_meta)
            ids.append(meta.get("vector_id"))
        
        # 添加到向量存储
        return self.vector_store.add_vectors(
            vectors=vectors,
            metadata=metadata_list,
            ids=ids if all(ids) else None,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "type": "vector",
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "vector_store": self.vector_store.get_stats(),
        }


if __name__ == "__main__":
    import numpy as np
    from src.storage.vector_store import FAISSVectorStore
    
    print("=" * 50)
    print("测试 Vector Retriever 模块")
    print("=" * 50)
    
    # Mock LLM Client for testing
    class MockLLMClient:
        def embed(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            return [np.random.randn(64).astype(np.float32).tolist() for _ in texts]
    
    mock_llm = MockLLMClient()
    
    # 测试 1: 创建向量存储
    vector_store = FAISSVectorStore(
        dimension=64,
        metric="cosine",
        index_type="flat"
    )
    print(f"✓ 创建向量存储: dimension={vector_store.dimension}")
    
    # 测试 2: 创建检索器
    retriever = VectorRetriever(
        vector_store=vector_store,
        llm_client=mock_llm,
        top_k=3,
        score_threshold=0.5
    )
    print(f"✓ 创建检索器: top_k={retriever.top_k}")
    
    # 测试 3: 添加文档
    docs = [
        {"doc_id": "doc_1", "content": "人工智能是计算机科学的一个分支。", "metadata": {}},
        {"doc_id": "doc_2", "content": "机器学习是人工智能的核心技术。", "metadata": {}},
        {"doc_id": "doc_3", "content": "深度学习使用多层神经网络进行学习。", "metadata": {}},
    ]
    vectors = np.random.randn(3, 64).astype(np.float32)
    retriever.add_documents(docs, vectors.tolist())
    print(f"✓ 添加文档: count={vector_store.count()}")
    
    # 测试 4: 检索
    results = retriever.retrieve("人工智能技术", top_k=2)
    print(f"✓ 检索: 返回 {len(results)} 个结果")
    
    # 测试 5: 检索结果
    if results:
        print(f"  - 第一个结果: score={results[0].score:.4f}")
        print(f"  - 内容: {results[0].content[:30]}...")
    
    # 测试 6: 统计信息
    stats = retriever.get_stats()
    print(f"✓ 统计信息: {stats['type']}, top_k={stats['top_k']}")
    
    print("\n所有测试通过!")
