# -*- coding: utf-8 -*-
"""基础向量存储抽象类

定义向量存储的统一接口，支持不同的向量数据库实现。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class VectorMetadata:
    """向量元数据
    
    存储与向量关联的元数据信息。
    
    Attributes:
        doc_id: 文档ID
        chunk_id: 切片ID
        content: 原始文本内容
        source: 文档来源
        page: 页码（可选）
        position: 位置信息（可选）
        extra: 额外元数据
    """
    doc_id: str
    chunk_id: str
    content: str
    source: str
    page: Optional[int] = None
    position: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
        }
        if self.page is not None:
            result["page"] = self.page
        if self.position is not None:
            result["position"] = self.position
        if self.extra:
            result["extra"] = self.extra
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorMetadata":
        """从字典创建实例"""
        return cls(
            doc_id=data["doc_id"],
            chunk_id=data["chunk_id"],
            content=data["content"],
            source=data["source"],
            page=data.get("page"),
            position=data.get("position"),
            extra=data.get("extra", {}),
        )


@dataclass
class SearchResult:
    """向量搜索结果
    
    Attributes:
        id: 向量ID
        score: 相似度分数
        metadata: 元数据
        vector: 向量（可选）
    """
    id: str
    score: float
    metadata: VectorMetadata
    vector: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "id": self.id,
            "score": self.score,
            "metadata": self.metadata.to_dict(),
        }
        if self.vector is not None:
            result["vector"] = self.vector.tolist()
        return result


class VectorStoreBase(ABC):
    """向量存储抽象基类
    
    定义向量存储的统一接口，所有向量数据库实现都应继承此类。
    
    Attributes:
        dimension: 向量维度
        metric: 相似度度量方式 (cosine, l2, ip)
    """
    
    def __init__(
        self,
        dimension: int = 1536,
        metric: str = "cosine",
        **kwargs: Any,
    ):
        """初始化向量存储
        
        Args:
            dimension: 向量维度
            metric: 相似度度量方式 (cosine, l2, ip)
            **kwargs: 额外参数
        """
        self.dimension = dimension
        self.metric = metric
        self._validate_metric()
    
    def _validate_metric(self) -> None:
        """验证相似度度量方式"""
        valid_metrics = ["cosine", "l2", "ip"]
        if self.metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric: {self.metric}. Must be one of {valid_metrics}"
            )
    
    @abstractmethod
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[VectorMetadata],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """添加向量到存储
        
        Args:
            vectors: 向量数组，shape=(n, dimension)
            metadata: 元数据列表
            ids: 向量ID列表（可选，不提供则自动生成）
            
        Returns:
            添加的向量ID列表
            
        Raises:
            ValueError: 向量维度不匹配或数量不一致
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """搜索相似向量
        
        Args:
            query_vector: 查询向量，shape=(dimension,)
            top_k: 返回结果数量
            filter_dict: 过滤条件（可选）
            
        Returns:
            搜索结果列表，按相似度降序排列
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """删除向量
        
        Args:
            ids: 要删除的向量ID列表
            
        Returns:
            是否删除成功
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """保存索引到文件
        
        Args:
            path: 保存路径
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """从文件加载索引
        
        Args:
            path: 加载路径
        """
        pass
    
    @abstractmethod
    def get_vector(self, id: str) -> Optional[Tuple[np.ndarray, VectorMetadata]]:
        """根据ID获取向量和元数据
        
        Args:
            id: 向量ID
            
        Returns:
            向量和元数据元组，不存在则返回 None
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """获取向量总数
        
        Returns:
            向量总数
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空所有向量"""
        pass
    
    def _validate_vectors(self, vectors: np.ndarray) -> None:
        """验证向量格式
        
        Args:
            vectors: 向量数组
            
        Raises:
            ValueError: 向量格式不正确
        """
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D array, got {vectors.ndim}D")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, "
                f"got {vectors.shape[1]}"
            )
    
    def _validate_metadata(
        self,
        vectors: np.ndarray,
        metadata: List[VectorMetadata],
    ) -> None:
        """验证元数据
        
        Args:
            vectors: 向量数组
            metadata: 元数据列表
            
        Raises:
            ValueError: 元数据数量不匹配
        """
        if len(vectors) != len(metadata):
            raise ValueError(
                f"Number of vectors ({len(vectors)}) and metadata ({len(metadata)}) "
                "must be equal"
            )
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """归一化向量（用于 cosine 相似度）
        
        Args:
            vector: 输入向量
            
        Returns:
            归一化后的向量
        """
        if self.metric == "cosine":
            norm = np.linalg.norm(vector)
            if norm > 0:
                return vector / norm
        return vector
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"dimension={self.dimension}, "
            f"metric={self.metric}, "
            f"count={self.count()})"
        )


if __name__ == "__main__":
    print("=" * 50)
    print("测试 Vector Store Base 模块")
    print("=" * 50)
    
    # 测试 1: VectorMetadata 类
    metadata = VectorMetadata(
        doc_id="doc_001",
        chunk_id="chunk_001",
        content="这是测试内容",
        source="test.txt"
    )
    print(f"✓ VectorMetadata 创建: doc_id={metadata.doc_id}")
    
    # 测试 2: SearchResult 类
    result = SearchResult(
        id="vec_001",
        score=0.95,
        metadata=metadata
    )
    print(f"✓ SearchResult 创建: score={result.score}")
    
    # 测试 3: 抽象类不能直接实例化
    try:
        store = VectorStoreBase(dimension=128)
    except TypeError as e:
        print(f"✓ 抽象类无法实例化: {type(e).__name__}")
    
    # 测试 4: 创建具体实现类
    class MockVectorStore(VectorStoreBase):
        def __init__(self, dimension: int):
            super().__init__(dimension, "cosine")
            self._vectors = {}
        
        def add_vectors(self, vectors, metadata, ids=None):
            for i, (vec, meta) in enumerate(zip(vectors, metadata)):
                vid = ids[i] if ids else f"vec_{i}"
                self._vectors[vid] = (vec, meta)
        
        def search(self, query, top_k=10):
            return []
        
        def delete(self, ids):
            pass
        
        def save(self, path):
            pass
        
        def load(self, path):
            pass
        
        def get_vector(self, id):
            return self._vectors.get(id)
        
        def count(self):
            return len(self._vectors)
        
        def clear(self):
            self._vectors.clear()
    
    store = MockVectorStore(dimension=128)
    print(f"✓ 具体实现类创建: dimension={store.dimension}, metric={store.metric}")
    
    # 测试 5: 向量验证
    import numpy as np
    vec = np.random.randn(1, 128).astype(np.float32)
    store._validate_vectors(vec)
    print(f"✓ 向量验证通过: shape={vec.shape}")
    
    # 测试 6: 归一化
    normalized = store._normalize_vector(vec[0])
    norm = np.linalg.norm(normalized)
    print(f"✓ 向量归一化: norm={norm:.4f}")
    
    print("\n所有测试通过!")
