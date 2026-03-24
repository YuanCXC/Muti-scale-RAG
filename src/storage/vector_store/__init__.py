# -*- coding: utf-8 -*-
"""向量存储模块

提供向量存储的抽象接口和具体实现。
"""

from .base_store import (
    VectorMetadata,
    SearchResult,
    VectorStoreBase,
)
from .faiss_store import (
    FAISSVectorStore,
    IndexType,
    StorageMode,
)

__all__ = [
    # 基础类
    "VectorStoreBase",
    "VectorMetadata",
    "SearchResult",
    # FAISS 实现
    "FAISSVectorStore",
    "IndexType",
    "StorageMode",
]
