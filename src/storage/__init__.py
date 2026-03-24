# -*- coding: utf-8 -*-
"""存储模块

提供向量存储、图存储和文本切分功能。
"""

from .vector_store import (
    VectorStoreBase,
    VectorMetadata,
    SearchResult,
    FAISSVectorStore,
    IndexType,
    StorageMode,
)

from .graph_store import (
    GraphStoreBase,
    Node,
    NodeType,
    Edge,
    EdgeType,
    GraphPath,
    Neo4jGraphStore,
    LocalGraphStore,
)

from .chunking_mapping import (
    TextChunker,
    Chunk,
    ChunkStrategy,
    ChunkMapper,
    ChunkMapping,
)

__all__ = [
    # 向量存储
    "VectorStoreBase",
    "VectorMetadata",
    "SearchResult",
    "FAISSVectorStore",
    "IndexType",
    "StorageMode",
    # 图存储
    "GraphStoreBase",
    "Node",
    "NodeType",
    "Edge",
    "EdgeType",
    "GraphPath",
    "Neo4jGraphStore",
    "LocalGraphStore",
    # 切分映射
    "TextChunker",
    "Chunk",
    "ChunkStrategy",
    "ChunkMapper",
    "ChunkMapping",
]
