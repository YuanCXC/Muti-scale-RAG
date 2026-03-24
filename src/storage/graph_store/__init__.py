# -*- coding: utf-8 -*-
"""图存储模块

提供图存储的抽象接口和具体实现。
"""

from .base_graph import (
    Node,
    NodeType,
    Edge,
    EdgeType,
    GraphPath,
    GraphStoreBase,
)
from .neo4j_store import Neo4jGraphStore
from .local_graph import LocalGraphStore

__all__ = [
    # 基础类
    "GraphStoreBase",
    "Node",
    "NodeType",
    "Edge",
    "EdgeType",
    "GraphPath",
    # Neo4j 实现
    "Neo4jGraphStore",
    # 本地图实现
    "LocalGraphStore",
]
