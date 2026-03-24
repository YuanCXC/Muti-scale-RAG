# -*- coding: utf-8 -*-
"""RAG Chain 模块

提供多种 RAG 实现策略：
- NaiveRAGChain: 简单的向量检索 + LLM 生成
- HybridRAGChain: 混合检索（向量+关键词）+ 重排序
- GraphRAGChain: 基于知识图谱的 RAG
"""

from src.chains.base_chain import ChainResult, RAGChainBase
from src.chains.naive_chain import NaiveRAGChain
from src.chains.hybrid_chain import HybridRAGChain
from src.chains.graph_chain import GraphRAGChain
from src.utils.config import get_config
from src.utils.logger import get_logger

__all__ = [
    "ChainResult",
    "RAGChainBase",
    "NaiveRAGChain",
    "HybridRAGChain",
    "GraphRAGChain",
    "create_chain",
]


def create_chain(
    chain_type: str = "naive",
    **kwargs,
) -> RAGChainBase:
    """创建 RAG Chain 实例
    
    Args:
        chain_type: Chain 类型 (naive, hybrid, graph)
        **kwargs: 传递给 Chain 构造函数的参数
        
    Returns:
        RAG Chain 实例
        
    Raises:
        ValueError: 不支持的 Chain 类型
    """
    logger = get_logger(__name__)
    config = get_config()
    
    chain_type = chain_type.lower()
    
    if chain_type == "naive":
        return NaiveRAGChain(**kwargs)
    
    elif chain_type == "hybrid":
        return HybridRAGChain(**kwargs)
    
    elif chain_type == "graph":
        use_neo4j = kwargs.pop("use_neo4j", False)
        return GraphRAGChain(use_neo4j=use_neo4j, **kwargs)
    
    else:
        raise ValueError(
            f"Unsupported chain type: {chain_type}. "
            f"Supported types: naive, hybrid, graph"
        )
