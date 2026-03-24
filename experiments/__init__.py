# -*- coding: utf-8 -*-
"""实验模块

提供 RAG 系统的多种实验实现：
- exp_01_no_retrieval: 无检索基线
- exp_02_naive_rag: 纯向量检索 RAG
- exp_03_hybrid_rag: 混合检索 RAG
- exp_04_graph_rag: 图谱增强 RAG
- exp_05_live_rag: 动态路由 RAG
- exp_06_full_system: 完整系统
- exp_07_hotpotqa: HotpotQA 多跳问答
"""

from .experiment_base import (
    ExperimentConfig,
    load_knowledge_base,
    load_test_dataset,
    load_knowledge_graph,
    save_results,
    save_report,
    create_llm_client,
    create_embedding_client,
)

__all__ = [
    "ExperimentConfig",
    "load_knowledge_base",
    "load_test_dataset",
    "load_knowledge_graph",
    "save_results",
    "save_report",
    "create_llm_client",
    "create_embedding_client",
]
