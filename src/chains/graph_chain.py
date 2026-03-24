# -*- coding: utf-8 -*-
"""Graph RAG Chain 实现

基于知识图谱的 RAG 流程。
"""

from typing import Any, Dict, List, Optional

from src.llms.base_client import LLMResponse, Message
from src.retrievers.base_retriever import RetrieverBase, SearchResult
from src.retrievers.graph_retriever import GraphRetriever
from src.storage.graph_store.base_graph import GraphStoreBase, Node, NodeType
from src.storage.graph_store.local_graph import LocalGraphStore
from src.storage.graph_store.neo4j_store import Neo4jGraphStore
from src.llms import create_client, create_embedding_client
from src.utils.config import get_config
from src.utils.logger import get_logger

from src.chains.base_chain import ChainResult, RAGChainBase

logger = get_logger(__name__)


class GraphRAGChain(RAGChainBase):
    """Graph RAG Chain
    
    基于知识图谱的 RAG 实现，支持实体链接和关系推理。
    
    Attributes:
        graph_store: 图存储实例
        max_depth: 关系扩展最大深度
        entity_types: 要检索的实体类型
    """
    
    def __init__(
        self,
        graph_store: Optional[GraphStoreBase] = None,
        retriever: Optional[GraphRetriever] = None,
        max_depth: int = 2,
        entity_types: Optional[List[str]] = None,
        use_neo4j: bool = False,
        **kwargs: Any,
    ):
        """初始化 Graph RAG Chain
        
        Args:
            graph_store: 图存储实例（可选）
            retriever: 图检索器实例（可选）
            max_depth: 关系扩展最大深度
            entity_types: 要检索的实体类型
            use_neo4j: 是否使用 Neo4j（默认使用本地图存储）
            **kwargs: 额外参数
        """
        config = get_config()
        
        self.max_depth = max_depth
        self.entity_types = entity_types or ["entity", "concept"]
        
        if retriever is not None:
            self.graph_store = retriever.graph_store
        elif graph_store is not None:
            self.graph_store = graph_store
            retriever = GraphRetriever(
                graph_store=graph_store,
                max_depth=max_depth,
                entity_types=self.entity_types,
            )
        else:
            if use_neo4j:
                self.graph_store = Neo4jGraphStore(
                    uri=config.neo4j_uri,
                    user=config.neo4j_user,
                    password=config.neo4j_password,
                    database=config.neo4j_database,
                )
            else:
                self.graph_store = LocalGraphStore()
            
            retriever = GraphRetriever(
                graph_store=self.graph_store,
                max_depth=max_depth,
                entity_types=self.entity_types,
            )
        
        super().__init__(retriever=retriever, **kwargs)
        
        logger.info(
            f"初始化 GraphRAGChain: "
            f"max_depth={max_depth}, entity_types={self.entity_types}, "
            f"use_neo4j={use_neo4j}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """执行知识图谱检索
        
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
            f"GraphRAG 检索完成: query='{query[:30]}...', "
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
            context: 检索上下文（包含实体和关系信息）
            **kwargs: 额外参数
            
        Returns:
            LLM 响应
        """
        context_str = self._build_graph_context(context)
        
        system_prompt = """你是一个专业的知识图谱问答助手。请根据提供的实体和关系信息回答用户问题。
要求：
1. 利用实体之间的关系进行推理
2. 回答要准确、有条理
3. 如果需要多跳推理，请展示推理过程
4. 如果图谱中没有相关信息，请明确说明"""
        
        messages = self._build_prompt(query, context_str, system_prompt)
        
        response = self.llm_client.chat(messages)
        
        logger.info(
            f"GraphRAG 生成完成: "
            f"tokens={response.usage.get('total_tokens', 'N/A')}"
        )
        
        return response
    
    def _build_graph_context(
        self,
        results: List[SearchResult],
    ) -> str:
        """构建图谱上下文
        
        Args:
            results: 检索结果列表
            
        Returns:
            图谱上下文字符串
        """
        context_parts = ["【知识图谱检索结果】\n"]
        
        for i, result in enumerate(results, start=1):
            context_parts.append(f"\n--- 实体 {i} ---")
            context_parts.append(result.content)
            
            if result.metadata.get("neighbor_count"):
                context_parts.append(f"关联实体数: {result.metadata['neighbor_count']}")
        
        return "\n".join(context_parts)
    
    def find_related_entities(
        self,
        entity_name: str,
        max_depth: Optional[int] = None,
    ) -> List[Node]:
        """查找相关实体
        
        Args:
            entity_name: 实体名称
            max_depth: 最大搜索深度
            
        Returns:
            相关实体列表
        """
        if isinstance(self.retriever, GraphRetriever):
            return self.retriever.find_related_entities(
                entity_name, max_depth or self.max_depth
            )
        return []
    
    def find_paths(
        self,
        source_entity: str,
        target_entity: str,
        max_depth: Optional[int] = None,
    ) -> List:
        """查找两个实体之间的路径
        
        Args:
            source_entity: 源实体名称
            target_entity: 目标实体名称
            max_depth: 最大搜索深度
            
        Returns:
            路径列表
        """
        if isinstance(self.retriever, GraphRetriever):
            return self.retriever.find_paths(
                source_entity, target_entity, max_depth or self.max_depth
            )
        return []
    
    def add_entities(
        self,
        entities: List[Dict[str, Any]],
        relations: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """添加实体和关系到图谱
        
        Args:
            entities: 实体列表
            relations: 关系列表（可选）
            
        Returns:
            添加的实体数量
        """
        from src.storage.graph_store.base_graph import Node, Edge, NodeType, EdgeType
        
        count = 0
        for entity in entities:
            node_type = NodeType(entity.get("type", "entity").lower())
            node = Node(
                id=entity["id"],
                type=node_type,
                name=entity["name"],
                properties=entity.get("properties", {}),
            )
            self.graph_store.add_node(node)
            count += 1
        
        if relations:
            for relation in relations:
                edge_type = EdgeType(relation.get("type", "related_to").upper())
                edge = Edge(
                    id=relation.get("id"),
                    source_id=relation["source_id"],
                    target_id=relation["target_id"],
                    type=edge_type,
                    weight=relation.get("weight", 1.0),
                    properties=relation.get("properties", {}),
                )
                self.graph_store.add_edge(edge)
        
        logger.info(f"添加实体: count={count}, relations={len(relations) if relations else 0}")
        return count
    
    def get_stats(self) -> dict:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "chain_type": "graph",
            "max_depth": self.max_depth,
            "entity_types": self.entity_types,
            "graph_store": {
                "type": self.graph_store.__class__.__name__,
                "node_count": self.graph_store.count_nodes(),
                "edge_count": self.graph_store.count_edges(),
            },
            "llm_model": self.config.llm_model,
        }


if __name__ == "__main__":
    from src.storage.graph_store.base_graph import Node, Edge, NodeType, EdgeType
    
    print("=" * 50)
    print("测试 Graph RAG Chain")
    print("=" * 50)
    
    class MockLLMClient:
        def __init__(self):
            self.calls = 0
        
        def chat(self, messages):
            self.calls += 1
            from src.llms.base_client import LLMResponse
            return LLMResponse(
                content="这是一个模拟的回答。",
                usage={"total_tokens": 100},
            )
    
    mock_llm = MockLLMClient()
    
    chain = GraphRAGChain(llm_client=mock_llm)
    print(f"✓ 创建 GraphRAGChain")
    
    entities = [
        {"id": "ai", "name": "人工智能", "type": "concept", "properties": {"category": "技术"}},
        {"id": "ml", "name": "机器学习", "type": "concept", "properties": {"category": "技术"}},
        {"id": "dl", "name": "深度学习", "type": "concept", "properties": {"category": "技术"}},
        {"id": "nlp", "name": "自然语言处理", "type": "concept", "properties": {"category": "应用"}},
    ]
    relations = [
        {"source_id": "ai", "target_id": "ml", "type": "includes", "weight": 0.9},
        {"source_id": "ml", "target_id": "dl", "type": "includes", "weight": 0.8},
        {"source_id": "ai", "target_id": "nlp", "type": "applies_to", "weight": 0.85},
    ]
    chain.add_entities(entities, relations)
    print(f"✓ 添加实体和关系: nodes={chain.graph_store.count_nodes()}, edges={chain.graph_store.count_edges()}")
    
    result = chain.run("人工智能")
    print(f"✓ 执行 RAG: answer='{result.answer[:50]}...'")
    print(f"  - 检索结果数: {len(result.retrieval_results)}")
    
    paths = chain.find_paths("ai", "dl")
    print(f"✓ 路径查找: ai -> dl, paths={len(paths)}")
    
    stats = chain.get_stats()
    print(f"✓ 统计信息: {stats}")
    
    print("\n所有测试通过!")
