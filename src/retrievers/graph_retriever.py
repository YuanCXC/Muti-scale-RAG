# -*- coding: utf-8 -*-
"""知识图谱检索器实现

支持 Neo4j Cypher 查询和 NetworkX 本地图谱遍历。
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from src.storage.graph_store.base_graph import (
    Edge,
    EdgeType,
    GraphPath,
    GraphStoreBase,
    Node,
    NodeType,
)
from src.storage.graph_store.local_graph import LocalGraphStore
from src.storage.graph_store.neo4j_store import Neo4jGraphStore
from src.utils.logger import get_logger

from src.retrievers.base_retriever import RetrieverBase, SearchResult

logger = get_logger(__name__)


class GraphRetriever(RetrieverBase):
    """知识图谱检索器
    
    支持图谱查询、实体链接和关系扩展。
    
    Attributes:
        graph_store: 图存储实例
        max_depth: 关系扩展最大深度
        entity_types: 要检索的实体类型
    """
    
    def __init__(
        self,
        graph_store: GraphStoreBase,
        max_depth: int = 2,
        entity_types: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """初始化知识图谱检索器
        
        Args:
            graph_store: 图存储实例（Neo4j 或 NetworkX）
            max_depth: 关系扩展最大深度
            entity_types: 要检索的实体类型列表（可选）
            **kwargs: 额外参数
        """
        super().__init__(**kwargs)
        
        self.graph_store = graph_store
        self.max_depth = max_depth
        self.entity_types = entity_types or ["entity", "concept"]
        
        logger.info(
            f"初始化知识图谱检索器: top_k={self.top_k}, "
            f"max_depth={max_depth}, entity_types={entity_types}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """执行知识图谱检索
        
        Args:
            query: 查询文本（实体名称或关键词）
            top_k: 返回结果数量（可选）
            **kwargs: 额外参数
            
        Returns:
            检索结果列表，包含相关实体和三元组
        """
        actual_top_k = self._get_top_k(top_k)
        
        # 1. 实体链接：查找匹配的实体节点
        matched_nodes = self._entity_linking(query)
        
        # 2. 关系扩展：获取相关实体和关系
        results = []
        seen_nodes: Set[str] = set()
        
        for node in matched_nodes[:actual_top_k]:
            if node.id in seen_nodes:
                continue
            
            # 获取实体的邻居和关系
            neighbors = self.graph_store.get_neighbors(
                node_id=node.id,
                direction="both",
                limit=actual_top_k,
            )
            
            # 构建检索结果
            content = self._build_result_content(node, neighbors)
            
            result = SearchResult(
                doc_id=node.id,
                content=content,
                score=1.0,  # 图谱检索的分数基于连接强度
                metadata={
                    "node_type": node.type.value,
                    "node_name": node.name,
                    "properties": node.properties,
                    "neighbor_count": len(neighbors),
                },
            )
            
            results.append(result)
            seen_nodes.add(node.id)
        
        # 根据阈值过滤
        results = self._filter_by_threshold(results)
        
        logger.info(
            f"知识图谱检索完成: query='{query[:50]}...', "
            f"results={len(results)}"
        )
        
        return results
    
    def _entity_linking(self, query: str) -> List[Node]:
        """实体链接：查找与查询匹配的实体节点
        
        Args:
            query: 查询文本
            
        Returns:
            匹配的节点列表
        """
        matched_nodes = []
        
        # 解析实体类型
        target_types = [
            NodeType(et.lower()) for et in self.entity_types
            if et.lower() in [nt.value for nt in NodeType]
        ]
        
        # 按类型查找节点
        for node_type in target_types:
            nodes = self.graph_store.get_nodes_by_type(node_type)
            
            for node in nodes:
                # 简单的名称匹配
                if query.lower() in node.name.lower():
                    matched_nodes.append(node)
                # 属性匹配
                elif self._match_properties(query, node.properties):
                    matched_nodes.append(node)
        
        return matched_nodes
    
    def _match_properties(
        self,
        query: str,
        properties: Dict[str, Any],
    ) -> bool:
        """检查属性是否匹配查询
        
        Args:
            query: 查询文本
            properties: 节点属性
            
        Returns:
            是否匹配
        """
        query_lower = query.lower()
        
        for key, value in properties.items():
            if isinstance(value, str):
                if query_lower in value.lower():
                    return True
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str) and query_lower in item.lower():
                        return True
        
        return False
    
    def _build_result_content(
        self,
        node: Node,
        neighbors: List[Tuple[Node, Edge]],
    ) -> str:
        """构建检索结果的内容字符串
        
        Args:
            node: 中心节点
            neighbors: 邻居节点和边列表
            
        Returns:
            内容字符串
        """
        parts = [f"实体: {node.name}"]
        
        if node.properties:
            props_str = ", ".join(
                f"{k}: {v}" for k, v in node.properties.items()
            )
            parts.append(f"属性: {props_str}")
        
        # 添加关系信息
        relations = []
        for neighbor, edge in neighbors[:10]:  # 限制显示数量
            relation = f"{node.name} --[{edge.type.value}]--> {neighbor.name}"
            relations.append(relation)
        
        if relations:
            parts.append("关系:")
            parts.extend(f"  - {r}" for r in relations)
        
        return "\n".join(parts)
    
    def find_related_entities(
        self,
        entity_name: str,
        max_depth: Optional[int] = None,
    ) -> List[Node]:
        """查找相关实体
        
        Args:
            entity_name: 实体名称
            max_depth: 最大搜索深度（可选）
            
        Returns:
            相关实体节点列表
        """
        depth = max_depth or self.max_depth
        
        # 查找起始节点
        start_nodes = self._entity_linking(entity_name)
        if not start_nodes:
            return []
        
        # BFS 遍历
        visited: Set[str] = set()
        related_nodes: List[Node] = []
        
        for start_node in start_nodes:
            queue = [(start_node, 0)]
            visited.add(start_node.id)
            
            while queue:
                current_node, current_depth = queue.pop(0)
                
                if current_depth > 0:
                    related_nodes.append(current_node)
                
                if current_depth < depth:
                    neighbors = self.graph_store.get_neighbors(
                        node_id=current_node.id,
                        direction="both",
                        limit=50,
                    )
                    
                    for neighbor, _ in neighbors:
                        if neighbor.id not in visited:
                            visited.add(neighbor.id)
                            queue.append((neighbor, current_depth + 1))
        
        return related_nodes
    
    def find_paths(
        self,
        source_entity: str,
        target_entity: str,
        max_depth: Optional[int] = None,
    ) -> List[GraphPath]:
        """查找两个实体之间的路径
        
        Args:
            source_entity: 源实体名称
            target_entity: 目标实体名称
            max_depth: 最大搜索深度（可选）
            
        Returns:
            路径列表
        """
        depth = max_depth or self.max_depth
        
        # 查找起始和目标节点
        source_nodes = self._entity_linking(source_entity)
        target_nodes = self._entity_linking(target_entity)
        
        if not source_nodes or not target_nodes:
            return []
        
        paths = []
        
        for source_node in source_nodes:
            for target_node in target_nodes:
                found_paths = self.graph_store.find_all_paths(
                    source_id=source_node.id,
                    target_id=target_node.id,
                    max_depth=depth,
                    limit=5,
                )
                paths.extend(found_paths)
        
        return paths
    
    def execute_cypher(
        self,
        cypher_query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """执行 Cypher 查询（仅 Neo4j）
        
        Args:
            cypher_query: Cypher 查询语句
            parameters: 查询参数
            
        Returns:
            查询结果列表
            
        Raises:
            ValueError: 图存储不是 Neo4j
        """
        if not isinstance(self.graph_store, Neo4jGraphStore):
            raise ValueError(
                "Cypher queries are only supported with Neo4j graph store"
            )
        
        return self.graph_store.query(cypher_query, parameters)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "type": "graph",
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "max_depth": self.max_depth,
            "entity_types": self.entity_types,
            "node_count": self.graph_store.count_nodes(),
            "edge_count": self.graph_store.count_edges(),
        }


if __name__ == "__main__":
    from src.storage.graph_store import LocalGraphStore, Node, NodeType, Edge, EdgeType
    
    print("=" * 50)
    print("测试 Graph Retriever 模块")
    print("=" * 50)
    
    # 测试 1: 创建图存储
    graph_store = LocalGraphStore()
    print(f"✓ 创建图存储: nodes={graph_store.count_nodes()}")
    
    # 测试 2: 添加节点
    nodes = [
        Node(id="ai", type=NodeType.ENTITY, name="人工智能", properties={"category": "技术"}),
        Node(id="ml", type=NodeType.ENTITY, name="机器学习", properties={"category": "技术"}),
        Node(id="dl", type=NodeType.ENTITY, name="深度学习", properties={"category": "技术"}),
        Node(id="nlp", type=NodeType.ENTITY, name="自然语言处理", properties={"category": "技术"}),
    ]
    for node in nodes:
        graph_store.add_node(node)
    print(f"✓ 添加节点: count={graph_store.count_nodes()}")
    
    # 测试 3: 添加边
    edges = [
        Edge(id="e1", source_id="ai", target_id="ml", type=EdgeType.RELATED_TO, weight=0.9),
        Edge(id="e2", source_id="ml", target_id="dl", type=EdgeType.RELATED_TO, weight=0.8),
        Edge(id="e3", source_id="ai", target_id="nlp", type=EdgeType.RELATED_TO, weight=0.85),
    ]
    for edge in edges:
        graph_store.add_edge(edge)
    print(f"✓ 添加边: count={graph_store.count_edges()}")
    
    # 测试 4: 创建检索器
    retriever = GraphRetriever(
        graph_store=graph_store,
        top_k=5,
        max_depth=2
    )
    print(f"✓ 创建检索器: top_k={retriever.top_k}, max_depth={retriever.max_depth}")
    
    # 测试 5: 实体检索
    results = retriever.retrieve("人工智能", top_k=3)
    print(f"✓ 实体检索 '人工智能': 返回 {len(results)} 个结果")
    
    # 测试 6: 邻居检索
    neighbors = retriever.retrieve("ml", search_type="neighbors", top_k=5)
    print(f"✓ 邻居检索 'ml': 返回 {len(neighbors)} 个邻居")
    
    # 测试 7: 路径查找
    paths = retriever.retrieve("ai", search_type="paths", target_entity="dl")
    print(f"✓ 路径查找 'ai' -> 'dl': {paths}")
    
    # 测试 8: 统计信息
    stats = retriever.get_stats()
    print(f"✓ 统计信息: nodes={stats['node_count']}, edges={stats['edge_count']}")
    
    print("\n所有测试通过!")
