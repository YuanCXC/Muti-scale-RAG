# -*- coding: utf-8 -*-
"""本地图存储实现

使用 NetworkX 实现本地图存储，支持实体链接和关系遍历。
适用于 real_kg 格式的知识图谱。
"""

import json
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
except ImportError:
    raise ImportError(
        "LocalGraphStore 需要依赖 networkx。请使用以下命令进行安装：pip install networkx"
    )

from src.storage.graph_store.base_graph import (
    Edge,
    EdgeType,
    GraphPath,
    GraphStoreBase,
    Node,
    NodeType,
)
from src.utils.config import get_config


class LocalGraphStore(GraphStoreBase):
    """本地图存储实现
    
    使用 NetworkX 实现图存储，支持图谱序列化和反序列化。
    适用于 real_kg 格式的知识图谱。
    
    Attributes:
        graph: NetworkX 图对象
    """
    
    def __init__(self, **kwargs: Any):
        """初始化本地图存储
        
        Args:
            **kwargs: 额外参数
        """
        super().__init__(**kwargs)
        
        config = get_config()
        
        self.graph_path = config.local_graph_path
        
        # 使用 MultiDiGraph 支持多重有向图
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        
        # 节点和边的索引
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}
    
    # ==================== 节点操作 ====================
    
    def add_node(self, node: Node) -> bool:
        """添加节点"""
        if node.id in self._nodes:
            # 更新现有节点
            self._nodes[node.id] = node
            self._graph.add_node(
                node.id,
                type=node.type.value,
                name=node.name,
                properties=node.properties,
                embedding=node.embedding,
            )
        else:
            # 添加新节点
            self._nodes[node.id] = node
            self._graph.add_node(
                node.id,
                type=node.type.value,
                name=node.name,
                properties=node.properties,
                embedding=node.embedding,
            )
        return True
    
    def add_nodes(self, nodes: List[Node]) -> int:
        """批量添加节点"""
        count = 0
        for node in nodes:
            if self.add_node(node):
                count += 1
        return count
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """获取节点"""
        return self._nodes.get(node_id)
    
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """更新节点属性"""
        if node_id not in self._nodes:
            return False
        
        node = self._nodes[node_id]
        node.properties.update(properties)
        
        # 更新图中的节点属性
        self._graph.nodes[node_id]["properties"] = node.properties
        return True
    
    def delete_node(self, node_id: str) -> bool:
        """删除节点"""
        if node_id not in self._nodes:
            return False
        
        # 删除相关的边
        edges_to_delete = [
            edge_id for edge_id, edge in self._edges.items()
            if edge.source_id == node_id or edge.target_id == node_id
        ]
        for edge_id in edges_to_delete:
            del self._edges[edge_id]
        
        # 删除节点
        del self._nodes[node_id]
        self._graph.remove_node(node_id)
        return True
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """根据类型获取节点"""
        return [
            node for node in self._nodes.values()
            if node.type == node_type
        ]
    
    def get_nodes_by_property(self, key: str, value: Any) -> List[Node]:
        """根据属性获取节点"""
        return [
            node for node in self._nodes.values()
            if node.properties.get(key) == value
        ]
    
    # ==================== 边操作 ====================
    
    def add_edge(self, edge: Edge) -> bool:
        """添加边"""
        # 检查源节点和目标节点是否存在
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            return False
        
        if edge.id in self._edges:
            # 更新现有边
            self._edges[edge.id] = edge
            # 删除旧边，添加新边
            self._graph.remove_edge(edge.source_id, edge.target_id, key=edge.id)
            self._graph.add_edge(
                edge.source_id,
                edge.target_id,
                key=edge.id,
                type=edge.type.value,
                properties=edge.properties,
                weight=edge.weight,
            )
        else:
            # 添加新边
            self._edges[edge.id] = edge
            self._graph.add_edge(
                edge.source_id,
                edge.target_id,
                key=edge.id,
                type=edge.type.value,
                properties=edge.properties,
                weight=edge.weight,
            )
        return True
    
    def add_edges(self, edges: List[Edge]) -> int:
        """批量添加边"""
        count = 0
        for edge in edges:
            if self.add_edge(edge):
                count += 1
        return count
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """获取边"""
        return self._edges.get(edge_id)
    
    def delete_edge(self, edge_id: str) -> bool:
        """删除边"""
        if edge_id not in self._edges:
            return False
        
        edge = self._edges[edge_id]
        del self._edges[edge_id]
        
        try:
            self._graph.remove_edge(edge.source_id, edge.target_id, key=edge_id)
        except nx.NetworkXError:
            pass
        
        return True
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[Edge]:
        """根据类型获取边"""
        return [
            edge for edge in self._edges.values()
            if edge.type == edge_type
        ]
    
    # ==================== 关系查询 ====================
    
    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "both",
        limit: int = 100,
    ) -> List[Tuple[Node, Edge]]:
        """获取节点的邻居"""
        if node_id not in self._nodes:
            return []
        
        neighbors = []
        
        if direction in ("out", "both"):
            # 出边
            for _, target_id, edge_key, edge_data in self._graph.out_edges(
                node_id, keys=True, data=True
            ):
                edge = self._edges.get(edge_key)
                if edge is None:
                    continue
                
                if edge_type is not None and edge.type != edge_type:
                    continue
                
                neighbor = self._nodes.get(target_id)
                if neighbor is not None:
                    neighbors.append((neighbor, edge))
        
        if direction in ("in", "both"):
            # 入边
            for source_id, _, edge_key, edge_data in self._graph.in_edges(
                node_id, keys=True, data=True
            ):
                edge = self._edges.get(edge_key)
                if edge is None:
                    continue
                
                if edge_type is not None and edge.type != edge_type:
                    continue
                
                neighbor = self._nodes.get(source_id)
                if neighbor is not None:
                    neighbors.append((neighbor, edge))
        
        return neighbors[:limit]
    
    def get_out_edges(self, node_id: str) -> List[Edge]:
        """获取节点的出边"""
        if node_id not in self._nodes:
            return []
        
        edges = []
        for _, _, edge_key in self._graph.out_edges(node_id, keys=True):
            edge = self._edges.get(edge_key)
            if edge is not None:
                edges.append(edge)
        return edges
    
    def get_in_edges(self, node_id: str) -> List[Edge]:
        """获取节点的入边"""
        if node_id not in self._nodes:
            return []
        
        edges = []
        for _, _, edge_key in self._graph.in_edges(node_id, keys=True):
            edge = self._edges.get(edge_key)
            if edge is not None:
                edges.append(edge)
        return edges
    
    # ==================== 路径查询 ====================
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
    ) -> Optional[GraphPath]:
        """查找两个节点之间的最短路径"""
        if source_id not in self._nodes or target_id not in self._nodes:
            return None
        
        try:
            # 使用 BFS 查找最短路径
            path_nodes = nx.shortest_path(
                self._graph,
                source=source_id,
                target=target_id,
            )
            
            if len(path_nodes) > max_depth + 1:
                return None
            
            # 构建路径对象
            nodes = [self._nodes[nid] for nid in path_nodes]
            edges = []
            total_weight = 0.0
            
            for i in range(len(path_nodes) - 1):
                source = path_nodes[i]
                target = path_nodes[i + 1]
                
                # 获取两个节点之间的边
                edge_data = self._graph.get_edge_data(source, target)
                if edge_data:
                    # 取第一条边
                    edge_key = list(edge_data.keys())[0]
                    edge = self._edges.get(edge_key)
                    if edge:
                        edges.append(edge)
                        total_weight += edge.weight
            
            return GraphPath(
                nodes=nodes,
                edges=edges,
                total_weight=total_weight,
            )
        except nx.NetworkXNoPath:
            return None
    
    def find_all_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
        limit: int = 10,
    ) -> List[GraphPath]:
        """查找两个节点之间的所有路径"""
        if source_id not in self._nodes or target_id not in self._nodes:
            return []
        
        try:
            # 查找所有简单路径
            all_paths = nx.all_simple_paths(
                self._graph,
                source=source_id,
                target=target_id,
                cutoff=max_depth,
            )
            
            paths = []
            for path_nodes in all_paths:
                if len(paths) >= limit:
                    break
                
                # 构建路径对象
                nodes = [self._nodes[nid] for nid in path_nodes]
                edges = []
                total_weight = 0.0
                
                for i in range(len(path_nodes) - 1):
                    source = path_nodes[i]
                    target = path_nodes[i + 1]
                    
                    # 获取两个节点之间的边
                    edge_data = self._graph.get_edge_data(source, target)
                    if edge_data:
                        # 取第一条边
                        edge_key = list(edge_data.keys())[0]
                        edge = self._edges.get(edge_key)
                        if edge:
                            edges.append(edge)
                            total_weight += edge.weight
                
                paths.append(GraphPath(
                    nodes=nodes,
                    edges=edges,
                    total_weight=total_weight,
                ))
            
            return paths
        except nx.NetworkXNoPath:
            return []
    
    # ==================== 图查询 ====================
    
    def query(
        self,
        query_str: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """执行查询
        
        注意：本地图存储不支持 Cypher 查询。
        此方法提供简单的节点/边查询功能。
        """
        # 简单的查询解析
        # 支持: "nodes", "edges", "node:{id}", "edge:{id}"
        query_str = query_str.strip().lower()
        
        if query_str == "nodes":
            return [node.to_dict() for node in self._nodes.values()]
        
        elif query_str == "edges":
            return [edge.to_dict() for edge in self._edges.values()]
        
        elif query_str.startswith("node:"):
            node_id = query_str[5:]
            node = self.get_node(node_id)
            return [node.to_dict()] if node else []
        
        elif query_str.startswith("edge:"):
            edge_id = query_str[5:]
            edge = self.get_edge(edge_id)
            return [edge.to_dict()] if edge else []
        
        else:
            raise ValueError(f"Unsupported query: {query_str}")
    
    def get_subgraph(
        self,
        node_ids: List[str],
        include_edges: bool = True,
    ) -> Tuple[List[Node], List[Edge]]:
        """获取子图"""
        nodes = [
            self._nodes[nid] for nid in node_ids
            if nid in self._nodes
        ]
        
        edges = []
        if include_edges:
            node_set = set(node_ids)
            for edge in self._edges.values():
                if edge.source_id in node_set and edge.target_id in node_set:
                    edges.append(edge)
        
        return nodes, edges
    
    # ==================== 统计信息 ====================
    
    def count_nodes(self) -> int:
        """获取节点总数"""
        return len(self._nodes)
    
    def count_edges(self) -> int:
        """获取边总数"""
        return len(self._edges)
    
    def get_node_types(self) -> Set[NodeType]:
        """获取所有节点类型"""
        return {node.type for node in self._nodes.values()}
    
    def get_edge_types(self) -> Set[EdgeType]:
        """获取所有边类型"""
        return {edge.type for edge in self._edges.values()}
    
    # ==================== 持久化 ====================
    
    def save(self, path: Optional[str] = None) -> None:
        """保存图到文件
        
        Args:
            path: 保存路径（可选，默认使用 self.graph_path）
        """
        save_path = Path(path or self.graph_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存为 JSON 格式
        data = {
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges.values()],
        }
        
        with open(save_path / "graph.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: Optional[str] = None) -> None:
        """从文件加载图
        
        Args:
            path: 加载路径（可选，默认使用 self.graph_path）
        """
        load_path = Path(path or self.graph_path)
        graph_file = load_path / "graph.json"
        
        if not graph_file.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_file}")
        
        with open(graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 清空现有数据
        self.clear()
        
        # 导入节点
        for node_data in data.get("nodes", []):
            node = Node.from_dict(node_data)
            self.add_node(node)
        
        # 导入边
        for edge_data in data.get("edges", []):
            edge = Edge.from_dict(edge_data)
            self.add_edge(edge)
    
    def clear(self) -> None:
        """清空图"""
        self._graph.clear()
        self._nodes.clear()
        self._edges.clear()
    
    # ==================== real_kg 格式支持 ====================
    
    def load_from_real_kg(self, data: Dict[str, Any]) -> None:
        """从 real_kg 格式加载图
        
        real_kg 格式示例:
        {
            "nodes": [
                {"id": "entity1", "name": "Entity 1", "type": "entity", ...},
                ...
            ],
            "edges": [
                {"source": "entity1", "target": "entity2", "relation": "related_to", ...},
                ...
            ]
        }
        
        Args:
            data: real_kg 格式的数据字典
        """
        # 清空现有数据
        self.clear()
        
        # 导入节点
        for node_data in data.get("nodes", []):
            node_id = node_data.get("id", str(uuid.uuid4()))
            node_name = node_data.get("name", "")
            node_type_str = node_data.get("type", "entity")
            
            # 解析节点类型
            try:
                node_type = NodeType(node_type_str.lower())
            except ValueError:
                node_type = NodeType.OTHER
            
            # 提取其他属性
            properties = {
                k: v for k, v in node_data.items()
                if k not in ["id", "name", "type", "embedding"]
            }
            
            node = Node(
                id=node_id,
                type=node_type,
                name=node_name,
                properties=properties,
                embedding=node_data.get("embedding"),
            )
            self.add_node(node)
        
        # 导入边
        for edge_data in data.get("edges", []):
            source_id = edge_data.get("source")
            target_id = edge_data.get("target")
            relation = edge_data.get("relation", "related_to")
            
            if not source_id or not target_id:
                continue
            
            # 解析边类型
            try:
                edge_type = EdgeType(relation.lower())
            except ValueError:
                edge_type = EdgeType.OTHER
            
            # 提取其他属性
            properties = {
                k: v for k, v in edge_data.items()
                if k not in ["source", "target", "relation", "weight"]
            }
            
            edge = Edge(
                id=str(uuid.uuid4()),
                source_id=source_id,
                target_id=target_id,
                type=edge_type,
                properties=properties,
                weight=edge_data.get("weight", 1.0),
            )
            self.add_edge(edge)
    
    def to_real_kg(self) -> Dict[str, Any]:
        """导出为 real_kg 格式
        
        Returns:
            real_kg 格式的数据字典
        """
        nodes = []
        for node in self._nodes.values():
            node_dict = {
                "id": node.id,
                "name": node.name,
                "type": node.type.value,
            }
            node_dict.update(node.properties)
            if node.embedding is not None:
                node_dict["embedding"] = node.embedding
            nodes.append(node_dict)
        
        edges = []
        for edge in self._edges.values():
            edge_dict = {
                "source": edge.source_id,
                "target": edge.target_id,
                "relation": edge.type.value,
                "weight": edge.weight,
            }
            edge_dict.update(edge.properties)
            edges.append(edge_dict)
        
        return {"nodes": nodes, "edges": edges}
    
    # ==================== 图算法 ====================
    
    def get_connected_components(self) -> List[Set[str]]:
        """获取连通分量"""
        # 转换为无向图
        undirected = self._graph.to_undirected()
        return list(nx.connected_components(undirected))
    
    def get_node_degree(self, node_id: str) -> int:
        """获取节点度数"""
        return self._graph.degree(node_id)
    
    def get_pagerank(self) -> Dict[str, float]:
        """计算 PageRank"""
        return nx.pagerank(self._graph)
    
    def get_clustering_coefficient(self) -> Dict[str, float]:
        """计算聚类系数"""
        undirected = self._graph.to_undirected()
        return nx.clustering(undirected)
    
    def get_node_centrality(self) -> Dict[str, float]:
        """计算节点中心性"""
        return nx.degree_centrality(self._graph)


if __name__ == "__main__":
    import os
    import shutil
    
    print("=" * 50)
    print("测试 Local Graph Store 模块")
    print("=" * 50)
    
    # 使用配置中的默认路径
    from src.utils.config import get_config
    config = get_config()
    
    # 测试 1: 创建图实例
    graph = LocalGraphStore()
    print(f"✓ 创建图实例: nodes={graph.count_nodes()}, edges={graph.count_edges()}")
    
    # 测试 2: 添加节点
    nodes = [
        Node(id="n1", type=NodeType.ENTITY, name="实体1", properties={"type": "概念"}),
        Node(id="n2", type=NodeType.ENTITY, name="实体2", properties={"type": "概念"}),
        Node(id="n3", type=NodeType.ENTITY, name="实体3", properties={"type": "概念"}),
    ]
    for node in nodes:
        graph.add_node(node)
    print(f"✓ 添加节点: count={graph.count_nodes()}")
    
    # 测试 3: 添加边
    edges = [
        Edge(id="e1", source_id="n1", target_id="n2", type=EdgeType.RELATED_TO, weight=0.8),
        Edge(id="e2", source_id="n2", target_id="n3", type=EdgeType.RELATED_TO, weight=0.6),
    ]
    for edge in edges:
        graph.add_edge(edge)
    print(f"✓ 添加边: count={graph.count_edges()}")
    
    # 测试 4: 获取节点
    node = graph.get_node("n1")
    print(f"✓ 获取节点: name={node.name}")
    
    # 测试 5: 获取邻居
    neighbors = graph.get_neighbors("n1")
    print(f"✓ 获取邻居: {[(n.id, e.type) for n, e in neighbors]}")
    
    # 测试 6: 查询
    results = graph.query("nodes")
    print(f"✓ 查询节点: 找到 {len(results)} 个")
    
    # 测试 7: 路径查找
    paths = graph.find_path("n1", "n3")
    print(f"✓ 路径查找: {paths}")
    
    # 测试 8: 图算法
    centrality = graph.get_node_centrality()
    print(f"✓ 节点中心性: {centrality}")
    
    # 测试 9: 保存和加载（使用配置中的默认路径）
    test_graph_path = os.path.join(config.local_graph_path, "test_graph")
    graph.save(test_graph_path)
    print(f"✓ 保存图到: {test_graph_path}")
    
    graph2 = LocalGraphStore()
    graph2.load(test_graph_path)
    print(f"✓ 加载图: nodes={graph2.count_nodes()}, edges={graph2.count_edges()}")
    
    # 清理测试数据
    if os.path.exists(test_graph_path):
        shutil.rmtree(test_graph_path)
    
    # 测试 10: 清空
    graph.clear()
    print(f"✓ 清空图: nodes={graph.count_nodes()}")
    
    print("\n所有测试通过!")
