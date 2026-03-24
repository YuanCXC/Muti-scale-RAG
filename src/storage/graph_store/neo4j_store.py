# -*- coding: utf-8 -*-
"""Neo4j 图数据库实现

使用 Neo4j 数据库实现图存储，支持 Cypher 查询。
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from neo4j import GraphDatabase, Driver, Session
    from neo4j.exceptions import ServiceUnavailable, AuthError
except ImportError:
    raise ImportError(
        "使用 Neo4jGraphStore 需要安装 neo4j。可通过以下命令安装：pip install neo4j"
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


class Neo4jGraphStore(GraphStoreBase):
    """Neo4j 图数据库实现
    
    使用 Neo4j 数据库存储和查询图数据。
    
    Attributes:
        uri: Neo4j 连接 URI
        user: 用户名
        password: 密码
        database: 数据库名称
    """
    
    def __init__(self, **kwargs: Any):
        """初始化 Neo4j 图存储
        
        Args:
            **kwargs: 额外参数
        """
        super().__init__(**kwargs)
        
        config = get_config()
        
        self.uri = config.neo4j_uri
        self.user = config.neo4j_user
        self.password = config.neo4j_password
        self.database = config.neo4j_database
        
        # 初始化连接
        self._driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self) -> None:
        """连接到 Neo4j 数据库"""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            # 验证连接
            with self._driver.session(database=self.database) as session:
                session.run("RETURN 1")
        except ServiceUnavailable as e:
            raise ConnectionError(
                f"Failed to connect to Neo4j at {self.uri}: {e}"
            )
        except AuthError as e:
            raise AuthError(
                f"Authentication failed for Neo4j user {self.user}: {e}"
            )
    
    def _get_session(self) -> Session:
        """获取数据库会话"""
        if self._driver is None:
            self._connect()
        return self._driver.session(database=self.database)
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
    
    # ==================== 节点操作 ====================
    
    def add_node(self, node: Node) -> bool:
        """添加节点"""
        try:
            with self._get_session() as session:
                query = """
                MERGE (n:Node {id: $id})
                SET n.type = $type,
                    n.name = $name,
                    n.properties = $properties,
                    n.embedding = $embedding
                RETURN n
                """
                result = session.run(
                    query,
                    id=node.id,
                    type=node.type.value,
                    name=node.name,
                    properties=json.dumps(node.properties),
                    embedding=node.embedding,
                )
                return result.single() is not None
        except Exception as e:
            raise RuntimeError(f"Failed to add node: {e}")
    
    def add_nodes(self, nodes: List[Node]) -> int:
        """批量添加节点"""
        count = 0
        for node in nodes:
            if self.add_node(node):
                count += 1
        return count
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """获取节点"""
        try:
            with self._get_session() as session:
                query = """
                MATCH (n:Node {id: $id})
                RETURN n.id, n.type, n.name, n.properties, n.embedding
                """
                result = session.run(query, id=node_id)
                record = result.single()
                
                if record is None:
                    return None
                
                return Node(
                    id=record["n.id"],
                    type=NodeType(record["n.type"]),
                    name=record["n.name"],
                    properties=json.loads(record["n.properties"] or "{}"),
                    embedding=record["n.embedding"],
                )
        except Exception as e:
            raise RuntimeError(f"Failed to get node: {e}")
    
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """更新节点属性"""
        try:
            with self._get_session() as session:
                query = """
                MATCH (n:Node {id: $id})
                SET n.properties = $properties
                RETURN n
                """
                result = session.run(
                    query,
                    id=node_id,
                    properties=json.dumps(properties),
                )
                return result.single() is not None
        except Exception as e:
            raise RuntimeError(f"Failed to update node: {e}")
    
    def delete_node(self, node_id: str) -> bool:
        """删除节点"""
        try:
            with self._get_session() as session:
                query = """
                MATCH (n:Node {id: $id})
                DETACH DELETE n
                """
                session.run(query, id=node_id)
                return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete node: {e}")
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """根据类型获取节点"""
        try:
            with self._get_session() as session:
                query = """
                MATCH (n:Node {type: $type})
                RETURN n.id, n.type, n.name, n.properties, n.embedding
                """
                result = session.run(query, type=node_type.value)
                
                nodes = []
                for record in result:
                    nodes.append(Node(
                        id=record["n.id"],
                        type=NodeType(record["n.type"]),
                        name=record["n.name"],
                        properties=json.loads(record["n.properties"] or "{}"),
                        embedding=record["n.embedding"],
                    ))
                return nodes
        except Exception as e:
            raise RuntimeError(f"Failed to get nodes by type: {e}")
    
    def get_nodes_by_property(self, key: str, value: Any) -> List[Node]:
        """根据属性获取节点"""
        try:
            with self._get_session() as session:
                query = """
                MATCH (n:Node)
                WHERE n.properties CONTAINS $search
                RETURN n.id, n.type, n.name, n.properties, n.embedding
                """
                search = json.dumps({key: value})
                result = session.run(query, search=search)
                
                nodes = []
                for record in result:
                    props = json.loads(record["n.properties"] or "{}")
                    if props.get(key) == value:
                        nodes.append(Node(
                            id=record["n.id"],
                            type=NodeType(record["n.type"]),
                            name=record["n.name"],
                            properties=props,
                            embedding=record["n.embedding"],
                        ))
                return nodes
        except Exception as e:
            raise RuntimeError(f"Failed to get nodes by property: {e}")
    
    # ==================== 边操作 ====================
    
    def add_edge(self, edge: Edge) -> bool:
        """添加边"""
        try:
            with self._get_session() as session:
                query = """
                MATCH (source:Node {id: $source_id})
                MATCH (target:Node {id: $target_id})
                MERGE (source)-[r:EDGE {id: $id}]->(target)
                SET r.type = $type,
                    r.properties = $properties,
                    r.weight = $weight
                RETURN r
                """
                result = session.run(
                    query,
                    id=edge.id,
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    type=edge.type.value,
                    properties=json.dumps(edge.properties),
                    weight=edge.weight,
                )
                return result.single() is not None
        except Exception as e:
            raise RuntimeError(f"Failed to add edge: {e}")
    
    def add_edges(self, edges: List[Edge]) -> int:
        """批量添加边"""
        count = 0
        for edge in edges:
            if self.add_edge(edge):
                count += 1
        return count
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """获取边"""
        try:
            with self._get_session() as session:
                query = """
                MATCH ()-[r:EDGE {id: $id}]-()
                RETURN r.id, r.type, r.properties, r.weight,
                       startNode(r).id as source_id,
                       endNode(r).id as target_id
                """
                result = session.run(query, id=edge_id)
                record = result.single()
                
                if record is None:
                    return None
                
                return Edge(
                    id=record["r.id"],
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    type=EdgeType(record["r.type"]),
                    properties=json.loads(record["r.properties"] or "{}"),
                    weight=record["r.weight"],
                )
        except Exception as e:
            raise RuntimeError(f"Failed to get edge: {e}")
    
    def delete_edge(self, edge_id: str) -> bool:
        """删除边"""
        try:
            with self._get_session() as session:
                query = """
                MATCH ()-[r:EDGE {id: $id}]-()
                DELETE r
                """
                session.run(query, id=edge_id)
                return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete edge: {e}")
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[Edge]:
        """根据类型获取边"""
        try:
            with self._get_session() as session:
                query = """
                MATCH ()-[r:EDGE {type: $type}]-()
                RETURN r.id, r.type, r.properties, r.weight,
                       startNode(r).id as source_id,
                       endNode(r).id as target_id
                """
                result = session.run(query, type=edge_type.value)
                
                edges = []
                for record in result:
                    edges.append(Edge(
                        id=record["r.id"],
                        source_id=record["source_id"],
                        target_id=record["target_id"],
                        type=EdgeType(record["r.type"]),
                        properties=json.loads(record["r.properties"] or "{}"),
                        weight=record["r.weight"],
                    ))
                return edges
        except Exception as e:
            raise RuntimeError(f"Failed to get edges by type: {e}")
    
    # ==================== 关系查询 ====================
    
    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "both",
        limit: int = 100,
    ) -> List[Tuple[Node, Edge]]:
        """获取节点的邻居"""
        try:
            with self._get_session() as session:
                # 构建方向条件
                if direction == "out":
                    pattern = "(n:Node {id: $node_id})-[r:EDGE]->(neighbor:Node)"
                elif direction == "in":
                    pattern = "(neighbor:Node)-[r:EDGE]->(n:Node {id: $node_id})"
                else:
                    pattern = "(n:Node {id: $node_id})-[r:EDGE]-(neighbor:Node)"
                
                # 构建类型条件
                type_condition = ""
                params = {"node_id": node_id, "limit": limit}
                if edge_type is not None:
                    type_condition = " AND r.type = $edge_type"
                    params["edge_type"] = edge_type.value
                
                query = f"""
                MATCH {pattern}
                WHERE 1=1 {type_condition}
                RETURN neighbor.id, neighbor.type, neighbor.name,
                       neighbor.properties, neighbor.embedding,
                       r.id, r.type, r.properties, r.weight,
                       startNode(r).id as source_id,
                       endNode(r).id as target_id
                LIMIT $limit
                """
                
                result = session.run(query, **params)
                
                neighbors = []
                for record in result:
                    node = Node(
                        id=record["neighbor.id"],
                        type=NodeType(record["neighbor.type"]),
                        name=record["neighbor.name"],
                        properties=json.loads(record["neighbor.properties"] or "{}"),
                        embedding=record["neighbor.embedding"],
                    )
                    edge = Edge(
                        id=record["r.id"],
                        source_id=record["source_id"],
                        target_id=record["target_id"],
                        type=EdgeType(record["r.type"]),
                        properties=json.loads(record["r.properties"] or "{}"),
                        weight=record["r.weight"],
                    )
                    neighbors.append((node, edge))
                
                return neighbors
        except Exception as e:
            raise RuntimeError(f"Failed to get neighbors: {e}")
    
    def get_out_edges(self, node_id: str) -> List[Edge]:
        """获取节点的出边"""
        try:
            with self._get_session() as session:
                query = """
                MATCH (n:Node {id: $node_id})-[r:EDGE]->()
                RETURN r.id, r.type, r.properties, r.weight,
                       startNode(r).id as source_id,
                       endNode(r).id as target_id
                """
                result = session.run(query, node_id=node_id)
                
                edges = []
                for record in result:
                    edges.append(Edge(
                        id=record["r.id"],
                        source_id=record["source_id"],
                        target_id=record["target_id"],
                        type=EdgeType(record["r.type"]),
                        properties=json.loads(record["r.properties"] or "{}"),
                        weight=record["r.weight"],
                    ))
                return edges
        except Exception as e:
            raise RuntimeError(f"Failed to get out edges: {e}")
    
    def get_in_edges(self, node_id: str) -> List[Edge]:
        """获取节点的入边"""
        try:
            with self._get_session() as session:
                query = """
                MATCH ()-[r:EDGE]->(n:Node {id: $node_id})
                RETURN r.id, r.type, r.properties, r.weight,
                       startNode(r).id as source_id,
                       endNode(r).id as target_id
                """
                result = session.run(query, node_id=node_id)
                
                edges = []
                for record in result:
                    edges.append(Edge(
                        id=record["r.id"],
                        source_id=record["source_id"],
                        target_id=record["target_id"],
                        type=EdgeType(record["r.type"]),
                        properties=json.loads(record["r.properties"] or "{}"),
                        weight=record["r.weight"],
                    ))
                return edges
        except Exception as e:
            raise RuntimeError(f"Failed to get in edges: {e}")
    
    # ==================== 路径查询 ====================
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
    ) -> Optional[GraphPath]:
        """查找两个节点之间的路径"""
        paths = self.find_all_paths(source_id, target_id, max_depth, limit=1)
        return paths[0] if paths else None
    
    def find_all_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
        limit: int = 10,
    ) -> List[GraphPath]:
        """查找两个节点之间的所有路径"""
        try:
            with self._get_session() as session:
                query = """
                MATCH path = (source:Node {id: $source_id})-[:EDGE*1..{max_depth}]-(target:Node {id: $target_id})
                RETURN path
                LIMIT $limit
                """
                result = session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    max_depth=max_depth,
                    limit=limit,
                )
                
                paths = []
                for record in result:
                    path = record["path"]
                    
                    # 提取节点
                    nodes = []
                    for node in path.nodes:
                        nodes.append(Node(
                            id=node["id"],
                            type=NodeType(node["type"]),
                            name=node["name"],
                            properties=json.loads(node["properties"] or "{}"),
                            embedding=node.get("embedding"),
                        ))
                    
                    # 提取边
                    edges = []
                    for rel in path.relationships:
                        edges.append(Edge(
                            id=rel["id"],
                            source_id=rel.start_node["id"],
                            target_id=rel.end_node["id"],
                            type=EdgeType(rel["type"]),
                            properties=json.loads(rel["properties"] or "{}"),
                            weight=rel["weight"],
                        ))
                    
                    # 计算总权重
                    total_weight = sum(e.weight for e in edges)
                    
                    paths.append(GraphPath(
                        nodes=nodes,
                        edges=edges,
                        total_weight=total_weight,
                    ))
                
                return paths
        except Exception as e:
            raise RuntimeError(f"Failed to find paths: {e}")
    
    # ==================== 图查询 ====================
    
    def query(
        self,
        query_str: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """执行 Cypher 查询"""
        try:
            with self._get_session() as session:
                result = session.run(query_str, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            raise RuntimeError(f"Failed to execute query: {e}")
    
    def get_subgraph(
        self,
        node_ids: List[str],
        include_edges: bool = True,
    ) -> Tuple[List[Node], List[Edge]]:
        """获取子图"""
        try:
            with self._get_session() as session:
                # 获取节点
                node_query = """
                MATCH (n:Node)
                WHERE n.id IN $node_ids
                RETURN n.id, n.type, n.name, n.properties, n.embedding
                """
                node_result = session.run(node_query, node_ids=node_ids)
                
                nodes = []
                for record in node_result:
                    nodes.append(Node(
                        id=record["n.id"],
                        type=NodeType(record["n.type"]),
                        name=record["n.name"],
                        properties=json.loads(record["n.properties"] or "{}"),
                        embedding=record["n.embedding"],
                    ))
                
                edges = []
                if include_edges:
                    # 获取边
                    edge_query = """
                    MATCH (source:Node)-[r:EDGE]->(target:Node)
                    WHERE source.id IN $node_ids AND target.id IN $node_ids
                    RETURN r.id, r.type, r.properties, r.weight,
                           source.id as source_id, target.id as target_id
                    """
                    edge_result = session.run(edge_query, node_ids=node_ids)
                    
                    for record in edge_result:
                        edges.append(Edge(
                            id=record["r.id"],
                            source_id=record["source_id"],
                            target_id=record["target_id"],
                            type=EdgeType(record["r.type"]),
                            properties=json.loads(record["r.properties"] or "{}"),
                            weight=record["r.weight"],
                        ))
                
                return nodes, edges
        except Exception as e:
            raise RuntimeError(f"Failed to get subgraph: {e}")
    
    # ==================== 统计信息 ====================
    
    def count_nodes(self) -> int:
        """获取节点总数"""
        try:
            with self._get_session() as session:
                result = session.run("MATCH (n:Node) RETURN count(n) as count")
                record = result.single()
                return record["count"] if record else 0
        except Exception as e:
            raise RuntimeError(f"Failed to count nodes: {e}")
    
    def count_edges(self) -> int:
        """获取边总数"""
        try:
            with self._get_session() as session:
                result = session.run("MATCH ()-[r:EDGE]->() RETURN count(r) as count")
                record = result.single()
                return record["count"] if record else 0
        except Exception as e:
            raise RuntimeError(f"Failed to count edges: {e}")
    
    def get_node_types(self) -> Set[NodeType]:
        """获取所有节点类型"""
        try:
            with self._get_session() as session:
                result = session.run("MATCH (n:Node) RETURN DISTINCT n.type as type")
                return {NodeType(record["type"]) for record in result}
        except Exception as e:
            raise RuntimeError(f"Failed to get node types: {e}")
    
    def get_edge_types(self) -> Set[EdgeType]:
        """获取所有边类型"""
        try:
            with self._get_session() as session:
                result = session.run("MATCH ()-[r:EDGE]->() RETURN DISTINCT r.type as type")
                return {EdgeType(record["type"]) for record in result}
        except Exception as e:
            raise RuntimeError(f"Failed to get edge types: {e}")
    
    # ==================== 持久化 ====================
    
    def save(self, path: str) -> None:
        """保存图到文件
        
        注意：Neo4j 通常不需要手动保存，数据会自动持久化。
        此方法导出图数据为 JSON 格式。
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 导出所有节点和边
        nodes = []
        with self._get_session() as session:
            result = session.run("""
                MATCH (n:Node)
                RETURN n.id, n.type, n.name, n.properties, n.embedding
            """)
            for record in result:
                nodes.append({
                    "id": record["n.id"],
                    "type": record["n.type"],
                    "name": record["n.name"],
                    "properties": json.loads(record["n.properties"] or "{}"),
                    "embedding": record["n.embedding"],
                })
        
        edges = []
        with self._get_session() as session:
            result = session.run("""
                MATCH ()-[r:EDGE]->()
                RETURN r.id, r.type, r.properties, r.weight,
                       startNode(r).id as source_id,
                       endNode(r).id as target_id
            """)
            for record in result:
                edges.append({
                    "id": record["r.id"],
                    "source_id": record["source_id"],
                    "target_id": record["target_id"],
                    "type": record["r.type"],
                    "properties": json.loads(record["r.properties"] or "{}"),
                    "weight": record["r.weight"],
                })
        
        # 保存到文件
        data = {"nodes": nodes, "edges": edges}
        with open(path / "graph.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str) -> None:
        """从文件加载图
        
        注意：此方法会清空现有数据并导入新数据。
        """
        path = Path(path)
        graph_file = path / "graph.json"
        
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
        try:
            with self._get_session() as session:
                session.run("MATCH (n) DETACH DELETE n")
        except Exception as e:
            raise RuntimeError(f"Failed to clear graph: {e}")
    
    def __del__(self):
        """析构函数"""
        self.close()


if __name__ == "__main__":
    import os
    
    print("=" * 50)
    print("测试 Neo4j Graph Store 模块")
    print("=" * 50)
    
    # 使用配置中的 Neo4j 连接参数
    from src.utils.config import get_config
    config = get_config()
    
    try:
        graph = Neo4jGraphStore()
        print(f"✓ 连接 Neo4j: {config.neo4j_uri}")
    except Exception as e:
        print(f"⚠ 无法连接 Neo4j: {e}")
        print("  跳过 Neo4j 测试。请确保 Neo4j 服务正在运行。")
        print("\n测试跳过（需要 Neo4j 服务）!")
        exit(0)
    
    try:
        # 清空测试数据
        graph.clear()
        print(f"✓ 清空图: nodes={graph.count_nodes()}, edges={graph.count_edges()}")
        
        # 测试 1: 添加节点
        nodes = [
            Node(id="n1", type=NodeType.ENTITY, name="实体1", properties={"type": "概念"}),
            Node(id="n2", type=NodeType.ENTITY, name="实体2", properties={"type": "概念"}),
            Node(id="n3", type=NodeType.ENTITY, name="实体3", properties={"type": "概念"}),
        ]
        for node in nodes:
            graph.add_node(node)
        print(f"✓ 添加节点: count={graph.count_nodes()}")
        
        # 测试 2: 添加边
        edges = [
            Edge(id="e1", source_id="n1", target_id="n2", type=EdgeType.RELATED_TO, weight=0.8),
            Edge(id="e2", source_id="n2", target_id="n3", type=EdgeType.RELATED_TO, weight=0.6),
        ]
        for edge in edges:
            graph.add_edge(edge)
        print(f"✓ 添加边: count={graph.count_edges()}")
        
        # 测试 3: 获取节点
        node = graph.get_node("n1")
        print(f"✓ 获取节点: name={node.name if node else 'None'}")
        
        # 测试 4: 获取邻居
        neighbors = graph.get_neighbors("n1")
        print(f"✓ 获取邻居: {[(n.id, e.type) for n, e in neighbors]}")
        
        # 测试 5: 查询
        results = graph.query("MATCH (n:Node) RETURN n.id, n.name LIMIT 3")
        print(f"✓ Cypher 查询: 找到 {len(results)} 个结果")
        
        # 测试 6: 路径查找
        path = graph.find_path("n1", "n3")
        print(f"✓ 路径查找: {'找到路径' if path else '未找到路径'}")
        
        # 测试 7: 统计信息
        node_count = graph.count_nodes()
        edge_count = graph.count_edges()
        print(f"✓ 统计信息: nodes={node_count}, edges={edge_count}")
        
        # 测试 8: 清理
        graph.clear()
        graph.close()
        print(f"✓ 清理并关闭连接")
        
        print("\n所有测试通过!")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        graph.close()
        raise
