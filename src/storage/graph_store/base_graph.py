# -*- coding: utf-8 -*-
"""基础图存储抽象类

定义图存储的统一接口，支持不同的图数据库实现。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class NodeType(Enum):
    """节点类型枚举"""
    ENTITY = "entity"       # 实体节点
    CONCEPT = "concept"     # 概念节点
    DOCUMENT = "document"   # 文档节点
    CHUNK = "chunk"         # 切片节点
    TOPIC = "topic"         # 主题节点
    OTHER = "other"         # 其他类型


class EdgeType(Enum):
    """边类型枚举"""
    RELATED_TO = "related_to"       # 相关关系
    PART_OF = "part_of"             # 部分关系
    HAS_ENTITY = "has_entity"       # 包含实体
    SIMILAR_TO = "similar_to"       # 相似关系
    DERIVED_FROM = "derived_from"   # 派生关系
    REFERENCES = "references"       # 引用关系
    INCLUDES = "includes"           # 包含关系
    APPLIES_TO = "applies_to"       # 应用于关系
    OTHER = "other"                 # 其他类型


@dataclass
class Node:
    """图节点
    
    Attributes:
        id: 节点唯一标识
        type: 节点类型
        name: 节点名称
        properties: 节点属性
        embedding: 节点嵌入向量（可选）
    """
    id: str
    type: NodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "properties": self.properties,
        }
        if self.embedding is not None:
            result["embedding"] = self.embedding
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """从字典创建实例"""
        return cls(
            id=data["id"],
            type=NodeType(data["type"]),
            name=data["name"],
            properties=data.get("properties", {}),
            embedding=data.get("embedding"),
        )


@dataclass
class Edge:
    """图边
    
    Attributes:
        id: 边唯一标识
        source_id: 源节点ID
        target_id: 目标节点ID
        type: 边类型
        properties: 边属性
        weight: 边权重
    """
    id: str
    source_id: str
    target_id: str
    type: EdgeType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
            "properties": self.properties,
            "weight": self.weight,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        """从字典创建实例"""
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=EdgeType(data["type"]),
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0),
        )


@dataclass
class GraphPath:
    """图路径
    
    Attributes:
        nodes: 路径上的节点列表
        edges: 路径上的边列表
        total_weight: 路径总权重
    """
    nodes: List[Node]
    edges: List[Edge]
    total_weight: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "total_weight": self.total_weight,
        }
    
    def get_node_ids(self) -> List[str]:
        """获取路径上的节点ID列表"""
        return [n.id for n in self.nodes]
    
    def get_edge_ids(self) -> List[str]:
        """获取路径上的边ID列表"""
        return [e.id for e in self.edges]


class GraphStoreBase(ABC):
    """图存储抽象基类
    
    定义图存储的统一接口，所有图数据库实现都应继承此类。
    """
    
    def __init__(self, **kwargs: Any):
        """初始化图存储
        
        Args:
            **kwargs: 额外参数
        """
        pass
    
    # ==================== 节点操作 ====================
    
    @abstractmethod
    def add_node(self, node: Node) -> bool:
        """添加节点
        
        Args:
            node: 节点对象
            
        Returns:
            是否添加成功
        """
        pass
    
    @abstractmethod
    def add_nodes(self, nodes: List[Node]) -> int:
        """批量添加节点
        
        Args:
            nodes: 节点列表
            
        Returns:
            成功添加的节点数量
        """
        pass
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Node]:
        """获取节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            节点对象，不存在则返回 None
        """
        pass
    
    @abstractmethod
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """更新节点属性
        
        Args:
            node_id: 节点ID
            properties: 要更新的属性
            
        Returns:
            是否更新成功
        """
        pass
    
    @abstractmethod
    def delete_node(self, node_id: str) -> bool:
        """删除节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            是否删除成功
        """
        pass
    
    @abstractmethod
    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """根据类型获取节点
        
        Args:
            node_type: 节点类型
            
        Returns:
            节点列表
        """
        pass
    
    @abstractmethod
    def get_nodes_by_property(
        self,
        key: str,
        value: Any,
    ) -> List[Node]:
        """根据属性获取节点
        
        Args:
            key: 属性键
            value: 属性值
            
        Returns:
            节点列表
        """
        pass
    
    # ==================== 边操作 ====================
    
    @abstractmethod
    def add_edge(self, edge: Edge) -> bool:
        """添加边
        
        Args:
            edge: 边对象
            
        Returns:
            是否添加成功
        """
        pass
    
    @abstractmethod
    def add_edges(self, edges: List[Edge]) -> int:
        """批量添加边
        
        Args:
            edges: 边列表
            
        Returns:
            成功添加的边数量
        """
        pass
    
    @abstractmethod
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """获取边
        
        Args:
            edge_id: 边ID
            
        Returns:
            边对象，不存在则返回 None
        """
        pass
    
    @abstractmethod
    def delete_edge(self, edge_id: str) -> bool:
        """删除边
        
        Args:
            edge_id: 边ID
            
        Returns:
            是否删除成功
        """
        pass
    
    @abstractmethod
    def get_edges_by_type(self, edge_type: EdgeType) -> List[Edge]:
        """根据类型获取边
        
        Args:
            edge_type: 边类型
            
        Returns:
            边列表
        """
        pass
    
    # ==================== 关系查询 ====================
    
    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "both",
        limit: int = 100,
    ) -> List[Tuple[Node, Edge]]:
        """获取节点的邻居
        
        Args:
            node_id: 节点ID
            edge_type: 边类型过滤（可选）
            direction: 方向 (in, out, both)
            limit: 返回数量限制
            
        Returns:
            (邻居节点, 边) 元组列表
        """
        pass
    
    @abstractmethod
    def get_out_edges(self, node_id: str) -> List[Edge]:
        """获取节点的出边
        
        Args:
            node_id: 节点ID
            
        Returns:
            出边列表
        """
        pass
    
    @abstractmethod
    def get_in_edges(self, node_id: str) -> List[Edge]:
        """获取节点的入边
        
        Args:
            node_id: 节点ID
            
        Returns:
            入边列表
        """
        pass
    
    # ==================== 路径查询 ====================
    
    @abstractmethod
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
    ) -> Optional[GraphPath]:
        """查找两个节点之间的路径
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            max_depth: 最大搜索深度
            
        Returns:
            路径对象，不存在则返回 None
        """
        pass
    
    @abstractmethod
    def find_all_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
        limit: int = 10,
    ) -> List[GraphPath]:
        """查找两个节点之间的所有路径
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            max_depth: 最大搜索深度
            limit: 返回路径数量限制
            
        Returns:
            路径列表
        """
        pass
    
    # ==================== 图查询 ====================
    
    @abstractmethod
    def query(
        self,
        query_str: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """执行查询
        
        Args:
            query_str: 查询语句（Cypher 或其他查询语言）
            parameters: 查询参数
            
        Returns:
            查询结果列表
        """
        pass
    
    @abstractmethod
    def get_subgraph(
        self,
        node_ids: List[str],
        include_edges: bool = True,
    ) -> Tuple[List[Node], List[Edge]]:
        """获取子图
        
        Args:
            node_ids: 节点ID列表
            include_edges: 是否包含节点之间的边
            
        Returns:
            (节点列表, 边列表) 元组
        """
        pass
    
    # ==================== 统计信息 ====================
    
    @abstractmethod
    def count_nodes(self) -> int:
        """获取节点总数
        
        Returns:
            节点总数
        """
        pass
    
    @abstractmethod
    def count_edges(self) -> int:
        """获取边总数
        
        Returns:
            边总数
        """
        pass
    
    @abstractmethod
    def get_node_types(self) -> Set[NodeType]:
        """获取所有节点类型
        
        Returns:
            节点类型集合
        """
        pass
    
    @abstractmethod
    def get_edge_types(self) -> Set[EdgeType]:
        """获取所有边类型
        
        Returns:
            边类型集合
        """
        pass
    
    # ==================== 持久化 ====================
    
    @abstractmethod
    def save(self, path: str) -> None:
        """保存图到文件
        
        Args:
            path: 保存路径
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """从文件加载图
        
        Args:
            path: 加载路径
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空图"""
        pass
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"nodes={self.count_nodes()}, "
            f"edges={self.count_edges()})"
        )


if __name__ == "__main__":
    print("=" * 50)
    print("测试 Graph Store Base 模块")
    print("=" * 50)
    
    # 测试 1: NodeType 枚举
    entity_type = NodeType.ENTITY
    print(f"✓ NodeType 枚举: {entity_type.value}")
    
    # 测试 2: EdgeType 枚举
    relation_type = EdgeType.RELATED_TO
    print(f"✓ EdgeType 枚举: {relation_type.value}")
    
    # 测试 3: Node 类
    node = Node(
        id="node_001",
        type=NodeType.ENTITY,
        name="测试实体",
        properties={"category": "概念"}
    )
    print(f"✓ Node 创建: id={node.id}, name={node.name}")
    
    # 测试 4: Edge 类
    edge = Edge(
        id="edge_001",
        source_id="node_001",
        target_id="node_002",
        type=EdgeType.RELATED_TO,
        weight=0.8
    )
    print(f"✓ Edge 创建: source={edge.source_id}, target={edge.target_id}")
    
    # 测试 5: Node 序列化
    node_dict = node.to_dict()
    node2 = Node.from_dict(node_dict)
    print(f"✓ Node 序列化/反序列化: {node2.name}")
    
    # 测试 6: Edge 序列化
    edge_dict = edge.to_dict()
    edge2 = Edge.from_dict(edge_dict)
    print(f"✓ Edge 序列化/反序列化: weight={edge2.weight}")
    
    # 测试 7: 抽象类不能实例化
    try:
        graph = GraphStoreBase()
    except TypeError:
        print(f"✓ 抽象类无法实例化")
    
    print("\n所有测试通过!")
