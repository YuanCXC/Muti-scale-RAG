# -*- coding: utf-8 -*-
"""图谱工具

封装图谱遍历与实体链接能力。
"""

from typing import Any, Dict, List, Optional, Type

from pydantic import Field

from src.retrievers.graph_retriever import GraphRetriever
from src.storage.graph_store.base_graph import GraphPath, Node
from src.utils.logger import get_logger

from src.tools.base_tool import ToolArgs, ToolBase, ToolResult

logger = get_logger(__name__)


class GraphSearchArgs(ToolArgs):
    """图谱检索工具参数
    
    Attributes:
        query: 查询文本（实体名称或关键词）
        entity: 实体名称（可选）
        top_k: 返回结果数量
        search_type: 搜索类型（entity, neighbors, paths）
        target_entity: 目标实体名称（用于路径查找）
        max_depth: 最大搜索深度
    """
    query: Optional[str] = Field(
        default=None,
        description="查询文本，用于实体匹配和检索"
    )
    entity: Optional[str] = Field(
        default=None,
        description="实体名称，用于查找相关实体或邻居"
    )
    top_k: Optional[int] = Field(
        default=10,
        description="返回结果数量，默认为 10",
        ge=1,
        le=100,
    )
    search_type: Optional[str] = Field(
        default="entity",
        description="搜索类型：entity(实体检索), neighbors(邻居查询), paths(路径查找)",
    )
    target_entity: Optional[str] = Field(
        default=None,
        description="目标实体名称，用于路径查找（search_type=paths 时必填）",
    )
    max_depth: Optional[int] = Field(
        default=2,
        description="最大搜索深度，默认为 2",
        ge=1,
        le=5,
    )


class GraphSearchTool(ToolBase):
    """图谱检索工具
    
    封装图谱遍历与实体链接能力，
    支持实体检索、邻居查询和路径查找。
    
    Attributes:
        name: 工具名称
        description: 工具描述
        args_schema: 参数定义
        retriever: GraphRetriever 实例
    """
    
    name: str = "graph_search"
    description: str = (
        "知识图谱检索工具：在知识图谱中查找实体、关系和路径。"
        "支持三种搜索类型："
        "1. entity - 实体检索：查找与查询匹配的实体及其关系"
        "2. neighbors - 邻居查询：查找实体的相关实体和关系"
        "3. paths - 路径查找：查找两个实体之间的路径"
        "输入实体名称或查询文本，返回相关实体、关系和三元组。"
    )
    args_schema: Type[ToolArgs] = GraphSearchArgs
    
    def __init__(
        self,
        retriever: GraphRetriever,
        **kwargs: Any,
    ):
        """初始化图谱检索工具
        
        Args:
            retriever: GraphRetriever 实例
            **kwargs: 额外配置参数
        """
        super().__init__(**kwargs)
        self.retriever = retriever
        
        logger.info(f"初始化图谱检索工具: retriever={retriever}")
    
    def run(self, **kwargs: Any) -> ToolResult:
        """执行图谱检索
        
        Args:
            **kwargs: 工具参数，包含:
                - query: 查询文本（可选）
                - entity: 实体名称（可选）
                - top_k: 返回结果数量（可选）
                - search_type: 搜索类型（可选）
                - target_entity: 目标实体名称（可选）
                - max_depth: 最大搜索深度（可选）
            
        Returns:
            ToolResult 实例，包含图谱检索结果
        """
        try:
            # 验证参数
            validated_args = self.validate_args(**kwargs)
            
            query = validated_args.get("query")
            entity = validated_args.get("entity")
            top_k = validated_args.get("top_k", 10)
            search_type = validated_args.get("search_type", "entity")
            target_entity = validated_args.get("target_entity")
            max_depth = validated_args.get("max_depth", 2)
            
            # 确定查询文本
            search_query = query or entity
            if not search_query:
                return ToolResult.error_result(
                    error="必须提供 query 或 entity 参数"
                )
            
            logger.info(
                f"执行图谱检索: query='{search_query}', "
                f"search_type={search_type}, top_k={top_k}"
            )
            
            # 根据搜索类型执行不同的检索
            if search_type == "entity":
                result = self._search_entities(
                    query=search_query,
                    top_k=top_k,
                )
            elif search_type == "neighbors":
                result = self._search_neighbors(
                    entity=search_query,
                    max_depth=max_depth,
                )
            elif search_type == "paths":
                if not target_entity:
                    return ToolResult.error_result(
                        error="路径查找需要提供 target_entity 参数"
                    )
                result = self._search_paths(
                    source_entity=search_query,
                    target_entity=target_entity,
                    max_depth=max_depth,
                )
            else:
                return ToolResult.error_result(
                    error=f"不支持的搜索类型: {search_type}"
                )
            
            logger.info(f"图谱检索完成: 找到 {len(result)} 个结果")
            
            return ToolResult.success_result(
                result=result,
                metadata={
                    "query": search_query,
                    "search_type": search_type,
                    "top_k": top_k,
                    "max_depth": max_depth,
                    "result_count": len(result),
                },
            )
            
        except Exception as e:
            error_msg = f"图谱检索失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ToolResult.error_result(error=error_msg)
    
    def _search_entities(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """实体检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            实体列表
        """
        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )
        
        return [result.to_dict() for result in results]
    
    def _search_neighbors(
        self,
        entity: str,
        max_depth: int,
    ) -> List[Dict[str, Any]]:
        """邻居查询
        
        Args:
            entity: 实体名称
            max_depth: 最大搜索深度
            
        Returns:
            相关实体列表
        """
        related_nodes = self.retriever.find_related_entities(
            entity_name=entity,
            max_depth=max_depth,
        )
        
        result_list = []
        for node in related_nodes:
            result_list.append({
                "node_id": node.id,
                "node_name": node.name,
                "node_type": node.type.value,
                "properties": node.properties,
            })
        
        return result_list
    
    def _search_paths(
        self,
        source_entity: str,
        target_entity: str,
        max_depth: int,
    ) -> List[Dict[str, Any]]:
        """路径查找
        
        Args:
            source_entity: 源实体名称
            target_entity: 目标实体名称
            max_depth: 最大搜索深度
            
        Returns:
            路径列表
        """
        paths = self.retriever.find_paths(
            source_entity=source_entity,
            target_entity=target_entity,
            max_depth=max_depth,
        )
        
        result_list = []
        for path in paths:
            path_dict = {
                "length": path.length,
                "nodes": [
                    {
                        "id": node.id,
                        "name": node.name,
                        "type": node.type.value,
                    }
                    for node in path.nodes
                ],
                "edges": [
                    {
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "type": edge.type.value,
                        "properties": edge.properties,
                    }
                    for edge in path.edges
                ],
            }
            result_list.append(path_dict)
        
        return result_list
    
    def get_info(self) -> Dict[str, Any]:
        """获取工具信息
        
        Returns:
            工具信息字典
        """
        info = super().get_info()
        info["retriever_type"] = "graph"
        info["retriever_stats"] = self.retriever.get_stats()
        return info


if __name__ == "__main__":
    from src.storage.graph_store import LocalGraphStore, Node, NodeType, Edge, EdgeType
    from src.retrievers.graph_retriever import GraphRetriever
    
    print("=" * 50)
    print("测试 Graph Tool 模块")
    print("=" * 50)
    
    # 创建图存储
    graph_store = LocalGraphStore()
    nodes = [
        Node(id="ai", type=NodeType.ENTITY, name="人工智能"),
        Node(id="ml", type=NodeType.ENTITY, name="机器学习"),
        Node(id="dl", type=NodeType.ENTITY, name="深度学习"),
    ]
    for node in nodes:
        graph_store.add_node(node)
    
    edges = [
        Edge(id="e1", source_id="ai", target_id="ml", type=EdgeType.RELATED_TO),
        Edge(id="e2", source_id="ml", target_id="dl", type=EdgeType.RELATED_TO),
    ]
    for edge in edges:
        graph_store.add_edge(edge)
    
    retriever = GraphRetriever(graph_store=graph_store)
    print(f"✓ 创建图检索器: nodes={graph_store.count_nodes()}")
    
    # 测试 1: 创建工具
    tool = GraphSearchTool(retriever=retriever)
    print(f"✓ 创建工具: name={tool.name}")
    
    # 测试 2: 实体检索
    result = tool.run(entity="人工智能", search_type="entity", top_k=3)
    print(f"✓ 实体检索: success={result.success}")
    if result.success:
        print(f"  返回结果数: {len(result.result)}")
    
    # 测试 3: 邻居检索
    result2 = tool.run(entity="ai", search_type="neighbors", top_k=5)
    print(f"✓ 邻居检索: success={result2.success}")
    
    # 测试 4: 路径查找
    result3 = tool.run(entity="ai", search_type="paths", target_entity="dl")
    print(f"✓ 路径查找: success={result3.success}")
    
    # 测试 5: 工具信息
    info = tool.get_info()
    print(f"✓ 工具信息: retriever_type={info['retriever_type']}")
    
    print("\n所有测试通过!")
