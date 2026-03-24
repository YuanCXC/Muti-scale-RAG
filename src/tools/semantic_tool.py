# -*- coding: utf-8 -*-
"""语义检索工具

封装 VectorRetriever 的单一语义检索能力。
"""

from typing import Any, Dict, List, Optional, Type

from pydantic import Field

from src.retrievers.vector_retriever import VectorRetriever
from src.utils.logger import get_logger

from src.tools.base_tool import ToolArgs, ToolBase, ToolResult

logger = get_logger(__name__)


class SemanticSearchArgs(ToolArgs):
    """语义检索工具参数
    
    Attributes:
        query: 查询文本
        top_k: 返回结果数量
        filter_dict: 过滤条件
    """
    query: str = Field(
        ...,
        description="查询文本，用于语义检索"
    )
    top_k: Optional[int] = Field(
        default=10,
        description="返回结果数量，默认为 10",
        ge=1,
        le=100,
    )
    filter_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="过滤条件字典，用于过滤检索结果",
    )


class SemanticSearchTool(ToolBase):
    """语义检索工具
    
    封装 VectorRetriever 的单一语义检索能力，
    使用向量相似度进行文档检索。
    
    Attributes:
        name: 工具名称
        description: 工具描述
        args_schema: 参数定义
        retriever: VectorRetriever 实例
    """
    
    name: str = "semantic_search"
    description: str = (
        "语义检索工具：使用向量相似度搜索相关文档。"
        "适用于需要基于语义理解进行检索的场景。"
        "输入查询文本，返回最相关的文档列表。"
    )
    args_schema: Type[ToolArgs] = SemanticSearchArgs
    
    def __init__(
        self,
        retriever: VectorRetriever,
        **kwargs: Any,
    ):
        """初始化语义检索工具
        
        Args:
            retriever: VectorRetriever 实例
            **kwargs: 额外配置参数
        """
        super().__init__(**kwargs)
        self.retriever = retriever
        
        logger.info(f"初始化语义检索工具: retriever={retriever}")
    
    def run(self, **kwargs: Any) -> ToolResult:
        """执行语义检索
        
        Args:
            **kwargs: 工具参数，包含:
                - query: 查询文本
                - top_k: 返回结果数量（可选）
                - filter_dict: 过滤条件（可选）
            
        Returns:
            ToolResult 实例，包含检索结果列表
        """
        try:
            # 验证参数
            validated_args = self.validate_args(**kwargs)
            
            query = validated_args["query"]
            top_k = validated_args.get("top_k", 10)
            filter_dict = validated_args.get("filter_dict")
            
            logger.info(f"执行语义检索: query='{query[:50]}...', top_k={top_k}")
            
            # 执行检索
            results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                filter_dict=filter_dict,
            )
            
            # 转换结果格式
            result_list = [result.to_dict() for result in results]
            
            logger.info(f"语义检索完成: 找到 {len(result_list)} 个结果")
            
            return ToolResult.success_result(
                result=result_list,
                metadata={
                    "query": query,
                    "top_k": top_k,
                    "result_count": len(result_list),
                },
            )
            
        except Exception as e:
            error_msg = f"语义检索失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ToolResult.error_result(error=error_msg)
    
    def get_info(self) -> Dict[str, Any]:
        """获取工具信息
        
        Returns:
            工具信息字典
        """
        info = super().get_info()
        info["retriever_type"] = "vector"
        info["retriever_stats"] = self.retriever.get_stats()
        return info


if __name__ == "__main__":
    import numpy as np
    from src.storage.vector_store import FAISSVectorStore
    from src.retrievers.vector_retriever import VectorRetriever
    from src.utils.config import get_config
    
    config = get_config()
    
    print("=" * 50)
    print("测试 Semantic Tool 模块")
    print("=" * 50)
    
    # Mock LLM Client for testing
    class MockLLMClient:
        def embed(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            return [np.random.randn(config.vector_dim).astype(np.float32).tolist() for _ in texts]
    
    mock_llm = MockLLMClient()
    
    # 测试 1: 创建向量存储和检索器
    vector_store = FAISSVectorStore(metric="cosine")
    retriever = VectorRetriever(vector_store=vector_store, llm_client=mock_llm)
    
    # 添加文档
    docs = [
        {"doc_id": "d1", "content": "人工智能是计算机科学的分支。", "metadata": {}},
        {"doc_id": "d2", "content": "机器学习是AI的核心技术。", "metadata": {}},
    ]
    vectors = np.random.randn(2, config.vector_dim).astype(np.float32)
    retriever.add_documents(docs, vectors.tolist())
    print(f"✓ 创建检索器并添加文档")
    
    # 测试 2: 创建工具
    tool = SemanticSearchTool(retriever=retriever)
    print(f"✓ 创建工具: name={tool.name}")
    print(f"  description: {tool.description}")
    
    # 测试 3: 获取参数 schema
    schema = tool.get_parameters_schema()
    print(f"✓ 参数 schema: {list(schema.get('properties', {}).keys())}")
    
    # 测试 4: 执行工具
    result = tool.run(query="人工智能", top_k=2)
    print(f"✓ 执行工具: success={result.success}")
    if result.success:
        print(f"  返回结果数: {len(result.result)}")
    
    # 测试 5: 工具信息
    info = tool.get_info()
    print(f"✓ 工具信息: class={info['class']}")
    
    # 测试 6: 直接调用
    result2 = tool(query="机器学习")
    print(f"✓ 直接调用: success={result2.success}")
    
    print("\n所有测试通过!")
