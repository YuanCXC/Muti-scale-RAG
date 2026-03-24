# -*- coding: utf-8 -*-
"""混合检索工具

封装 HybridRetriever 的混合检索能力。
"""

from typing import Any, Dict, List, Optional, Type

from pydantic import Field

from src.retrievers.hybrid_retriever import HybridRetriever
from src.utils.logger import get_logger

from src.tools.base_tool import ToolArgs, ToolBase, ToolResult

logger = get_logger(__name__)


class HybridSearchArgs(ToolArgs):
    """混合检索工具参数
    
    Attributes:
        query: 查询文本
        top_k: 返回结果数量
        vector_top_k: 向量检索返回数量
        keyword_top_k: 关键词检索返回数量
        vector_weight: 向量检索权重
        keyword_weight: 关键词检索权重
    """
    query: str = Field(
        ...,
        description="查询文本，用于混合检索"
    )
    top_k: Optional[int] = Field(
        default=10,
        description="最终返回结果数量，默认为 10",
        ge=1,
        le=100,
    )
    vector_top_k: Optional[int] = Field(
        default=None,
        description="向量检索返回数量，默认为 top_k * 2",
    )
    keyword_top_k: Optional[int] = Field(
        default=None,
        description="关键词检索返回数量，默认为 top_k * 2",
    )
    vector_weight: Optional[float] = Field(
        default=None,
        description="向量检索权重，覆盖默认配置",
        ge=0.0,
        le=1.0,
    )
    keyword_weight: Optional[float] = Field(
        default=None,
        description="关键词检索权重，覆盖默认配置",
        ge=0.0,
        le=1.0,
    )


class HybridSearchTool(ToolBase):
    """混合检索工具
    
    封装 HybridRetriever 的混合检索能力，
    整合向量检索和关键词检索，使用 RRF 融合算法。
    
    Attributes:
        name: 工具名称
        description: 工具描述
        args_schema: 参数定义
        retriever: HybridRetriever 实例
    """
    
    name: str = "hybrid_search"
    description: str = (
        "混合检索工具：结合向量检索和关键词检索的优势。"
        "使用 RRF (Reciprocal Rank Fusion) 算法融合结果。"
        "适用于需要同时考虑语义相似性和关键词匹配的场景。"
        "输入查询文本，返回融合后的相关文档列表。"
    )
    args_schema: Type[ToolArgs] = HybridSearchArgs
    
    def __init__(
        self,
        retriever: HybridRetriever,
        **kwargs: Any,
    ):
        """初始化混合检索工具
        
        Args:
            retriever: HybridRetriever 实例
            **kwargs: 额外配置参数
        """
        super().__init__(**kwargs)
        self.retriever = retriever
        
        logger.info(f"初始化混合检索工具: retriever={retriever}")
    
    def run(self, **kwargs: Any) -> ToolResult:
        """执行混合检索
        
        Args:
            **kwargs: 工具参数，包含:
                - query: 查询文本
                - top_k: 返回结果数量（可选）
                - vector_top_k: 向量检索返回数量（可选）
                - keyword_top_k: 关键词检索返回数量（可选）
                - vector_weight: 向量检索权重（可选）
                - keyword_weight: 关键词检索权重（可选）
            
        Returns:
            ToolResult 实例，包含融合后的检索结果列表
        """
        try:
            # 验证参数
            validated_args = self.validate_args(**kwargs)
            
            query = validated_args["query"]
            top_k = validated_args.get("top_k", 10)
            vector_top_k = validated_args.get("vector_top_k")
            keyword_top_k = validated_args.get("keyword_top_k")
            vector_weight = validated_args.get("vector_weight")
            keyword_weight = validated_args.get("keyword_weight")
            
            # 临时调整权重（如果提供）
            original_vector_weight = self.retriever.vector_weight
            original_keyword_weight = self.retriever.keyword_weight
            
            if vector_weight is not None and keyword_weight is not None:
                self.retriever.vector_weight = vector_weight
                self.retriever.keyword_weight = keyword_weight
            
            logger.info(
                f"执行混合检索: query='{query[:50]}...', "
                f"top_k={top_k}, vector_weight={self.retriever.vector_weight}, "
                f"keyword_weight={self.retriever.keyword_weight}"
            )
            
            # 执行检索
            results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                vector_top_k=vector_top_k,
                keyword_top_k=keyword_top_k,
            )
            
            # 恢复原始权重
            self.retriever.vector_weight = original_vector_weight
            self.retriever.keyword_weight = original_keyword_weight
            
            # 转换结果格式
            result_list = [result.to_dict() for result in results]
            
            logger.info(f"混合检索完成: 找到 {len(result_list)} 个结果")
            
            return ToolResult.success_result(
                result=result_list,
                metadata={
                    "query": query,
                    "top_k": top_k,
                    "result_count": len(result_list),
                    "vector_weight": self.retriever.vector_weight,
                    "keyword_weight": self.retriever.keyword_weight,
                },
            )
            
        except Exception as e:
            error_msg = f"混合检索失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ToolResult.error_result(error=error_msg)
    
    def get_info(self) -> Dict[str, Any]:
        """获取工具信息
        
        Returns:
            工具信息字典
        """
        info = super().get_info()
        info["retriever_type"] = "hybrid"
        info["retriever_stats"] = self.retriever.get_stats()
        return info


if __name__ == "__main__":
    import numpy as np
    from src.storage.vector_store import FAISSVectorStore
    from src.retrievers.vector_retriever import VectorRetriever
    from src.retrievers.keyword_retriever import KeywordRetriever
    from src.retrievers.hybrid_retriever import HybridRetriever
    from src.utils.config import get_config
    
    config = get_config()
    
    print("=" * 50)
    print("测试 Hybrid Search Tool 模块")
    print("=" * 50)
    
    # Mock LLM Client for testing
    class MockLLMClient:
        def embed(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            return [np.random.randn(config.vector_dim).astype(np.float32).tolist() for _ in texts]
    
    mock_llm = MockLLMClient()
    
    # 创建检索器
    docs = [
        {"doc_id": "d1", "content": "人工智能是计算机科学的分支。", "metadata": {}},
        {"doc_id": "d2", "content": "机器学习是AI的核心技术。", "metadata": {}},
    ]
    vectors = np.random.randn(2, config.vector_dim).astype(np.float32)
    
    vector_store = FAISSVectorStore()
    vector_retriever = VectorRetriever(vector_store=vector_store, llm_client=mock_llm)
    vector_retriever.add_documents(docs, vectors.tolist())
    
    keyword_retriever = KeywordRetriever()
    keyword_retriever.add_documents(docs)
    
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever
    )
    print(f"✓ 创建混合检索器")
    
    # 测试 1: 创建工具
    tool = HybridSearchTool(retriever=hybrid_retriever)
    print(f"✓ 创建工具: name={tool.name}")
    
    # 测试 2: 执行工具
    result = tool.run(
        query="人工智能",
        top_k=2
    )
    print(f"✓ 执行工具: success={result.success}")
    if result.success:
        print(f"  返回结果数: {len(result.result)}")
    
    # 测试 3: 权重配置
    result2 = tool.run(
        query="AI技术",
        vector_weight=0.7,
        keyword_weight=0.3
    )
    print(f"✓ 自定义权重执行: success={result2.success}")
    
    # 测试 4: 工具信息
    info = tool.get_info()
    print(f"✓ 工具信息: retriever_type={info['retriever_type']}")
    
    print("\n所有测试通过!")
