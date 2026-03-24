# -*- coding: utf-8 -*-
"""RAG Chain 基础模块

定义 RAG Chain 的基础类和数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llms.base_client import BaseLLMClient, LLMResponse, Message
from src.llms import create_client, create_embedding_client
from src.retrievers.base_retriever import RetrieverBase, SearchResult
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.context_formatter import format_search_results

logger = get_logger(__name__)


@dataclass
class ChainResult:
    """RAG Chain 执行结果
    
    Attributes:
        query: 原始查询
        answer: 生成的答案
        sources: 引用的来源文档
        metadata: 额外元数据
        retrieval_results: 检索结果列表
        llm_response: LLM 原始响应
    """
    query: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_results: List[SearchResult] = field(default_factory=list)
    llm_response: Optional[LLMResponse] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            结果字典
        """
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "metadata": self.metadata,
            "retrieval_results": [r.to_dict() for r in self.retrieval_results],
            "llm_response": self.llm_response.to_dict() if self.llm_response else None,
        }


class RAGChainBase(ABC):
    """RAG Chain 抽象基类
    
    定义 RAG Chain 的统一接口，所有 Chain 实现都应继承此类。
    
    Attributes:
        llm_client: LLM 客户端实例
        retriever: 检索器实例
        config: 配置实例
    """
    
    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        retriever: Optional[RetrieverBase] = None,
        **kwargs: Any,
    ):
        """初始化 RAG Chain
        
        Args:
            llm_client: LLM 客户端实例（可选，默认从配置创建）
            retriever: 检索器实例（可选，由子类实现）
            **kwargs: 额外参数
        """
        self.config = get_config()
        self.llm_client = llm_client or self._create_default_llm_client()
        self.retriever = retriever
        self.chain_config = kwargs
        
        logger.info(f"初始化 {self.__class__.__name__}")
    
    def _create_default_llm_client(self) -> BaseLLMClient:
        """创建默认的 LLM 客户端
        
        Returns:
            LLM 客户端实例
        """
        return create_client(
            provider="deepseek",
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """执行检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 额外参数
            
        Returns:
            检索结果列表
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        query: str,
        context: List[SearchResult],
        **kwargs: Any,
    ) -> LLMResponse:
        """生成答案
        
        Args:
            query: 查询文本
            context: 检索上下文
            **kwargs: 额外参数
            
        Returns:
            LLM 响应
        """
        pass
    
    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> ChainResult:
        """执行完整的 RAG 流程
        
        Args:
            query: 查询文本
            top_k: 检索结果数量
            **kwargs: 额外参数
            
        Returns:
            Chain 执行结果
        """
        logger.info(f"开始执行 RAG Chain: query='{query[:50]}...'")
        
        retrieval_results = self.retrieve(query, top_k, **kwargs)
        
        llm_response = self.generate(query, retrieval_results, **kwargs)
        
        sources = self._extract_sources(retrieval_results)
        
        result = ChainResult(
            query=query,
            answer=llm_response.content,
            sources=sources,
            retrieval_results=retrieval_results,
            llm_response=llm_response,
            metadata={
                "chain_type": self.__class__.__name__,
                "retrieval_count": len(retrieval_results),
            },
        )
        
        logger.info(f"RAG Chain 执行完成: answer_length={len(result.answer)}")
        
        return result
    
    def _extract_sources(
        self,
        results: List[SearchResult],
    ) -> List[Dict[str, Any]]:
        """从检索结果中提取来源信息
        
        Args:
            results: 检索结果列表
            
        Returns:
            来源信息列表
        """
        sources = []
        for result in results:
            source = {
                "doc_id": result.doc_id,
                "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                "score": result.score,
            }
            if "source" in result.metadata:
                source["source"] = result.metadata["source"]
            sources.append(source)
        return sources
    
    def _build_context(
        self,
        results: List[SearchResult],
        max_length: Optional[int] = None,
    ) -> str:
        """构建上下文字符串
        
        Args:
            results: 检索结果列表
            max_length: 最大上下文长度
            
        Returns:
            上下文字符串
        """
        max_len = max_length or 4000
        return format_search_results(results, max_len)
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> List[Message]:
        """构建提示消息
        
        Args:
            query: 查询文本
            context: 上下文
            system_prompt: 系统提示（可选）
            
        Returns:
            消息列表
        """
        default_system = """你是一个专业的问答助手。请根据提供的参考文档回答用户问题。
要求：
1. 回答要准确、简洁、有条理
2. 如果参考文档中没有相关信息，请明确说明
3. 引用信息时请注明来源文档编号"""

        system = system_prompt or default_system
        
        user_prompt = f"""参考文档：
{context}

用户问题：{query}

请根据参考文档回答问题："""
        
        return [
            Message(role="system", content=system),
            Message(role="user", content=user_prompt),
        ]
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"llm_client={self.llm_client.__class__.__name__}, "
            f"retriever={self.retriever.__class__.__name__ if self.retriever else 'None'})"
        )
