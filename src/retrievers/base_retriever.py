# -*- coding: utf-8 -*-
"""基础检索器抽象类

定义检索器的统一接口，支持不同的检索策略实现。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.config import get_config


@dataclass
class SearchResult:
    """检索结果数据类
    
    存储单个检索结果的信息。
    
    Attributes:
        doc_id: 文档ID
        content: 文档内容
        score: 相关性分数
        metadata: 额外元数据
    """
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            结果字典
        """
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """从字典创建实例
        
        Args:
            data: 结果字典
            
        Returns:
            SearchResult 实例
        """
        return cls(
            doc_id=data["doc_id"],
            content=data["content"],
            score=data["score"],
            metadata=data.get("metadata", {}),
        )


class RetrieverBase(ABC):
    """检索器抽象基类
    
    定义检索器的统一接口，所有检索器实现都应继承此类。
    
    Attributes:
        top_k: 默认返回结果数量
        score_threshold: 分数阈值，低于此值的结果将被过滤
    """
    
    def __init__(self, **kwargs: Any):
        """初始化检索器
        
        Args:
            **kwargs: 额外参数
        """
        config = get_config()
        
        self.top_k = config.top_k
        self.score_threshold = config.similarity_threshold
        self.config = kwargs
    
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
            top_k: 返回结果数量（可选，覆盖默认值）
            **kwargs: 额外参数
            
        Returns:
            检索结果列表，按相关性分数降序排列
        """
        pass
    
    def _get_top_k(self, top_k: Optional[int]) -> int:
        """获取 top_k 参数
        
        Args:
            top_k: 传入的 top_k 参数
            
        Returns:
            实际使用的 top_k 值
        """
        return top_k if top_k is not None else self.top_k
    
    def _filter_by_threshold(
        self,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """根据分数阈值过滤结果
        
        Args:
            results: 检索结果列表
            
        Returns:
            过滤后的结果列表
        """
        if self.score_threshold is None:
            return results
        
        return [
            result for result in results
            if result.score >= self.score_threshold
        ]
    
    def _deduplicate(
        self,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """去重，保留分数最高的结果
        
        Args:
            results: 检索结果列表
            
        Returns:
            去重后的结果列表
        """
        seen_docs: Dict[str, SearchResult] = {}
        
        for result in results:
            if result.doc_id not in seen_docs:
                seen_docs[result.doc_id] = result
            elif result.score > seen_docs[result.doc_id].score:
                seen_docs[result.doc_id] = result
        
        return list(seen_docs.values())
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"top_k={self.top_k}, "
            f"score_threshold={self.score_threshold})"
        )
