# -*- coding: utf-8 -*-
"""上下文格式化工具

提供统一的上下文格式化功能，避免各模块重复实现。
"""

from typing import Any, Dict, List, Optional


class ContextFormatter:
    """上下文格式化器
    
    提供多种上下文格式化方法，支持不同场景的需求。
    """
    
    @staticmethod
    def format_sources(
        sources: List[Dict[str, Any]],
        max_length: int = 4000,
        include_score: bool = True,
    ) -> str:
        """格式化来源列表为上下文字符串
        
        Args:
            sources: 来源列表
            max_length: 最大长度
            include_score: 是否包含分数
            
        Returns:
            格式化后的上下文字符串
        """
        context_parts = []
        total_length = 0
        
        for i, source in enumerate(sources, start=1):
            content = source.get("content", "")
            
            if total_length + len(content) > max_length:
                remaining = max_length - total_length
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break
            
            if include_score and "score" in source:
                context_parts.append(f"[文档 {i}] (相关性: {source['score']:.2f})\n{content}")
            else:
                context_parts.append(f"[文档 {i}]\n{content}")
            
            total_length += len(content)
        
        return "\n\n".join(context_parts)
    
    @staticmethod
    def format_search_results(
        results: List[Any],
        max_length: int = 4000,
    ) -> str:
        """格式化 SearchResult 对象列表
        
        Args:
            results: SearchResult 对象列表
            max_length: 最大长度
            
        Returns:
            格式化后的上下文字符串
        """
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(results, start=1):
            content = result.content
            
            if total_length + len(content) > max_length:
                remaining = max_length - total_length
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break
            
            context_parts.append(f"[文档 {i}]\n{content}")
            total_length += len(content)
        
        return "\n\n".join(context_parts)
    
    @staticmethod
    def format_graph_context(
        results: List[Any],
    ) -> str:
        """格式化图谱检索结果
        
        Args:
            results: 检索结果列表
            
        Returns:
            图谱上下文字符串
        """
        context_parts = ["【知识图谱检索结果】\n"]
        
        for i, result in enumerate(results, start=1):
            context_parts.append(f"\n--- 实体 {i} ---")
            context_parts.append(result.content)
            
            if hasattr(result, "metadata") and result.metadata.get("neighbor_count"):
                context_parts.append(f"关联实体数: {result.metadata['neighbor_count']}")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def format_agent_sources(
        sources: List[Any],
        max_length: int = 4000,
    ) -> str:
        """格式化 Agent 来源列表
        
        Args:
            sources: Source 对象列表
            max_length: 最大长度
            
        Returns:
            格式化后的字符串
        """
        formatted = []
        total_length = 0
        
        for i, source in enumerate(sources):
            content = source.content.strip()
            
            if total_length + len(content) > max_length:
                break
            
            formatted.append(f"[来源 {i+1}]\n{content}\n")
            total_length += len(content)
        
        return "\n".join(formatted)
    
    @staticmethod
    def format_with_metadata(
        sources: List[Any],
        max_length: int = 6000,
    ) -> str:
        """格式化带元数据的来源
        
        Args:
            sources: 来源列表
            max_length: 最大长度
            
        Returns:
            格式化后的上下文
        """
        formatted = []
        
        for i, source in enumerate(sources):
            score = getattr(source, "score", 0.0)
            source_type = getattr(source, "source_type", "unknown")
            content = getattr(source, "content", "")
            
            formatted.append(
                f"[来源 {i+1}] (相关性: {score:.2f}, 类型: {source_type})\n"
                f"{content}\n"
            )
        
        return "\n".join(formatted)


format_sources = ContextFormatter.format_sources
format_search_results = ContextFormatter.format_search_results
format_graph_context = ContextFormatter.format_graph_context
format_agent_sources = ContextFormatter.format_agent_sources
format_with_metadata = ContextFormatter.format_with_metadata
