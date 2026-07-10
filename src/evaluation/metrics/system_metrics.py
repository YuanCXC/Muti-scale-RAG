# -*- coding: utf-8 -*-
"""系统指标模块

提供 RAG 系统运行时的评估指标，包括上下文长度、token成本、证据条数等。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SystemResult:
    """系统评估结果
    
    Attributes:
        avg_context_length: 平均上下文长度（字符数）
        avg_token_cost: 平均 token 成本
        avg_evidence_count: 平均证据条数
        avg_parent_recall_count: 平均父级回升次数
        avg_graph_expansion_count: 平均图扩展节点次数
        total_samples: 总样本数
    """
    avg_context_length: float = 0.0
    avg_token_cost: float = 0.0
    avg_evidence_count: float = 0.0
    avg_parent_recall_count: float = 0.0
    avg_graph_expansion_count: float = 0.0
    total_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "avg_context_length": self.avg_context_length,
            "avg_token_cost": self.avg_token_cost,
            "avg_evidence_count": self.avg_evidence_count,
            "avg_parent_recall_count": self.avg_parent_recall_count,
            "avg_graph_expansion_count": self.avg_graph_expansion_count,
            "total_samples": self.total_samples,
        }


@dataclass
class SampleMetrics:
    """单个样本的系统指标
    
    Attributes:
        context_length: 上下文长度（字符数）
        token_cost: token 成本
        evidence_count: 证据条数
        parent_recall_count: 父级回升次数
        graph_expansion_count: 图扩展节点次数
    """
    context_length: int = 0
    token_cost: int = 0
    evidence_count: int = 0
    parent_recall_count: int = 0
    graph_expansion_count: int = 0


class SystemMetrics:
    """系统指标计算器
    
    提供系统运行时指标的统一计算接口。
    """
    
    def __init__(self):
        """初始化系统指标计算器"""
        self.samples: List[SampleMetrics] = []
    
    def add_sample(
        self,
        context_length: int = 0,
        token_cost: int = 0,
        evidence_count: int = 0,
        parent_recall_count: int = 0,
        graph_expansion_count: int = 0,
    ) -> None:
        """添加样本指标
        
        Args:
            context_length: 上下文长度（字符数）
            token_cost: token 成本
            evidence_count: 证据条数
            parent_recall_count: 父级回升次数
            graph_expansion_count: 图扩展节点次数
        """
        sample = SampleMetrics(
            context_length=context_length,
            token_cost=token_cost,
            evidence_count=evidence_count,
            parent_recall_count=parent_recall_count,
            graph_expansion_count=graph_expansion_count,
        )
        self.samples.append(sample)
    
    def add_sample_from_dict(self, data: Dict[str, Any]) -> None:
        """从字典添加样本指标
        
        Args:
            data: 包含指标的字典
        """
        self.add_sample(
            context_length=data.get("context_length", 0),
            token_cost=data.get("token_cost", 0),
            evidence_count=data.get("evidence_count", 0),
            parent_recall_count=data.get("parent_recall_count", 0),
            graph_expansion_count=data.get("graph_expansion_count", 0),
        )
    
    def compute(self) -> SystemResult:
        """计算所有系统指标的平均值
        
        Returns:
            SystemResult 实例
        """
        if not self.samples:
            return SystemResult()
        
        total_context_length = sum(s.context_length for s in self.samples)
        total_token_cost = sum(s.token_cost for s in self.samples)
        total_evidence_count = sum(s.evidence_count for s in self.samples)
        total_parent_recall_count = sum(s.parent_recall_count for s in self.samples)
        total_graph_expansion_count = sum(s.graph_expansion_count for s in self.samples)
        
        n = len(self.samples)
        
        return SystemResult(
            avg_context_length=total_context_length / n,
            avg_token_cost=total_token_cost / n,
            avg_evidence_count=total_evidence_count / n,
            avg_parent_recall_count=total_parent_recall_count / n,
            avg_graph_expansion_count=total_graph_expansion_count / n,
            total_samples=n,
        )
    
    def reset(self) -> None:
        """重置所有样本"""
        self.samples = []
    
    @staticmethod
    def calculate_token_count(text: str) -> int:
        """估算文本的 token 数量
        
        使用简单的启发式方法估算 token 数量。
        中文约 1.5 字符/token，英文约 4 字符/token。
        
        Args:
            text: 输入文本
            
        Returns:
            估算的 token 数量
        """
        if not text:
            return 0
        
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        
        chinese_tokens = chinese_chars / 1.5
        other_tokens = other_chars / 4
        
        return int(chinese_tokens + other_tokens)
    
    @staticmethod
    def calculate_context_length(context: str) -> int:
        """计算上下文长度
        
        Args:
            context: 上下文文本
            
        Returns:
            上下文长度（字符数）
        """
        return len(context) if context else 0
    
    @staticmethod
    def calculate_evidence_count(sources: List[Dict[str, Any]]) -> int:
        """计算证据条数
        
        Args:
            sources: 检索到的文档来源列表
            
        Returns:
            证据条数
        """
        return len(sources) if sources else 0


if __name__ == "__main__":
    print("=" * 50)
    print("测试系统指标")
    print("=" * 50)
    
    metrics = SystemMetrics()
    
    metrics.add_sample(
        context_length=1500,
        token_cost=500,
        evidence_count=5,
        parent_recall_count=2,
        graph_expansion_count=3,
    )
    
    metrics.add_sample(
        context_length=2000,
        token_cost=600,
        evidence_count=4,
        parent_recall_count=1,
        graph_expansion_count=2,
    )
    
    result = metrics.compute()
    
    print(f"✓ 平均上下文长度: {result.avg_context_length:.2f}")
    print(f"✓ 平均 token 成本: {result.avg_token_cost:.2f}")
    print(f"✓ 平均证据条数: {result.avg_evidence_count:.2f}")
    print(f"✓ 平均父级回升次数: {result.avg_parent_recall_count:.2f}")
    print(f"✓ 平均图扩展节点次数: {result.avg_graph_expansion_count:.2f}")
    print(f"✓ 总样本数: {result.total_samples}")
    
    test_text = "这是一个测试文本，用于测试 token 计算功能。This is a test text for token calculation."
    token_count = SystemMetrics.calculate_token_count(test_text)
    print(f"\n✓ Token 估算测试: '{test_text[:30]}...' -> {token_count} tokens")
    
    print("\n所有测试通过!")
