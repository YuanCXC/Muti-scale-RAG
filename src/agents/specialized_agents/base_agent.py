# -*- coding: utf-8 -*-
"""基础 Agent 抽象类

定义专用 Agent 的统一接口和数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.context_formatter import format_agent_sources

logger = get_logger(__name__)


class IntentType(str, Enum):
    """意图类型枚举"""
    FACTUAL = "FACTUAL"          # 事实型问题
    EXPLANATORY = "EXPLANATORY"  # 解释型问题
    REASONING = "REASONING"      # 推理型问题
    UNKNOWN = "UNKNOWN"          # 未知类型


@dataclass
class Source:
    """信息来源数据类
    
    Attributes:
        content: 来源内容
        source_id: 来源标识
        source_type: 来源类型（document, graph, web等）
        score: 相关性分数
        metadata: 额外元数据
    """
    content: str
    source_id: Optional[str] = None
    source_type: str = "document"
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "content": self.content,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class AgentResponse:
    """Agent 响应数据类
    
    Attributes:
        answer: 生成的答案
        sources: 信息来源列表
        confidence: 置信度 (0.0-1.0)
        intent_type: 意图类型
        reasoning_chain: 推理链（用于推理型问题）
        metadata: 额外元数据
        success: 是否成功
        error: 错误信息（如果失败）
    """
    answer: str
    sources: List[Source] = field(default_factory=list)
    confidence: float = 0.0
    intent_type: IntentType = IntentType.UNKNOWN
    reasoning_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence,
            "intent_type": self.intent_type.value,
            "reasoning_chain": self.reasoning_chain,
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error,
        }
    
    @classmethod
    def error_response(cls, error: str, intent_type: IntentType = IntentType.UNKNOWN) -> "AgentResponse":
        """创建错误响应
        
        Args:
            error: 错误信息
            intent_type: 意图类型
            
        Returns:
            AgentResponse 实例
        """
        return cls(
            answer="",
            confidence=0.0,
            intent_type=intent_type,
            success=False,
            error=error,
        )


@dataclass
class AgentContext:
    """Agent 上下文数据类
    
    Attributes:
        query: 用户查询
        conversation_history: 对话历史
        retrieved_docs: 已检索的文档
        entities: 提取的实体
        metadata: 额外元数据
    """
    query: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpecializedAgentBase(ABC):
    """专用 Agent 抽象基类
    
    所有专用 Agent 都应继承此类并实现 process 方法。
    
    Attributes:
        name: Agent 名称
        description: Agent 描述
        intent_type: 处理的意图类型
        llm_client: LLM 客户端
        tools: 可用工具列表
    """
    
    name: str = "base_agent"
    description: str = "基础 Agent"
    intent_type: IntentType = IntentType.UNKNOWN
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        tools: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """初始化 Agent
        
        Args:
            llm_client: LLM 客户端实例
            tools: 可用工具字典
            **kwargs: 其他配置参数
        """
        config = get_config()
        
        self.llm_client = llm_client
        self.tools = tools or {}
        self.max_iterations = config.agent_max_iterations
        self.timeout = config.agent_timeout
        self.config = kwargs
        
        logger.info(f"初始化 Agent: {self.name} (意图类型: {self.intent_type.value})")
    
    @abstractmethod
    def process(self, query: str, context: AgentContext) -> AgentResponse:
        """处理查询并生成响应
        
        Args:
            query: 用户查询
            context: Agent 上下文
            
        Returns:
            AgentResponse 实例
        """
        pass
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Source]:
        """执行检索操作
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            Source 列表
        """
        sources = []
        
        # 使用语义检索工具
        if "semantic" in self.tools:
            try:
                result = self.tools["semantic"].run(query=query, top_k=top_k)
                if result.success:
                    for item in result.result:
                        sources.append(Source(
                            content=item.get("content", ""),
                            source_id=item.get("id"),
                            source_type="semantic",
                            score=item.get("score", 0.0),
                            metadata=item.get("metadata", {}),
                        ))
            except Exception as e:
                logger.warning(f"语义检索失败: {e}")
        
        # 使用混合检索工具
        if "hybrid" in self.tools and not sources:
            try:
                result = self.tools["hybrid"].run(query=query, top_k=top_k)
                if result.success:
                    for item in result.result:
                        sources.append(Source(
                            content=item.get("content", ""),
                            source_id=item.get("id"),
                            source_type="hybrid",
                            score=item.get("score", 0.0),
                            metadata=item.get("metadata", {}),
                        ))
            except Exception as e:
                logger.warning(f"混合检索失败: {e}")
        
        # 使用图谱检索工具
        if "graph" in self.tools:
            try:
                result = self.tools["graph"].run(query=query, top_k=top_k)
                if result.success:
                    for item in result.result:
                        sources.append(Source(
                            content=item.get("content", ""),
                            source_id=item.get("id"),
                            source_type="graph",
                            score=item.get("score", 0.0),
                            metadata=item.get("metadata", {}),
                        ))
            except Exception as e:
                logger.warning(f"图谱检索失败: {e}")
        
        logger.info(f"检索完成，获取 {len(sources)} 个来源")
        return sources
    
    def generate_answer(
        self,
        query: str,
        context: str,
        temperature: float = 0.7,
    ) -> str:
        """使用 LLM 生成答案
        
        Args:
            query: 用户查询
            context: 上下文信息
            temperature: 温度参数
            
        Returns:
            生成的答案
        """
        if not self.llm_client:
            raise ValueError("LLM 客户端未配置")
        
        from src.llms import Message
        
        messages = [
            Message.system(self._get_system_prompt()),
            Message.user(self._format_user_prompt(query, context)),
        ]
        
        response = self.llm_client.generate(messages, temperature=temperature)
        return response.content
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词
        
        Returns:
            系统提示词字符串
        """
        return "你是一个专业的知识问答助手。请根据提供的信息准确回答问题。"
    
    def _format_user_prompt(self, query: str, context: str) -> str:
        """格式化用户提示词
        
        Args:
            query: 用户查询
            context: 上下文信息
            
        Returns:
            格式化后的提示词
        """
        return f"""请根据以下信息回答问题。

## 相关信息：
{context}

## 问题：
{query}

## 请回答："""
    
    def format_sources(self, sources: List[Source], max_length: int = 4000) -> str:
        """格式化信息来源
        
        Args:
            sources: 来源列表
            max_length: 最大长度
            
        Returns:
            格式化后的字符串
        """
        return format_agent_sources(sources, max_length)
    
    def calculate_confidence(self, sources: List[Source]) -> float:
        """计算置信度
        
        Args:
            sources: 信息来源列表
            
        Returns:
            置信度分数 (0.0-1.0)
        """
        if not sources:
            return 0.0
        
        # 基于来源数量和相关性分数计算
        avg_score = sum(s.score for s in sources) / len(sources)
        source_count_factor = min(len(sources) / 5.0, 1.0)  # 5个来源为满分
        
        confidence = avg_score * 0.7 + source_count_factor * 0.3
        return min(max(confidence, 0.0), 1.0)
    
    def get_info(self) -> Dict[str, Any]:
        """获取 Agent 信息
        
        Returns:
            Agent 信息字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "intent_type": self.intent_type.value,
            "tools": list(self.tools.keys()),
            "has_llm": self.llm_client is not None,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, intent_type={self.intent_type.value})"


if __name__ == "__main__":
    print("=" * 50)
    print("测试 Specialized Agent Base 模块")
    print("=" * 50)
    
    # 测试 1: IntentType 枚举
    print(f"✓ IntentType 枚举值: {[t.value for t in IntentType]}")
    
    # 测试 2: Source 类
    source = Source(
        content="这是测试内容",
        source_id="doc_001",
        source_type="vector",
        score=0.95,
        metadata={"author": "test"}
    )
    print(f"✓ Source 创建: id={source.source_id}, score={source.score}")
    
    # 测试 3: AgentContext 类
    context = AgentContext(
        query="什么是人工智能？",
        conversation_history=[{"role": "user", "content": "你好"}],
        retrieved_docs=[source],
        entities=["人工智能", "机器学习"]
    )
    print(f"✓ AgentContext 创建: query={context.query}")
    print(f"  entities: {context.entities}")
    
    # 测试 4: AgentResponse 类
    response = AgentResponse(
        answer="人工智能是计算机科学的分支。",
        sources=[source],
        confidence=0.85,
        reasoning_chain=["步骤1", "步骤2"],
        metadata={"model": "test"}
    )
    print(f"✓ AgentResponse 创建: confidence={response.confidence}")
    print(f"  answer: {response.answer[:20]}...")
    
    # 测试 5: 抽象类不能实例化
    try:
        agent = SpecializedAgentBase()
    except TypeError:
        print(f"✓ 抽象类无法实例化")
    
    # 测试 6: 创建具体实现类
    class TestAgent(SpecializedAgentBase):
        @property
        def name(self) -> str:
            return "test_agent"
        
        @property
        def description(self) -> str:
            return "测试 Agent"
        
        @property
        def intent_type(self) -> IntentType:
            return IntentType.FACTUAL
        
        def process(self, query: str, context: AgentContext) -> AgentResponse:
            return AgentResponse(
                answer="测试答案",
                sources=context.retrieved_docs,
                confidence=0.9
            )
    
    agent = TestAgent()
    print(f"✓ 具体实现类创建: name={agent.name}, intent={agent.intent_type.value}")
    
    # 测试 7: 处理查询
    result = agent.process("测试问题", context)
    print(f"✓ 处理查询: answer={result.answer}")
    
    # 测试 8: 获取信息
    info = agent.get_info()
    print(f"✓ Agent 信息: {info}")
    
    print("\n所有测试通过!")
