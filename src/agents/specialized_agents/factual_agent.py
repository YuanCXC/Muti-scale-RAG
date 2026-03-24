# -*- coding: utf-8 -*-
"""事实型问题 Agent

处理需要精确事实性答案的问题。
使用精确检索策略，返回高置信度答案。
"""

from typing import Any, Dict, Optional

from src.agents.specialized_agents.base_agent import (
    SpecializedAgentBase,
    AgentResponse,
    AgentContext,
    IntentType,
    Source,
)
from src.agents.prompts import get_template
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FactualAgent(SpecializedAgentBase):
    """事实型问题 Agent
    
    处理需要精确事实性答案的问题，例如：
    - "什么是 CLIP？"
    - "ResNet 是哪一年提出的？"
    - "BERT 的参数量是多少？"
    
    特点：
    - 使用精确检索策略
    - 优先使用高相关性的单一来源
    - 返回简洁、准确的答案
    - 高置信度要求
    """
    
    name = "factual_agent"
    description = "处理事实型问题，提供精确的答案"
    intent_type = IntentType.FACTUAL
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        tools: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
        min_confidence: float = 0.6,
        **kwargs
    ):
        """初始化事实型 Agent
        
        Args:
            llm_client: LLM 客户端
            tools: 可用工具字典
            top_k: 检索结果数量
            min_confidence: 最小置信度阈值
            **kwargs: 其他参数
        """
        super().__init__(llm_client=llm_client, tools=tools, **kwargs)
        self.top_k = top_k
        self.min_confidence = min_confidence
    
    def process(self, query: str, context: AgentContext) -> AgentResponse:
        """处理事实型问题
        
        Args:
            query: 用户查询
            context: Agent 上下文
            
        Returns:
            AgentResponse 实例
        """
        logger.info(f"[{self.name}] 处理事实型问题: {query[:50]}...")
        
        try:
            # 1. 执行精确检索
            sources = self._precise_retrieval(query)
            
            if not sources:
                return AgentResponse.error_response(
                    error="未找到相关信息",
                    intent_type=self.intent_type,
                )
            
            # 2. 选择最佳来源
            best_sources = self._select_best_sources(sources)
            
            # 3. 生成答案
            answer = self._generate_factual_answer(query, best_sources)
            
            # 4. 计算置信度
            confidence = self._calculate_factual_confidence(best_sources)
            
            # 5. 构建响应
            response = AgentResponse(
                answer=answer,
                sources=best_sources,
                confidence=confidence,
                intent_type=self.intent_type,
                metadata={
                    "retrieval_count": len(sources),
                    "selected_count": len(best_sources),
                    "agent": self.name,
                },
            )
            
            logger.info(f"[{self.name}] 处理完成，置信度: {confidence:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"[{self.name}] 处理失败: {e}")
            return AgentResponse.error_response(
                error=str(e),
                intent_type=self.intent_type,
            )
    
    def _precise_retrieval(self, query: str) -> list:
        """执行精确检索
        
        优先使用语义检索，确保高相关性。
        
        Args:
            query: 查询文本
            
        Returns:
            Source 列表
        """
        sources = []
        
        # 优先使用语义检索
        if "semantic" in self.tools:
            try:
                result = self.tools["semantic"].run(query=query, top_k=self.top_k)
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
        
        # 如果语义检索结果不足，使用混合检索
        if len(sources) < self.top_k and "hybrid" in self.tools:
            try:
                result = self.tools["hybrid"].run(query=query, top_k=self.top_k)
                if result.success:
                    for item in result.result:
                        # 避免重复
                        if not any(s.source_id == item.get("id") for s in sources):
                            sources.append(Source(
                                content=item.get("content", ""),
                                source_id=item.get("id"),
                                source_type="hybrid",
                                score=item.get("score", 0.0),
                                metadata=item.get("metadata", {}),
                            ))
            except Exception as e:
                logger.warning(f"混合检索失败: {e}")
        
        # 按分数排序
        sources.sort(key=lambda x: x.score, reverse=True)
        return sources
    
    def _select_best_sources(self, sources: list, min_score: float = 0.5) -> list:
        """选择最佳来源
        
        对于事实型问题，优先选择高相关性的来源。
        
        Args:
            sources: 来源列表
            min_score: 最小分数阈值
            
        Returns:
            筛选后的来源列表
        """
        # 过滤低分来源
        filtered = [s for s in sources if s.score >= min_score]
        
        # 限制数量
        return filtered[:self.top_k]
    
    def _generate_factual_answer(self, query: str, sources: list) -> str:
        """生成事实型答案
        
        Args:
            query: 用户查询
            sources: 信息来源
            
        Returns:
            生成的答案
        """
        if not self.llm_client:
            # 如果没有 LLM，直接返回最佳来源的内容
            return sources[0].content if sources else "无法生成答案"
        
        from src.llms import Message
        
        # 格式化上下文
        context = self.format_sources(sources, max_length=2000)
        
        # 使用事实型答案生成提示词
        prompt_template = get_template("answer_factual")
        user_prompt = prompt_template.format(query=query, context=context)
        
        messages = [
            Message.system(self._get_system_prompt()),
            Message.user(user_prompt),
        ]
        
        response = self.llm_client.generate(messages, temperature=0.3)
        return response.content.strip()
    
    def _calculate_factual_confidence(self, sources: list) -> float:
        """计算事实型答案的置信度
        
        事实型问题需要更高的置信度要求。
        
        Args:
            sources: 信息来源
            
        Returns:
            置信度分数
        """
        if not sources:
            return 0.0
        
        # 基础置信度：平均相关性分数
        avg_score = sum(s.score for s in sources) / len(sources)
        
        # 来源一致性加成：如果多个来源都支持同一事实
        consistency_bonus = min(len(sources) * 0.1, 0.2)
        
        # 最高分来源加成
        top_score_bonus = sources[0].score * 0.1
        
        confidence = avg_score * 0.7 + consistency_bonus + top_score_bonus
        return min(max(confidence, 0.0), 1.0)
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的知识问答助手，专门处理事实型问题。

你的回答应该：
1. 直接、准确地回答问题
2. 提供具体的事实和数据
3. 如果信息来源明确，请标注引用
4. 保持简洁，不要添加不必要的解释
5. 如果信息不足以确定答案，请诚实说明"""


if __name__ == "__main__":
    from src.agents.specialized_agents.base_agent import (
        AgentContext, AgentResponse, IntentType, Source
    )
    
    print("=" * 50)
    print("测试 Factual Agent 模块")
    print("=" * 50)
    
    # 测试 1: 创建 Agent
    agent = FactualAgent()
    print(f"✓ 创建 Agent: name={agent.name}")
    print(f"  intent_type={agent.intent_type.value}")
    
    # 测试 2: Agent 信息
    info = agent.get_info()
    print(f"✓ Agent 信息: {info['name']}")
    
    # 测试 3: 系统提示词
    system_prompt = agent._get_system_prompt()
    print(f"✓ 系统提示词长度: {len(system_prompt)}")
    
    # 测试 4: 计算置信度
    sources = [
        Source(content="内容1", source_id="1", source_type="vector", score=0.9),
        Source(content="内容2", source_id="2", source_type="vector", score=0.8),
    ]
    confidence = agent.calculate_confidence(sources)
    print(f"✓ 置信度计算: {confidence:.2f}")
    
    # 测试 5: 格式化来源
    formatted = agent.format_sources(sources, max_length=100)
    print(f"✓ 格式化来源: {len(formatted)} 字符")
    
    # 测试 6: 无 LLM 时的处理
    context = AgentContext(query="测试问题")
    try:
        agent.process("测试问题", context)
    except ValueError as e:
        print(f"✓ 无 LLM 时正确抛出错误: {str(e)[:30]}...")
    
    print("\n所有测试通过!")
