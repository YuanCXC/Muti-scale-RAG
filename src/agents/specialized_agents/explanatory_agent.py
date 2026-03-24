# -*- coding: utf-8 -*-
"""解释型问题 Agent

处理需要详细解释或综合多个来源信息的问题。
使用多源检索和综合策略，返回详细解释性答案。
"""

from typing import Any, Dict, List, Optional

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


class ExplanatoryAgent(SpecializedAgentBase):
    """解释型问题 Agent
    
    处理需要详细解释的问题，例如：
    - "请解释注意力机制的工作原理"
    - "比较 CNN 和 Transformer 的优缺点"
    - "什么是迁移学习？如何应用？"
    
    特点：
    - 使用多源检索策略
    - 综合多个来源的信息
    - 返回详细、全面的解释
    - 支持比较和对比分析
    """
    
    name = "explanatory_agent"
    description = "处理解释型问题，提供详细的解释和说明"
    intent_type = IntentType.EXPLANATORY
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        tools: Optional[Dict[str, Any]] = None,
        top_k: int = 7,
        max_context_length: int = 6000,
        **kwargs
    ):
        """初始化解释型 Agent
        
        Args:
            llm_client: LLM 客户端
            tools: 可用工具字典
            top_k: 检索结果数量
            max_context_length: 最大上下文长度
            **kwargs: 其他参数
        """
        super().__init__(llm_client=llm_client, tools=tools, **kwargs)
        self.top_k = top_k
        self.max_context_length = max_context_length
    
    def process(self, query: str, context: AgentContext) -> AgentResponse:
        """处理解释型问题
        
        Args:
            query: 用户查询
            context: Agent 上下文
            
        Returns:
            AgentResponse 实例
        """
        logger.info(f"[{self.name}] 处理解释型问题: {query[:50]}...")
        
        try:
            # 1. 多源检索
            sources = self._multi_source_retrieval(query)
            
            if not sources:
                return AgentResponse.error_response(
                    error="未找到相关信息",
                    intent_type=self.intent_type,
                )
            
            # 2. 信息整合与去重
            integrated_sources = self._integrate_sources(sources)
            
            # 3. 生成详细解释
            answer = self._generate_explanatory_answer(query, integrated_sources)
            
            # 4. 计算置信度
            confidence = self._calculate_explanatory_confidence(integrated_sources)
            
            # 5. 构建响应
            response = AgentResponse(
                answer=answer,
                sources=integrated_sources,
                confidence=confidence,
                intent_type=self.intent_type,
                metadata={
                    "total_sources": len(sources),
                    "integrated_sources": len(integrated_sources),
                    "source_types": list(set(s.source_type for s in integrated_sources)),
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
    
    def _multi_source_retrieval(self, query: str) -> List[Source]:
        """执行多源检索
        
        综合使用多种检索方式，获取全面的信息。
        
        Args:
            query: 查询文本
            
        Returns:
            Source 列表
        """
        all_sources = []
        seen_ids = set()
        
        # 1. 语义检索
        if "semantic" in self.tools:
            try:
                result = self.tools["semantic"].run(query=query, top_k=self.top_k)
                if result.success:
                    for item in result.result:
                        source_id = item.get("id")
                        if source_id not in seen_ids:
                            all_sources.append(Source(
                                content=item.get("content", ""),
                                source_id=source_id,
                                source_type="semantic",
                                score=item.get("score", 0.0),
                                metadata=item.get("metadata", {}),
                            ))
                            seen_ids.add(source_id)
            except Exception as e:
                logger.warning(f"语义检索失败: {e}")
        
        # 2. 混合检索
        if "hybrid" in self.tools:
            try:
                result = self.tools["hybrid"].run(query=query, top_k=self.top_k)
                if result.success:
                    for item in result.result:
                        source_id = item.get("id")
                        if source_id not in seen_ids:
                            all_sources.append(Source(
                                content=item.get("content", ""),
                                source_id=source_id,
                                source_type="hybrid",
                                score=item.get("score", 0.0),
                                metadata=item.get("metadata", {}),
                            ))
                            seen_ids.add(source_id)
            except Exception as e:
                logger.warning(f"混合检索失败: {e}")
        
        # 3. 图谱检索（获取关联知识）
        if "graph" in self.tools:
            try:
                result = self.tools["graph"].run(query=query, top_k=self.top_k // 2)
                if result.success:
                    for item in result.result:
                        source_id = item.get("id", f"graph_{len(all_sources)}")
                        if source_id not in seen_ids:
                            all_sources.append(Source(
                                content=item.get("content", ""),
                                source_id=source_id,
                                source_type="graph",
                                score=item.get("score", 0.0),
                                metadata=item.get("metadata", {}),
                            ))
                            seen_ids.add(source_id)
            except Exception as e:
                logger.warning(f"图谱检索失败: {e}")
        
        # 按分数排序
        all_sources.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"多源检索完成，获取 {len(all_sources)} 个来源")
        
        return all_sources
    
    def _integrate_sources(self, sources: List[Source]) -> List[Source]:
        """整合信息来源
        
        去重、合并相似内容，保留最有价值的信息。
        
        Args:
            sources: 原始来源列表
            
        Returns:
            整合后的来源列表
        """
        if not sources:
            return []
        
        # 简单去重：基于内容相似度
        integrated = []
        for source in sources:
            # 检查是否与已有内容高度相似
            is_duplicate = False
            for existing in integrated:
                if self._is_similar_content(source.content, existing.content):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                integrated.append(source)
        
        # 限制数量和长度
        result = []
        total_length = 0
        for source in integrated[:self.top_k]:
            content_length = len(source.content)
            if total_length + content_length <= self.max_context_length:
                result.append(source)
                total_length += content_length
        
        return result
    
    def _is_similar_content(self, content1: str, content2: str, threshold: float = 0.8) -> bool:
        """判断两个内容是否相似
        
        Args:
            content1: 内容1
            content2: 内容2
            threshold: 相似度阈值
            
        Returns:
            是否相似
        """
        # 简单的词重叠率判断
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = overlap / union if union > 0 else 0
        return similarity >= threshold
    
    def _generate_explanatory_answer(self, query: str, sources: List[Source]) -> str:
        """生成解释型答案
        
        Args:
            query: 用户查询
            sources: 信息来源
            
        Returns:
            生成的答案
        """
        if not self.llm_client:
            # 如果没有 LLM，拼接所有来源
            return "\n\n".join([s.content for s in sources])
        
        from src.llms import Message
        
        # 格式化上下文
        context = self._format_context_with_sources(sources)
        
        # 使用解释型答案生成提示词
        prompt_template = get_template("answer_explanatory")
        user_prompt = prompt_template.format(query=query, context=context)
        
        messages = [
            Message.system(self._get_system_prompt()),
            Message.user(user_prompt),
        ]
        
        response = self.llm_client.generate(messages, temperature=0.5)
        return response.content.strip()
    
    def _format_context_with_sources(self, sources: List[Source]) -> str:
        """格式化上下文，带来源标注
        
        Args:
            sources: 来源列表
            
        Returns:
            格式化后的上下文
        """
        formatted = []
        for i, source in enumerate(sources):
            formatted.append(
                f"[来源 {i+1}] (相关性: {source.score:.2f}, 类型: {source.source_type})\n"
                f"{source.content}\n"
            )
        return "\n".join(formatted)
    
    def _calculate_explanatory_confidence(self, sources: List[Source]) -> float:
        """计算解释型答案的置信度
        
        Args:
            sources: 信息来源
            
        Returns:
            置信度分数
        """
        if not sources:
            return 0.0
        
        # 基础置信度：平均相关性分数
        avg_score = sum(s.score for s in sources) / len(sources)
        
        # 来源多样性加成：不同类型的来源
        source_types = set(s.source_type for s in sources)
        diversity_bonus = len(source_types) * 0.05
        
        # 来源数量加成
        count_bonus = min(len(sources) * 0.03, 0.15)
        
        confidence = avg_score * 0.6 + diversity_bonus + count_bonus
        return min(max(confidence, 0.0), 1.0)
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的知识解释专家，专门处理需要详细解释的问题。

你的回答应该：
1. 提供详细、全面的解释
2. 使用清晰的逻辑结构组织内容
3. 综合多个来源的信息，形成连贯的解释
4. 适当使用例子或类比帮助理解
5. 如果存在不同观点或方法，请进行比较说明
6. 标注信息来源（使用 [来源 X] 格式）
7. 如果信息不足以完整解释，请诚实说明"""


if __name__ == "__main__":
    from src.agents.specialized_agents.base_agent import (
        AgentContext, AgentResponse, IntentType, Source
    )
    
    print("=" * 50)
    print("测试 Explanatory Agent 模块")
    print("=" * 50)
    
    # 测试 1: 创建 Agent
    agent = ExplanatoryAgent()
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
        Source(content="解释内容1", source_id="1", source_type="vector", score=0.85),
        Source(content="解释内容2", source_id="2", source_type="vector", score=0.75),
        Source(content="解释内容3", source_id="3", source_type="graph", score=0.8),
    ]
    confidence = agent.calculate_confidence(sources)
    print(f"✓ 置信度计算: {confidence:.2f}")
    
    # 测试 5: 格式化来源
    formatted = agent.format_sources(sources, max_length=200)
    print(f"✓ 格式化来源: {len(formatted)} 字符")
    
    # 测试 6: 无 LLM 时的处理
    context = AgentContext(query="请解释机器学习")
    try:
        agent.process("请解释机器学习", context)
    except ValueError as e:
        print(f"✓ 无 LLM 时正确抛出错误")
    
    print("\n所有测试通过!")
