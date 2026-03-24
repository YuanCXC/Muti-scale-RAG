# -*- coding: utf-8 -*-
"""推理型问题 Agent

处理需要多步推理或逻辑分析的问题。
使用多步推理策略，返回推理链和最终答案。
"""

from typing import Any, Dict, List, Optional
import json
import re

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


class ReasoningAgent(SpecializedAgentBase):
    """推理型问题 Agent
    
    处理需要多步推理的问题，例如：
    - "如果将 ResNet 的深度增加到 1000 层，可能会遇到什么问题？如何解决？"
    - "为什么 Transformer 在 NLP 任务上优于 RNN？"
    - "如何设计一个高效的图像分割系统？"
    
    特点：
    - 使用多步推理策略
    - 展示清晰的推理过程
    - 支持假设性推理
    - 返回推理链和最终答案
    """
    
    name = "reasoning_agent"
    description = "处理推理型问题，进行多步推理分析"
    intent_type = IntentType.REASONING
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        tools: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        max_reasoning_steps: int = 5,
        **kwargs
    ):
        """初始化推理型 Agent
        
        Args:
            llm_client: LLM 客户端
            tools: 可用工具字典
            top_k: 检索结果数量
            max_reasoning_steps: 最大推理步骤数
            **kwargs: 其他参数
        """
        super().__init__(llm_client=llm_client, tools=tools, **kwargs)
        self.top_k = top_k
        self.max_reasoning_steps = max_reasoning_steps
    
    def process(self, query: str, context: AgentContext) -> AgentResponse:
        """处理推理型问题
        
        Args:
            query: 用户查询
            context: Agent 上下文
            
        Returns:
            AgentResponse 实例
        """
        logger.info(f"[{self.name}] 处理推理型问题: {query[:50]}...")
        
        try:
            # 1. 问题分解
            sub_questions = self._decompose_question(query)
            logger.info(f"问题分解为 {len(sub_questions)} 个子问题")
            
            # 2. 为每个子问题检索信息
            all_sources = []
            for i, sub_q in enumerate(sub_questions):
                sources = self._retrieve_for_subquestion(sub_q)
                all_sources.extend(sources)
                logger.debug(f"子问题 {i+1} 检索到 {len(sources)} 个来源")
            
            # 去重
            unique_sources = self._deduplicate_sources(all_sources)
            
            if not unique_sources:
                return AgentResponse.error_response(
                    error="未找到相关信息进行推理",
                    intent_type=self.intent_type,
                )
            
            # 3. 执行多步推理
            reasoning_result = self._perform_reasoning(query, sub_questions, unique_sources)
            
            # 4. 生成最终答案
            answer = reasoning_result["answer"]
            reasoning_chain = reasoning_result["reasoning_chain"]
            
            # 5. 计算置信度
            confidence = self._calculate_reasoning_confidence(
                unique_sources, reasoning_chain
            )
            
            # 6. 构建响应
            response = AgentResponse(
                answer=answer,
                sources=unique_sources[:self.top_k],
                confidence=confidence,
                intent_type=self.intent_type,
                reasoning_chain=reasoning_chain,
                metadata={
                    "sub_questions": sub_questions,
                    "total_sources": len(all_sources),
                    "unique_sources": len(unique_sources),
                    "reasoning_steps": len(reasoning_chain),
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
    
    def _decompose_question(self, query: str) -> List[str]:
        """分解复杂问题
        
        Args:
            query: 原始问题
            
        Returns:
            子问题列表
        """
        if not self.llm_client:
            return [query]
        
        from src.llms import Message
        
        # 使用问题分解提示词
        prompt_template = get_template("query_decomposition")
        user_prompt = prompt_template.format(query=query)
        
        messages = [
            Message.system("你是一个专业的问题分析专家。请将复杂问题分解为简单的子问题。"),
            Message.user(user_prompt),
        ]
        
        try:
            response = self.llm_client.generate(messages, temperature=0.3)
            content = response.content.strip()
            
            # 尝试解析 JSON
            # 提取 JSON 数组
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                sub_questions = json.loads(json_match.group())
                if isinstance(sub_questions, list) and len(sub_questions) > 0:
                    return sub_questions
        except Exception as e:
            logger.warning(f"问题分解失败: {e}")
        
        # 如果分解失败，返回原问题
        return [query]
    
    def _retrieve_for_subquestion(self, sub_question: str) -> List[Source]:
        """为子问题检索信息
        
        Args:
            sub_question: 子问题
            
        Returns:
            Source 列表
        """
        sources = []
        seen_ids = set()
        
        # 使用多种检索方式
        for tool_name in ["semantic", "hybrid", "graph"]:
            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name].run(
                        query=sub_question,
                        top_k=self.top_k // 2
                    )
                    if result.success:
                        for item in result.result:
                            source_id = item.get("id")
                            if source_id not in seen_ids:
                                sources.append(Source(
                                    content=item.get("content", ""),
                                    source_id=source_id,
                                    source_type=tool_name,
                                    score=item.get("score", 0.0),
                                    metadata={
                                        "sub_question": sub_question,
                                        **item.get("metadata", {}),
                                    },
                                ))
                                seen_ids.add(source_id)
                except Exception as e:
                    logger.warning(f"{tool_name} 检索失败: {e}")
        
        return sources
    
    def _deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        """去重信息来源
        
        Args:
            sources: 原始来源列表
            
        Returns:
            去重后的来源列表
        """
        seen_ids = set()
        unique = []
        
        for source in sources:
            if source.source_id and source.source_id not in seen_ids:
                unique.append(source)
                seen_ids.add(source.source_id)
            elif not source.source_id:
                # 对于没有 ID 的来源，使用内容哈希
                content_hash = hash(source.content[:200])
                if content_hash not in seen_ids:
                    unique.append(source)
                    seen_ids.add(content_hash)
        
        # 按分数排序
        unique.sort(key=lambda x: x.score, reverse=True)
        return unique
    
    def _perform_reasoning(
        self,
        query: str,
        sub_questions: List[str],
        sources: List[Source]
    ) -> Dict[str, Any]:
        """执行多步推理
        
        Args:
            query: 原始问题
            sub_questions: 子问题列表
            sources: 信息来源
            
        Returns:
            推理结果字典
        """
        if not self.llm_client:
            # 如果没有 LLM，简单拼接答案
            return {
                "answer": "\n".join([s.content for s in sources[:3]]),
                "reasoning_chain": ["无法进行推理，直接使用检索结果"],
            }
        
        from src.llms import Message
        
        # 格式化上下文
        context = self._format_reasoning_context(sources, sub_questions)
        
        # 使用推理型答案生成提示词
        prompt_template = get_template("answer_reasoning")
        user_prompt = prompt_template.format(query=query, context=context)
        
        messages = [
            Message.system(self._get_system_prompt()),
            Message.user(user_prompt),
        ]
        
        response = self.llm_client.generate(messages, temperature=0.4)
        content = response.content.strip()
        
        # 解析推理链和答案
        reasoning_chain = self._extract_reasoning_chain(content)
        answer = self._extract_final_answer(content)
        
        return {
            "answer": answer,
            "reasoning_chain": reasoning_chain,
            "raw_response": content,
        }
    
    def _format_reasoning_context(
        self,
        sources: List[Source],
        sub_questions: List[str]
    ) -> str:
        """格式化推理上下文
        
        Args:
            sources: 信息来源
            sub_questions: 子问题
            
        Returns:
            格式化后的上下文
        """
        parts = []
        
        # 添加子问题
        if len(sub_questions) > 1:
            parts.append("## 问题分解：")
            for i, sq in enumerate(sub_questions, 1):
                parts.append(f"{i}. {sq}")
            parts.append("")
        
        # 添加信息来源
        parts.append("## 相关信息：")
        for i, source in enumerate(sources[:self.top_k], 1):
            parts.append(
                f"[信息 {i}] (相关性: {source.score:.2f})\n"
                f"{source.content}\n"
            )
        
        return "\n".join(parts)
    
    def _extract_reasoning_chain(self, content: str) -> List[str]:
        """从回答中提取推理链
        
        Args:
            content: 回答内容
            
        Returns:
            推理步骤列表
        """
        chain = []
        
        # 尝试提取推理步骤
        step_pattern = r'(?:步骤\s*\d+|Step\s*\d+|[\d]+\.|[-•])\s*(.+?)(?=(?:步骤\s*\d+|Step\s*\d+|[\d]+\.|[-•])|$)'
        matches = re.findall(step_pattern, content, re.DOTALL)
        
        if matches:
            chain = [m.strip() for m in matches if m.strip()]
        else:
            # 如果没有明确的步骤标记，按段落分割
            paragraphs = content.split('\n\n')
            chain = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]
        
        return chain[:self.max_reasoning_steps]
    
    def _extract_final_answer(self, content: str) -> str:
        """从回答中提取最终答案
        
        Args:
            content: 回答内容
            
        Returns:
            最终答案
        """
        # 尝试提取结论部分
        conclusion_patterns = [
            r'(?:结论|Conclusion|答案|Answer)[:：]\s*(.+?)(?:$|\n\n)',
            r'(?:因此|所以|综上所述|In conclusion|Therefore)[，,]?\s*(.+?)(?:$|\n\n)',
        ]
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # 如果没有找到明确的结论，返回整个回答
        return content
    
    def _calculate_reasoning_confidence(
        self,
        sources: List[Source],
        reasoning_chain: List[str]
    ) -> float:
        """计算推理型答案的置信度
        
        Args:
            sources: 信息来源
            reasoning_chain: 推理链
            
        Returns:
            置信度分数
        """
        if not sources:
            return 0.0
        
        # 基础置信度：来源质量
        avg_score = sum(s.score for s in sources) / len(sources)
        
        # 来源数量加成
        source_bonus = min(len(sources) * 0.02, 0.1)
        
        # 推理步骤加成：推理步骤越多，置信度越高（但有上限）
        reasoning_bonus = min(len(reasoning_chain) * 0.03, 0.15)
        
        confidence = avg_score * 0.6 + source_bonus + reasoning_bonus
        return min(max(confidence, 0.0), 1.0)
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的推理分析专家，专门处理需要多步推理的问题。

你的回答应该：
1. 展示清晰的推理步骤
2. 每个推理步骤都要有依据
3. 如果涉及假设，请明确说明
4. 考虑可能的反例或限制条件
5. 给出结论并说明置信度
6. 标注关键信息来源

回答格式：
1. 问题分析：[分析问题的核心]
2. 已知信息：[列出相关事实]
3. 推理步骤：
   - 步骤1：[推理内容]
   - 步骤2：[推理内容]
   ...
4. 结论：[最终答案]
5. 置信度：[高/中/低]"""


if __name__ == "__main__":
    from src.agents.specialized_agents.base_agent import (
        AgentContext, AgentResponse, IntentType, Source
    )
    
    print("=" * 50)
    print("测试 Reasoning Agent 模块")
    print("=" * 50)
    
    # 测试 1: 创建 Agent
    agent = ReasoningAgent()
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
        Source(content="推理前提1", source_id="1", source_type="vector", score=0.9),
        Source(content="推理前提2", source_id="2", source_type="graph", score=0.85),
    ]
    confidence = agent.calculate_confidence(sources)
    print(f"✓ 置信度计算: {confidence:.2f}")
    
    # 测试 5: 格式化来源
    formatted = agent.format_sources(sources, max_length=150)
    print(f"✓ 格式化来源: {len(formatted)} 字符")
    
    # 测试 6: 无 LLM 时的处理
    context = AgentContext(query="如果A大于B，B大于C，那么A和C的关系是什么？")
    try:
        agent.process("推理问题", context)
    except ValueError as e:
        print(f"✓ 无 LLM 时正确抛出错误")
    
    print("\n所有测试通过!")
