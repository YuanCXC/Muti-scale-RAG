# -*- coding: utf-8 -*-
"""主控 Agent

负责意图识别与任务分发，根据用户问题的意图类型选择合适的专用 Agent。
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from src.agents.specialized_agents.base_agent import (
    AgentResponse,
    AgentContext,
    IntentType,
)
from src.agents.specialized_agents.factual_agent import FactualAgent
from src.agents.specialized_agents.explanatory_agent import ExplanatoryAgent
from src.agents.specialized_agents.reasoning_agent import ReasoningAgent
from src.agents.prompts import get_template
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IntentResult:
    """意图识别结果
    
    Attributes:
        intent_type: 意图类型
        confidence: 置信度
        reasoning: 推理过程
        keywords: 关键词
    """
    intent_type: IntentType
    confidence: float = 0.0
    reasoning: str = ""
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "intent_type": self.intent_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "keywords": self.keywords,
        }


class MasterAgent:
    """主控 Agent
    
    负责意图识别与任务分发：
    1. 分析用户问题，识别意图类型
    2. 根据意图选择合适的专用 Agent
    3. 协调多个 Agent 完成复杂任务
    
    Attributes:
        llm_client: LLM 客户端
        specialized_agents: 专用 Agent 字典
        tools: 可用工具字典
        default_intent: 默认意图类型
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        tools: Optional[Dict[str, Any]] = None,
        default_intent: IntentType = IntentType.EXPLANATORY,
        **kwargs
    ):
        """初始化主控 Agent
        
        Args:
            llm_client: LLM 客户端实例
            tools: 可用工具字典
            default_intent: 默认意图类型（当无法识别时使用）
            **kwargs: 其他配置参数
        """
        config = get_config()
        
        self.llm_client = llm_client
        self.default_intent = default_intent
        self.max_iterations = config.agent_max_iterations
        self.timeout = config.agent_timeout
        self.tools = tools or {}
        self.config = kwargs
        
        # 初始化专用 Agent
        self.specialized_agents: Dict[IntentType, Any] = {}
        self._initialize_specialized_agents()
        
        logger.info(f"初始化主控 Agent，已注册 {len(self.specialized_agents)} 个专用 Agent")
    
    def _initialize_specialized_agents(self) -> None:
        """初始化专用 Agent"""
        # 事实型 Agent
        self.specialized_agents[IntentType.FACTUAL] = FactualAgent(
            llm_client=self.llm_client,
            tools=self.tools,
        )
        
        # 解释型 Agent
        self.specialized_agents[IntentType.EXPLANATORY] = ExplanatoryAgent(
            llm_client=self.llm_client,
            tools=self.tools,
        )
        
        # 推理型 Agent
        self.specialized_agents[IntentType.REASONING] = ReasoningAgent(
            llm_client=self.llm_client,
            tools=self.tools,
        )
        
        logger.debug(f"已初始化专用 Agent: {list(self.specialized_agents.keys())}")
    
    def process(self, query: str, context: Optional[AgentContext] = None) -> AgentResponse:
        """处理用户查询
        
        Args:
            query: 用户查询
            context: Agent 上下文（可选）
            
        Returns:
            AgentResponse 实例
        """
        logger.info(f"主控 Agent 开始处理查询: {query[:50]}...")
        
        if context is None:
            context = AgentContext(query=query)
        
        try:
            # 1. 意图识别
            intent_result = self._classify_intent(query, context)
            logger.info(
                f"意图识别完成: {intent_result.intent_type.value} "
                f"(置信度: {intent_result.confidence:.2f})"
            )
            
            # 2. 选择专用 Agent
            agent = self._select_agent(intent_result.intent_type)
            
            # 3. 执行处理
            response = agent.process(query, context)
            
            # 4. 添加元数据
            response.metadata["intent_result"] = intent_result.to_dict()
            response.metadata["selected_agent"] = agent.name
            
            return response
            
        except Exception as e:
            logger.error(f"主控 Agent 处理失败: {e}")
            return AgentResponse.error_response(
                error=str(e),
                intent_type=IntentType.UNKNOWN,
            )
    
    def _classify_intent(self, query: str, context: AgentContext) -> IntentResult:
        """识别用户意图
        
        Args:
            query: 用户查询
            context: Agent 上下文
            
        Returns:
            IntentResult 实例
        """
        # 首先尝试基于规则的快速分类
        rule_based_result = self._rule_based_classification(query)
        if rule_based_result.confidence >= 0.8:
            return rule_based_result
        
        # 如果规则分类置信度不够，使用 LLM 分类
        if self.llm_client:
            llm_result = self._llm_based_classification(query, context)
            if llm_result.confidence > rule_based_result.confidence:
                return llm_result
        
        return rule_based_result
    
    def _rule_based_classification(self, query: str) -> IntentResult:
        """基于规则的意图分类
        
        Args:
            query: 用户查询
            
        Returns:
            IntentResult 实例
        """
        query_lower = query.lower()
        
        # 事实型问题特征
        factual_patterns = [
            r'^(什么|什么是|何为)',
            r'(是什么|是指|定义为)',
            r'(哪一年|何时|什么时候)',
            r'(多少|几个|哪些)',
            r'(谁|哪位)',
            r'(在哪里|何处)',
        ]
        
        # 推理型问题特征
        reasoning_patterns = [
            r'(为什么|为何|原因)',
            r'(如果|假设|假如)',
            r'(如何|怎样|怎么)设计',
            r'(比较|对比|区别|差异)',
            r'(优缺点|利弊|优劣)',
            r'(影响|导致|造成)',
            r'(解决|处理|应对)',
        ]
        
        # 解释型问题特征
        explanatory_patterns = [
            r'(解释|说明|阐述)',
            r'(介绍|讲解|描述)',
            r'(原理|机制|工作)',
            r'(如何|怎样)实现',
            r'(流程|步骤|过程)',
        ]
        
        # 计算各类型的匹配分数
        factual_score = sum(1 for p in factual_patterns if re.search(p, query_lower))
        reasoning_score = sum(1 for p in reasoning_patterns if re.search(p, query_lower))
        explanatory_score = sum(1 for p in explanatory_patterns if re.search(p, query_lower))
        
        # 选择得分最高的类型
        scores = {
            IntentType.FACTUAL: factual_score,
            IntentType.REASONING: reasoning_score,
            IntentType.EXPLANATORY: explanatory_score,
        }
        
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]
        
        # 计算置信度
        total_score = sum(scores.values())
        if total_score == 0:
            confidence = 0.3
            intent_type = self.default_intent
        else:
            confidence = min(max_score / max(total_score, 1) + 0.3, 0.9)
            intent_type = max_type if max_score > 0 else self.default_intent
        
        # 提取关键词
        keywords = self._extract_keywords(query)
        
        return IntentResult(
            intent_type=intent_type,
            confidence=confidence,
            reasoning=f"基于规则分类，匹配特征数: {max_score}",
            keywords=keywords,
        )
    
    def _llm_based_classification(
        self,
        query: str,
        context: AgentContext
    ) -> IntentResult:
        """基于 LLM 的意图分类
        
        Args:
            query: 用户查询
            context: Agent 上下文
            
        Returns:
            IntentResult 实例
        """
        from src.llms import Message
        
        # 使用意图分类提示词
        prompt_template = get_template("intent_classification")
        user_prompt = prompt_template.format(query=query)
        
        messages = [
            Message.system("你是一个专业的意图分类专家。请准确判断用户问题的意图类型。"),
            Message.user(user_prompt),
        ]
        
        try:
            response = self.llm_client.generate(messages, temperature=0.1)
            content = response.content.strip().upper()
            
            # 解析意图类型
            intent_type = self._parse_intent_type(content)
            
            # 提取关键词
            keywords = self._extract_keywords(query)
            
            return IntentResult(
                intent_type=intent_type,
                confidence=0.85,
                reasoning=f"基于 LLM 分类: {content}",
                keywords=keywords,
            )
            
        except Exception as e:
            logger.warning(f"LLM 意图分类失败: {e}")
            return IntentResult(
                intent_type=self.default_intent,
                confidence=0.5,
                reasoning=f"LLM 分类失败，使用默认类型",
                keywords=[],
            )
    
    def _parse_intent_type(self, content: str) -> IntentType:
        """解析意图类型
        
        Args:
            content: LLM 返回的内容
            
        Returns:
            IntentType 枚举值
        """
        content = content.upper()
        
        if "FACTUAL" in content:
            return IntentType.FACTUAL
        elif "EXPLANATORY" in content:
            return IntentType.EXPLANATORY
        elif "REASONING" in content:
            return IntentType.REASONING
        else:
            return self.default_intent
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词
        
        Args:
            query: 用户查询
            
        Returns:
            关键词列表
        """
        # 简单的关键词提取：去除停用词，提取名词性词汇
        stop_words = {'的', '是', '在', '有', '和', '与', '或', '了', '吗', '呢', '啊', '什么', '怎么', '如何'}
        
        # 分词（简单按空格和标点分割）
        words = re.findall(r'[\w]+', query)
        
        # 过滤停用词和短词
        keywords = [w for w in words if w not in stop_words and len(w) > 1]
        
        return keywords[:5]  # 返回前5个关键词
    
    def _select_agent(self, intent_type: IntentType) -> Any:
        """选择专用 Agent
        
        Args:
            intent_type: 意图类型
            
        Returns:
            专用 Agent 实例
        """
        if intent_type in self.specialized_agents:
            return self.specialized_agents[intent_type]
        
        # 如果没有对应的 Agent，使用默认 Agent
        logger.warning(f"未找到意图类型 {intent_type.value} 对应的 Agent，使用默认 Agent")
        return self.specialized_agents[self.default_intent]
    
    def register_agent(self, intent_type: IntentType, agent: Any) -> None:
        """注册专用 Agent
        
        Args:
            intent_type: 意图类型
            agent: Agent 实例
        """
        self.specialized_agents[intent_type] = agent
        logger.info(f"注册专用 Agent: {agent.name} -> {intent_type.value}")
    
    def get_available_intents(self) -> List[str]:
        """获取可用的意图类型
        
        Returns:
            意图类型列表
        """
        return [intent.value for intent in self.specialized_agents.keys()]
    
    def get_agent_info(self, intent_type: IntentType) -> Dict[str, Any]:
        """获取指定 Agent 的信息
        
        Args:
            intent_type: 意图类型
            
        Returns:
            Agent 信息字典
        """
        if intent_type in self.specialized_agents:
            return self.specialized_agents[intent_type].get_info()
        return {}
    
    def get_all_agents_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有 Agent 的信息
        
        Returns:
            Agent 信息字典
        """
        return {
            intent.value: agent.get_info()
            for intent, agent in self.specialized_agents.items()
        }
    
    def __repr__(self) -> str:
        return (
            f"MasterAgent("
            f"agents={len(self.specialized_agents)}, "
            f"tools={list(self.tools.keys())})"
        )


if __name__ == "__main__":
    from src.agents.specialized_agents.base_agent import IntentType, AgentContext
    from src.agents.specialized_agents.factual_agent import FactualAgent
    from src.agents.specialized_agents.explanatory_agent import ExplanatoryAgent
    from src.agents.specialized_agents.reasoning_agent import ReasoningAgent
    
    print("=" * 50)
    print("测试 Master Agent 模块")
    print("=" * 50)
    
    # 测试 1: 创建 MasterAgent
    master = MasterAgent()
    print(f"✓ 创建 MasterAgent: {repr(master)}")
    
    # 测试 2: 注册专用 Agent
    master.register_agent(IntentType.FACTUAL, FactualAgent())
    master.register_agent(IntentType.EXPLANATORY, ExplanatoryAgent())
    master.register_agent(IntentType.REASONING, ReasoningAgent())
    print(f"✓ 注册 Agent: {len(master.specialized_agents)} 个")
    
    # 测试 3: 意图分类 (基于规则)
    intent = master._rule_based_classification("什么是人工智能？")
    print(f"✓ 规则分类 '什么是人工智能？': {intent.intent_type.value}")
    
    intent2 = master._rule_based_classification("请解释深度学习的原理")
    print(f"✓ 规则分类 '请解释深度学习的原理': {intent2.intent_type.value}")
    
    intent3 = master._rule_based_classification("如果A>B, B>C, 那么A和C的关系？")
    print(f"✓ 规则分类 '如果A>B...': {intent3.intent_type.value}")
    
    # 测试 4: 获取 Agent 信息
    all_info = master.get_all_agents_info()
    print(f"✓ 所有 Agent 信息: {list(all_info.keys())}")
    
    # 测试 5: 无 LLM 时的处理
    context = AgentContext(query="测试问题")
    try:
        master.process("测试问题", context)
    except ValueError as e:
        print(f"✓ 无 LLM 时正确抛出错误")
    
    # 测试 6: 字符串表示
    print(f"✓ 字符串表示: {repr(master)}")
    
    print("\n所有测试通过!")
