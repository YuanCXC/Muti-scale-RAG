# -*- coding: utf-8 -*-
"""Agent 模块

提供智能体调度和处理功能，包括：
- 主控 Agent：意图识别与任务分发
- 专用 Agent：针对不同类型问题的处理
- 提示词管理：提示词加载、管理和变量替换

使用示例:
    from src.agents import MasterAgent, FactualAgent, IntentType
    from src.llms import DeepSeekClient
    
    # 创建 LLM 客户端
    llm_client = DeepSeekClient(model="deepseek-chat")
    
    # 创建主控 Agent
    master_agent = MasterAgent(llm_client=llm_client)
    
    # 处理查询
    response = master_agent.process("什么是 CLIP？")
    print(response.answer)
"""

# 主控 Agent
from src.agents.master_agent import MasterAgent, IntentResult

# 专用 Agent
from src.agents.specialized_agents import (
    # 基础类
    SpecializedAgentBase,
    AgentResponse,
    AgentContext,
    IntentType,
    Source,
    # 专用 Agent
    FactualAgent,
    ExplanatoryAgent,
    ReasoningAgent,
)

# 提示词管理
from src.agents.prompts import (
    PromptManager,
    PromptVersion,
    PromptTemplates,
    get_template,
    list_templates,
)

__all__ = [
    # 主控 Agent
    "MasterAgent",
    "IntentResult",
    # 基础类
    "SpecializedAgentBase",
    "AgentResponse",
    "AgentContext",
    "IntentType",
    "Source",
    # 专用 Agent
    "FactualAgent",
    "ExplanatoryAgent",
    "ReasoningAgent",
    # 提示词管理
    "PromptManager",
    "PromptVersion",
    "PromptTemplates",
    "get_template",
    "list_templates",
]
