# -*- coding: utf-8 -*-
"""提示词管理模块

提供提示词的加载、管理和变量替换功能。

使用示例:
    from src.agents.prompts import PromptManager, get_template
    
    # 使用预定义模板
    template = get_template("intent_classification")
    
    # 使用提示词管理器
    manager = PromptManager()
    manager.load_from_string("greeting", "你好，{name}！")
    prompt = manager.render("greeting", name="张三")
"""

from .prompt_manager import PromptManager, PromptVersion
from .templates import (
    PromptTemplates,
    get_template,
    list_templates,
)

__all__ = [
    # 提示词管理器
    "PromptManager",
    "PromptVersion",
    # 预定义模板
    "PromptTemplates",
    "get_template",
    "list_templates",
]
