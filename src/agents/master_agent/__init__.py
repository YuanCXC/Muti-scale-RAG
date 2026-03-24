# -*- coding: utf-8 -*-
"""主控 Agent 模块

提供意图识别与任务分发功能。
"""

from .master_agent import MasterAgent, IntentResult

__all__ = [
    "MasterAgent",
    "IntentResult",
]
