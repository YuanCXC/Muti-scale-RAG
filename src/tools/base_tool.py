# -*- coding: utf-8 -*-
"""基础工具抽象类

定义工具的统一接口，支持 LangChain Tool 接口兼容。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field


@dataclass
class ToolResult:
    """工具执行结果数据类
    
    存储工具执行的结果信息。
    
    Attributes:
        success: 执行是否成功
        result: 执行结果数据
        error: 错误信息（如果失败）
        metadata: 额外元数据
    """
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            结果字典
        """
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }
    
    @classmethod
    def success_result(
        cls,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """创建成功结果
        
        Args:
            result: 执行结果
            metadata: 额外元数据
            
        Returns:
            ToolResult 实例
        """
        return cls(
            success=True,
            result=result,
            metadata=metadata or {},
        )
    
    @classmethod
    def error_result(
        cls,
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """创建错误结果
        
        Args:
            error: 错误信息
            metadata: 额外元数据
            
        Returns:
            ToolResult 实例
        """
        return cls(
            success=False,
            error=error,
            metadata=metadata or {},
        )


class ToolArgs(BaseModel):
    """工具参数基类
    
    所有工具的参数定义都应继承此类。
    使用 Pydantic 进行参数验证。
    """
    pass


class ToolBase(ABC):
    """工具抽象基类
    
    定义工具的统一接口，所有工具实现都应继承此类。
    支持 LangChain Tool 接口兼容。
    
    Attributes:
        name: 工具名称
        description: 工具描述
        args_schema: 参数定义的 Pydantic 模型
    """
    
    name: str = "base_tool"
    description: str = "基础工具"
    args_schema: Type[ToolArgs] = ToolArgs
    
    def __init__(self, **kwargs: Any):
        """初始化工具
        
        Args:
            **kwargs: 工具配置参数
        """
        self.config = kwargs
    
    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """执行工具
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            ToolResult 实例
        """
        pass
    
    def validate_args(self, **kwargs: Any) -> Dict[str, Any]:
        """验证参数
        
        Args:
            **kwargs: 待验证的参数
            
        Returns:
            验证后的参数字典
            
        Raises:
            ValueError: 参数验证失败
        """
        if self.args_schema == ToolArgs:
            return kwargs
        
        try:
            validated = self.args_schema(**kwargs)
            return validated.model_dump()
        except Exception as e:
            raise ValueError(f"参数验证失败: {str(e)}")
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """获取参数定义的 JSON Schema
        
        Returns:
            参数定义的 JSON Schema
        """
        if self.args_schema == ToolArgs:
            return {"type": "object", "properties": {}}
        
        return self.args_schema.model_json_schema()
    
    def to_langchain_tool(self) -> "LangChainTool":
        """转换为 LangChain Tool 格式
        
        Returns:
            LangChain Tool 实例
        """
        try:
            from langchain_core.tools import BaseTool as LangChainBaseTool
            from langchain_core.tools import Tool as LangChainTool
        except ImportError:
            raise ImportError(
                "需要安装 langchain-core 才能使用 LangChain 兼容功能。"
                "请运行: pip install langchain-core"
            )
        
        # 创建 LangChain Tool
        def tool_func(query: str) -> str:
            """工具函数"""
            result = self.run(query=query)
            if result.success:
                return str(result.result)
            else:
                return f"Error: {result.error}"
        
        return LangChainTool(
            name=self.name,
            description=self.description,
            func=tool_func,
            args_schema=self.args_schema,
        )
    
    def __call__(self, **kwargs: Any) -> ToolResult:
        """使工具可调用
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            ToolResult 实例
        """
        return self.run(**kwargs)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"description='{self.description[:50]}...')"
        )
    
    def get_info(self) -> Dict[str, Any]:
        """获取工具信息
        
        Returns:
            工具信息字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema(),
            "class": self.__class__.__name__,
        }
