# -*- coding: utf-8 -*-
"""LLM 基础客户端模块

定义 LLM 客户端的通用接口和数据结构。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator
from abc import ABC, abstractmethod
import time

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """消息数据结构
    
    Attributes:
        role: 角色 (system, user, assistant)
        content: 消息内容
        name: 名称（可选）
    """
    role: str
    content: str
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class LLMResponse:
    """LLM 响应数据结构
    
    Attributes:
        content: 响应内容
        model: 使用的模型
        usage: token 使用情况
        latency: 响应延迟
        success: 是否成功
        error: 错误信息
    """
    content: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    latency: float = 0.0
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "latency": self.latency,
            "success": self.success,
            "error": self.error,
        }


class BaseLLMClient(ABC):
    """LLM 客户端基类
    
    定义 LLM 客户端的通用接口。
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """初始化 LLM 客户端
        
        Args:
            model: 模型名称
            api_key: API 密钥
            base_url: API 基础 URL
            **kwargs: 其他参数
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs
        
        self.max_retries = kwargs.get("max_retries", 3)
        self.retry_delay = kwargs.get("retry_delay", 1.0)
        self.timeout = kwargs.get("timeout", 120)
        
        logger.info(f"初始化 {self.__class__.__name__}: model={model}")
    
    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        """生成响应
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他参数
            
        Returns:
            LLMResponse 实例
        """
        pass
    
    def generate_with_retry(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> LLMResponse:
        """带重试的生成
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他参数
            
        Returns:
            LLMResponse 实例
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return self.generate(messages, temperature, max_tokens, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"生成失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        return LLMResponse(
            content="",
            model=self.model,
            success=False,
            error=str(last_error),
        )
    
    def stream_generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Iterator[str]:
        """流式生成响应
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 其他参数
            
        Yields:
            生成的文本片段
        """
        response = self.generate(messages, temperature, max_tokens, **kwargs)
        yield response.content
    
    def _build_request_body(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """构建请求体"""
        return {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }


if __name__ == "__main__":
    print("=" * 50)
    print("测试 LLM 基础模块")
    print("=" * 50)
    
    message = Message(role="user", content="你好")
    print(f"✓ Message 创建: role={message.role}, content={message.content}")
    
    message_dict = message.to_dict()
    print(f"✓ Message 转字典: {message_dict}")
    
    response = LLMResponse(
        content="你好！有什么可以帮助你的？",
        model="test-model",
        latency=0.5,
    )
    print(f"✓ LLMResponse 创建: content={response.content[:20]}...")
    
    print("\n所有测试通过!")
