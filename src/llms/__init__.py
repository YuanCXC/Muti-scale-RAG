# -*- coding: utf-8 -*-
"""LLM 客户端模块

提供多种 LLM 的统一接口，支持 OpenAI、DeepSeek、通义千问等。
"""

from src.llms.base_client import BaseLLMClient, Message, LLMResponse
from src.llms.deepseek_client import DeepSeekClient
from src.llms.embedding_client import EmbeddingClient
from src.utils.config import get_config
from src.utils.logger import get_logger

__all__ = [
    "BaseLLMClient",
    "Message",
    "LLMResponse",
    "DeepSeekClient",
    "EmbeddingClient",
]


def create_client(
    provider: str = "deepseek",
    model: str = None,
    **kwargs,
) -> BaseLLMClient:
    """创建 LLM 客户端
    
    Args:
        provider: 提供商名称 (deepseek, openai, qwen, ollama)
        model: 模型名称
        **kwargs: 其他参数
        
    Returns:
        LLM 客户端实例
    """
    logger = get_logger(__name__)
    config = get_config()
    
    if provider == "deepseek":
        model = model or config.llm_model
        api_key = kwargs.pop("api_key", None) or config.openai_api_key
        base_url = kwargs.pop("base_url", None) or config.openai_api_base or "https://api.deepseek.com"
        return DeepSeekClient(api_key=api_key, model=model, base_url=base_url, **kwargs)
    
    elif provider == "openai":
        try:
            from src.llms.openai_client import OpenAIClient
            model = model or config.llm_model
            api_key = kwargs.pop("api_key", None) or config.openai_api_key
            base_url = kwargs.pop("base_url", None) or config.openai_api_base
            return OpenAIClient(api_key=api_key, model=model, base_url=base_url, **kwargs)
        except ImportError:
            logger.warning("OpenAI 客户端未安装，使用 DeepSeek 客户端替代")
            return create_client("deepseek", model, **kwargs)
    
    elif provider == "qwen":
        model = model or config.llm_model
        api_key = kwargs.pop("api_key", None) or config.openai_api_key
        base_url = kwargs.pop("base_url", None) or config.openai_api_base
        return DeepSeekClient(api_key=api_key, model=model, base_url=base_url, **kwargs)
    
    elif provider == "ollama":
        try:
            from src.llms.ollama_client import OllamaClient
            model = model or config.ollama_model or "llama2"
            base_url = kwargs.pop("base_url", None) or config.ollama_base_url
            return OllamaClient(model=model, base_url=base_url, **kwargs)
        except ImportError:
            logger.warning("Ollama 客户端未安装，使用 DeepSeek 客户端替代")
            return create_client("deepseek", model, **kwargs)
    
    else:
        logger.warning(f"未知的提供商 '{provider}'，使用 DeepSeek 客户端")
        return create_client("deepseek", model, **kwargs)


def create_embedding_client(
    provider: str = None,
    model: str = None,
    **kwargs,
) -> EmbeddingClient:
    """创建 Embedding 客户端
    
    Args:
        provider: 提供商名称 (local, openai, api)
        model: 模型名称
        **kwargs: 其他参数
        
    Returns:
        EmbeddingClient 实例
    """
    config = get_config()
    
    provider = provider or config.embedding_mode
    model = model or config.embedding_model
    api_key = kwargs.pop("api_key", None) or config.embed_api_key
    base_url = kwargs.pop("base_url", None) or config.embed_base_url
    device = kwargs.pop("device", None) or config.embedding_device
    
    return EmbeddingClient(
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        device=device,
        **kwargs,
    )
