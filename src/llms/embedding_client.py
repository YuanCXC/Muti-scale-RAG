# -*- coding: utf-8 -*-
"""Embedding 客户端模块

提供文本嵌入向量的生成功能，支持本地模型和 API。
"""

from typing import Any, Dict, List, Optional, Union, Iterator
import time

import numpy as np
from openai import OpenAI

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingClient:
    """Embedding 客户端
    
    支持多种 embedding 提供商，包括本地模型和 API。
    """
    
    def __init__(
        self,
        model: str = None,
        provider: str = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        """初始化 Embedding 客户端
        
        Args:
            model: 模型名称
            provider: 提供商 (local, openai, api, auto)
            api_key: API 密钥（用于 API 提供商）
            base_url: API 基础 URL
            device: 设备 (cuda, cpu, auto)
            **kwargs: 其他参数
        """
        config = get_config()
        
        self.model = model or config.embedding_model
        self.provider = provider or config.embedding_mode
        self.api_key = api_key or config.embed_api_key
        self.base_url = base_url or config.embed_base_url
        self.kwargs = kwargs
        
        self._model = None
        self._dimension = None
        
        if device is None:
            device = config.embedding_device or "auto"
        
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
        
        self.use_fp16 = kwargs.get("use_fp16", config.embedding_use_fp16)
        
        self._init_model()
    
    def _init_model(self) -> None:
        """初始化 embedding 模型"""
        if self.provider == "local" or self.provider == "bge" or self.provider == "auto":
            try:
                from sentence_transformers import SentenceTransformer
                
                logger.info(f"正在加载 Embedding 模型: {self.model}")
                self._model = SentenceTransformer(self.model, device=self.device)
                self._dimension = self._model.get_sentence_embedding_dimension()
                
                logger.info(
                    f"Embedding 模型加载成功: model={self.model}, "
                    f"dimension={self._dimension}, device={self.device}"
                )
                return
            except ImportError:
                logger.warning(
                    "未安装 sentence-transformers 库。"
                    "请运行: pip install sentence-transformers"
                )
                if self.provider == "auto":
                    logger.info("尝试使用 API 模式...")
                    self.provider = "api"
                else:
                    self._dimension = 1024
                    return
            except Exception as e:
                logger.warning(f"本地模型加载失败: {e}")
                if self.provider == "auto":
                    logger.info("尝试使用 API 模式...")
                    self.provider = "api"
                else:
                    self._dimension = 1024
                    return
        
        if self.provider == "api" or self.provider == "openai":
            self._dimension = 1024
            logger.info(f"使用 API Embedding: model={self.model}, base_url={self.base_url}")
    
    @property
    def dimension(self) -> int:
        """获取 embedding 维度"""
        return self._dimension or 1024
    
    def embed(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """生成文本的 embedding 向量
        
        Args:
            texts: 单个文本或文本列表
            normalize: 是否归一化向量
            
        Returns:
            embedding 向量数组
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.provider == "local" or self.provider == "bge" or self.provider == "auto":
            if self._model is not None:
                return self._embed_local(texts, normalize)
            else:
                logger.warning("本地模型未加载，尝试使用 API")
                return self._embed_api(texts, normalize)
        elif self.provider == "api" or self.provider == "openai":
            return self._embed_api(texts, normalize)
        else:
            raise ValueError(f"未知的 provider: {self.provider}")
    
    def _embed_local(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """使用本地模型生成 embedding"""
        if self._model is None:
            logger.warning("模型未加载，返回零向量")
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
        
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        
        return embeddings.astype(np.float32)
    
    def _embed_api(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """使用 API 生成 embedding"""
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            
            all_embeddings = []
            
            for text in texts:
                response = client.embeddings.create(
                    model=self.model,
                    input=text,
                )
                all_embeddings.append(response.data[0].embedding)
            
            embeddings = np.array(all_embeddings, dtype=np.float32)
            
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-8)
            
            logger.debug(f"API Embedding 完成: shape={embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"API Embedding 失败: {e}")
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
    
    def similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度分数 (0-1)
        """
        embeddings = self.embed([text1, text2], normalize=True)
        
        similarity = np.dot(embeddings[0], embeddings[1])
        
        return float(similarity)
    
    def batch_similarity(
        self,
        query: str,
        texts: List[str],
    ) -> List[float]:
        """计算查询与多个文本的相似度
        
        Args:
            query: 查询文本
            texts: 文本列表
            
        Returns:
            相似度分数列表
        """
        all_texts = [query] + texts
        embeddings = self.embed(all_texts, normalize=True)
        
        query_embedding = embeddings[0]
        text_embeddings = embeddings[1:]
        
        similarities = np.dot(text_embeddings, query_embedding)
        
        return similarities.tolist()


if __name__ == "__main__":
    print("=" * 50)
    print("测试 Embedding 客户端")
    print("=" * 50)
    
    client = EmbeddingClient()
    
    print(f"✓ 模型: {client.model}")
    print(f"✓ Provider: {client.provider}")
    print(f"✓ Embedding 维度: {client.dimension}")
    print(f"✓ 设备: {client.device}")
    
    texts = ["你好世界", "人工智能"]
    embeddings = client.embed(texts)
    
    print(f"✓ Embedding 形状: {embeddings.shape}")
    print(f"✓ Embedding 类型: {embeddings.dtype}")
    
    similarity = client.similarity("你好", "你好吗")
    print(f"✓ 相似度: {similarity:.4f}")
    
    print("\n测试完成!")
