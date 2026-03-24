# -*- coding: utf-8 -*-
"""分块映射模块

提供文本切分和切片映射功能。
"""

from .chunker import (
    Chunk,
    ChunkStrategy,
    TextChunker,
)
from .mapper import (
    ChunkMapping,
    ChunkMapper,
)

__all__ = [
    # 切分器
    "TextChunker",
    "Chunk",
    "ChunkStrategy",
    # 映射器
    "ChunkMapper",
    "ChunkMapping",
]
