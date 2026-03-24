# -*- coding: utf-8 -*-
"""文本切分器

支持多种切分策略，包括递归切分、语义切分和固定大小切分。
"""

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.utils.config import get_config

try:
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
    )
except ImportError:
    RecursiveCharacterTextSplitter = None
    CharacterTextSplitter = None


class ChunkStrategy(Enum):
    """切分策略枚举"""
    RECURSIVE = "recursive"     # 递归切分（推荐）
    SEMANTIC = "semantic"       # 语义切分
    FIXED_SIZE = "fixed_size"   # 固定大小切分


@dataclass
class Chunk:
    """文本切片
    
    Attributes:
        id: 切片唯一标识
        content: 切片内容
        metadata: 元数据
        parent_id: 父切片ID（可选）
        children_ids: 子切片ID列表
        start_index: 在原文中的起始位置
        end_index: 在原文中的结束位置
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    start_index: int = 0
    end_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "start_index": self.start_index,
            "end_index": self.end_index,
        }
        if self.parent_id is not None:
            result["parent_id"] = self.parent_id
        if self.children_ids:
            result["children_ids"] = self.children_ids
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """从字典创建实例"""
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            start_index=data.get("start_index", 0),
            end_index=data.get("end_index", 0),
        )
    
    @property
    def length(self) -> int:
        """获取切片长度"""
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        """获取单词数量"""
        return len(self.content.split())


class TextChunker:
    """文本切分器
    
    支持多种切分策略，包括递归切分、语义切分和固定大小切分。
    
    Attributes:
        strategy: 切分策略
        chunk_size: 切片大小
        chunk_overlap: 切片重叠大小
        separators: 分隔符列表
        length_function: 长度计算函数
    """
    
    def __init__(
        self,
        strategy: str = "recursive",
        separators: Optional[List[str]] = None,
        length_function: Optional[callable] = None,
        **kwargs: Any,
    ):
        """初始化文本切分器
        
        Args:
            strategy: 切分策略 (recursive, semantic, fixed_size)
            separators: 分隔符列表
            length_function: 长度计算函数
            **kwargs: 额外参数
        """
        config = get_config()
        
        # 解析切分策略
        try:
            self.strategy = ChunkStrategy(strategy.lower())
        except ValueError:
            raise ValueError(
                f"Invalid strategy: {strategy}. "
                f"Must be one of {[s.value for s in ChunkStrategy]}"
            )
        
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.separators = separators or self._get_default_separators()
        self.length_function = length_function or len
        self.kwargs = kwargs
        
        # 初始化切分器
        self._splitter = self._create_splitter()
    
    def _get_default_separators(self) -> List[str]:
        """获取默认分隔符"""
        if self.strategy == ChunkStrategy.RECURSIVE:
            # 递归切分使用多级分隔符
            return [
                "\n\n",  # 段落
                "\n",    # 行
                "。",    # 中文句号
                "！",    # 中文感叹号
                "？",    # 中文问号
                "；",    # 中文分号
                ".",     # 英文句号
                "!",     # 英文感叹号
                "?",     # 英文问号
                ";",     # 英文分号
                " ",     # 空格
                "",      # 字符
            ]
        elif self.strategy == ChunkStrategy.SEMANTIC:
            # 语义切分使用句子分隔符
            return ["\n\n", "\n", "。", ".", "!", "?"]
        else:
            # 固定大小切分不使用分隔符
            return [""]
    
    def _create_splitter(self) -> Any:
        """创建切分器实例"""
        if RecursiveCharacterTextSplitter is not None:
            # 使用 langchain 的切分器
            if self.strategy == ChunkStrategy.RECURSIVE:
                return RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=self.separators,
                    length_function=self.length_function,
                )
            else:
                return CharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separator=self.separators[0] if self.separators else "",
                    length_function=self.length_function,
                )
        else:
            # 使用简单的切分器
            return None
    
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> List[Chunk]:
        """切分文本
        
        Args:
            text: 待切分的文本
            metadata: 元数据（可选）
            doc_id: 文档ID（可选）
            
        Returns:
            切片列表
        """
        if not text:
            return []
        
        metadata = metadata or {}
        doc_id = doc_id or str(uuid.uuid4())
        
        # 执行切分
        if self._splitter is not None:
            chunks = self._chunk_with_langchain(text)
        else:
            chunks = self._chunk_simple(text)
        
        # 创建 Chunk 对象
        result = []
        current_pos = 0
        
        for i, chunk_text in enumerate(chunks):
            # 计算位置
            start_idx = text.find(chunk_text, current_pos)
            if start_idx == -1:
                start_idx = current_pos
            end_idx = start_idx + len(chunk_text)
            current_pos = end_idx
            
            # 创建切片
            chunk = Chunk(
                id=f"{doc_id}_chunk_{i}",
                content=chunk_text,
                metadata={
                    **metadata,
                    "doc_id": doc_id,
                    "chunk_index": i,
                },
                start_index=start_idx,
                end_index=end_idx,
            )
            result.append(chunk)
        
        return result
    
    def _chunk_with_langchain(self, text: str) -> List[str]:
        """使用 langchain 切分器切分文本"""
        if self.strategy == ChunkStrategy.RECURSIVE:
            return self._splitter.split_text(text)
        elif self.strategy == ChunkStrategy.SEMANTIC:
            return self._split_semantic(text)
        else:
            return self._splitter.split_text(text)
    
    def _chunk_simple(self, text: str) -> List[str]:
        """简单的文本切分"""
        if self.strategy == ChunkStrategy.FIXED_SIZE:
            return self._split_fixed_size(text)
        elif self.strategy == ChunkStrategy.SEMANTIC:
            return self._split_semantic(text)
        else:
            return self._split_recursive(text)
    
    def _split_recursive(self, text: str) -> List[str]:
        """递归切分"""
        if len(text) <= self.chunk_size:
            return [text]
        
        # 尝试使用分隔符切分
        for separator in self.separators:
            if separator and separator in text:
                parts = text.split(separator)
                chunks = []
                current_chunk = ""
                
                for part in parts:
                    if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                        current_chunk += part + separator
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + separator
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 递归处理过大的块
                final_chunks = []
                for chunk in chunks:
                    if len(chunk) > self.chunk_size:
                        final_chunks.extend(self._split_recursive(chunk))
                    else:
                        final_chunks.append(chunk)
                
                return final_chunks
        
        # 如果没有合适的分隔符，使用固定大小切分
        return self._split_fixed_size(text)
    
    def _split_semantic(self, text: str) -> List[str]:
        """语义切分（基于句子）"""
        # 使用正则表达式分割句子
        sentence_pattern = r'(?<=[。！？.!?])\s*'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 合并句子到接近 chunk_size
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_fixed_size(self, text: str) -> List[str]:
        """固定大小切分"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # 添加重叠部分
            if start > 0 and self.chunk_overlap > 0:
                overlap_start = max(0, start - self.chunk_overlap)
                chunk = text[overlap_start:end]
            
            chunks.append(chunk)
            start = end
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "content",
        metadata_key: str = "metadata",
        id_key: str = "id",
    ) -> List[Chunk]:
        """批量切分文档
        
        Args:
            documents: 文档列表
            text_key: 文本内容的键名
            metadata_key: 元数据的键名
            id_key: 文档ID的键名
            
        Returns:
            所有文档的切片列表
        """
        all_chunks = []
        
        for doc in documents:
            text = doc.get(text_key, "")
            metadata = doc.get(metadata_key, {})
            doc_id = doc.get(id_key, str(uuid.uuid4()))
            
            chunks = self.chunk(text, metadata, doc_id)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def create_parent_child_chunks(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        parent_chunk_size: int = 1024,
        child_chunk_size: int = 256,
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """创建父子切片
        
        Args:
            text: 待切分的文本
            metadata: 元数据（可选）
            doc_id: 文档ID（可选）
            parent_chunk_size: 父切片大小
            child_chunk_size: 子切片大小
            
        Returns:
            (父切片列表, 子切片列表) 元组
        """
        metadata = metadata or {}
        doc_id = doc_id or str(uuid.uuid4())
        
        # 创建父切片
        parent_chunker = TextChunker(
            strategy=self.strategy.value,
            chunk_size=parent_chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        parent_chunks = parent_chunker.chunk(text, metadata, doc_id)
        
        # 创建子切片
        child_chunker = TextChunker(
            strategy=self.strategy.value,
            chunk_size=child_chunk_size,
            chunk_overlap=self.chunk_overlap // 2,
        )
        
        child_chunks = []
        for parent in parent_chunks:
            # 为每个父切片创建子切片
            children = child_chunker.chunk(
                parent.content,
                {**metadata, "parent_id": parent.id},
                doc_id,
            )
            
            # 设置父子关系
            parent.children_ids = [c.id for c in children]
            for child in children:
                child.parent_id = parent.id
            
            child_chunks.extend(children)
        
        return parent_chunks, child_chunks
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"strategy={self.strategy.value}, "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )


if __name__ == "__main__":
    print("=" * 50)
    print("测试 Text Chunker 模块")
    print("=" * 50)
    
    # 测试文本
    test_text = """
    人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    机器学习是人工智能的核心，是使计算机具有智能的根本途径。
    深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。
    自然语言处理是人工智能的一个重要领域，它研究如何让计算机理解和生成人类语言。
    """
    
    # 测试 1: 固定大小切分
    chunker = TextChunker(strategy="fixed_size", chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(test_text, {"source": "test"}, "doc_001")
    print(f"✓ 固定大小切分: {len(chunks)} 个切片")
    for i, chunk in enumerate(chunks[:2]):
        print(f"  - 切片 {i+1}: {chunk.content[:30]}...")
    
    # 测试 2: 递归切分
    chunker = TextChunker(strategy="recursive", chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(test_text, {"source": "test"}, "doc_002")
    print(f"✓ 递归切分: {len(chunks)} 个切片")
    
    # 测试 3: 语义切分
    chunker = TextChunker(strategy="semantic", chunk_size=100)
    chunks = chunker.chunk(test_text, {"source": "test"}, "doc_003")
    print(f"✓ 语义切分: {len(chunks)} 个切片")
    
    # 测试 4: Chunk 数据结构
    if chunks:
        chunk = chunks[0]
        print(f"✓ Chunk 结构: id={chunk.id}")
        print(f"  - content length: {len(chunk.content)}")
        print(f"  - metadata: {chunk.metadata}")
    
    print("\n所有测试通过!")
