# -*- coding: utf-8 -*-
"""父切片映射器

维护子切片到父切片的映射关系，支持上下文检索。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.storage.chunking_mapping.chunker import Chunk


@dataclass
class ChunkMapping:
    """切片映射关系
    
    Attributes:
        chunk_id: 切片ID
        parent_id: 父切片ID
        children_ids: 子切片ID列表
        sibling_ids: 兄弟切片ID列表
        doc_id: 文档ID
        chunk_index: 切片索引
    """
    chunk_id: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    sibling_ids: List[str] = field(default_factory=list)
    doc_id: Optional[str] = None
    chunk_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
        }
        if self.parent_id is not None:
            result["parent_id"] = self.parent_id
        if self.children_ids:
            result["children_ids"] = self.children_ids
        if self.sibling_ids:
            result["sibling_ids"] = self.sibling_ids
        if self.doc_id is not None:
            result["doc_id"] = self.doc_id
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMapping":
        """从字典创建实例"""
        return cls(
            chunk_id=data["chunk_id"],
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            sibling_ids=data.get("sibling_ids", []),
            doc_id=data.get("doc_id"),
            chunk_index=data.get("chunk_index", 0),
        )


class ChunkMapper:
    """切片映射器
    
    维护子切片到父切片的映射关系，支持上下文检索。
    
    Attributes:
        mappings: 切片ID到映射关系的字典
        chunks: 切片ID到切片对象的字典
    """
    
    def __init__(self):
        """初始化切片映射器"""
        # 切片ID到映射关系的字典
        self._mappings: Dict[str, ChunkMapping] = {}
        
        # 切片ID到切片对象的字典
        self._chunks: Dict[str, Chunk] = {}
        
        # 文档ID到切片ID列表的字典
        self._doc_chunks: Dict[str, List[str]] = {}
    
    def add_chunk(self, chunk: Chunk) -> None:
        """添加切片
        
        Args:
            chunk: 切片对象
        """
        # 存储切片
        self._chunks[chunk.id] = chunk
        
        # 创建映射关系
        mapping = ChunkMapping(
            chunk_id=chunk.id,
            parent_id=chunk.parent_id,
            children_ids=chunk.children_ids.copy(),
            doc_id=chunk.metadata.get("doc_id"),
            chunk_index=chunk.metadata.get("chunk_index", 0),
        )
        self._mappings[chunk.id] = mapping
        
        # 更新文档索引
        doc_id = chunk.metadata.get("doc_id")
        if doc_id:
            if doc_id not in self._doc_chunks:
                self._doc_chunks[doc_id] = []
            self._doc_chunks[doc_id].append(chunk.id)
        
        # 更新兄弟关系
        if mapping.parent_id:
            self._update_sibling_relations(chunk.id, mapping.parent_id)
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """批量添加切片
        
        Args:
            chunks: 切片列表
        """
        for chunk in chunks:
            self.add_chunk(chunk)
    
    def _update_sibling_relations(
        self,
        chunk_id: str,
        parent_id: str,
    ) -> None:
        """更新兄弟关系
        
        Args:
            chunk_id: 切片ID
            parent_id: 父切片ID
        """
        # 找到所有具有相同父切片的子切片
        siblings = [
            cid for cid, mapping in self._mappings.items()
            if mapping.parent_id == parent_id
        ]
        
        # 更新每个子切片的兄弟关系
        for sibling_id in siblings:
            other_siblings = [s for s in siblings if s != sibling_id]
            self._mappings[sibling_id].sibling_ids = other_siblings
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """获取切片
        
        Args:
            chunk_id: 切片ID
            
        Returns:
            切片对象，不存在则返回 None
        """
        return self._chunks.get(chunk_id)
    
    def get_mapping(self, chunk_id: str) -> Optional[ChunkMapping]:
        """获取映射关系
        
        Args:
            chunk_id: 切片ID
            
        Returns:
            映射关系，不存在则返回 None
        """
        return self._mappings.get(chunk_id)
    
    def get_parent(self, chunk_id: str) -> Optional[Chunk]:
        """获取父切片
        
        Args:
            chunk_id: 切片ID
            
        Returns:
            父切片对象，不存在则返回 None
        """
        mapping = self._mappings.get(chunk_id)
        if mapping is None or mapping.parent_id is None:
            return None
        
        return self._chunks.get(mapping.parent_id)
    
    def get_children(self, chunk_id: str) -> List[Chunk]:
        """获取子切片列表
        
        Args:
            chunk_id: 切片ID
            
        Returns:
            子切片列表
        """
        mapping = self._mappings.get(chunk_id)
        if mapping is None:
            return []
        
        return [
            self._chunks[cid] for cid in mapping.children_ids
            if cid in self._chunks
        ]
    
    def get_siblings(self, chunk_id: str) -> List[Chunk]:
        """获取兄弟切片列表
        
        Args:
            chunk_id: 切片ID
            
        Returns:
            兄弟切片列表
        """
        mapping = self._mappings.get(chunk_id)
        if mapping is None:
            return []
        
        return [
            self._chunks[cid] for cid in mapping.sibling_ids
            if cid in self._chunks
        ]
    
    def get_adjacent_chunks(
        self,
        chunk_id: str,
        num_before: int = 1,
        num_after: int = 1,
    ) -> List[Chunk]:
        """获取相邻切片
        
        Args:
            chunk_id: 切片ID
            num_before: 前面的切片数量
            num_after: 后面的切片数量
            
        Returns:
            相邻切片列表（按顺序排列）
        """
        mapping = self._mappings.get(chunk_id)
        if mapping is None:
            return []
        
        doc_id = mapping.doc_id
        if doc_id is None:
            return []
        
        # 获取文档的所有切片
        doc_chunk_ids = self._doc_chunks.get(doc_id, [])
        
        # 找到当前切片的位置
        try:
            current_index = doc_chunk_ids.index(chunk_id)
        except ValueError:
            return []
        
        # 获取相邻切片
        start_index = max(0, current_index - num_before)
        end_index = min(len(doc_chunk_ids), current_index + num_after + 1)
        
        adjacent_ids = doc_chunk_ids[start_index:end_index]
        
        return [
            self._chunks[cid] for cid in adjacent_ids
            if cid in self._chunks
        ]
    
    def get_context(
        self,
        chunk_id: str,
        include_parent: bool = True,
        include_siblings: bool = False,
        include_adjacent: bool = True,
        num_adjacent: int = 1,
    ) -> Dict[str, Any]:
        """获取切片的上下文信息
        
        Args:
            chunk_id: 切片ID
            include_parent: 是否包含父切片
            include_siblings: 是否包含兄弟切片
            include_adjacent: 是否包含相邻切片
            num_adjacent: 相邻切片数量
            
        Returns:
            上下文信息字典
        """
        chunk = self.get_chunk(chunk_id)
        if chunk is None:
            return {}
        
        context = {
            "chunk": chunk,
            "parent": None,
            "siblings": [],
            "adjacent": [],
        }
        
        # 获取父切片
        if include_parent:
            context["parent"] = self.get_parent(chunk_id)
        
        # 获取兄弟切片
        if include_siblings:
            context["siblings"] = self.get_siblings(chunk_id)
        
        # 获取相邻切片
        if include_adjacent:
            context["adjacent"] = self.get_adjacent_chunks(
                chunk_id,
                num_before=num_adjacent,
                num_after=num_adjacent,
            )
        
        return context
    
    def get_expanded_context(
        self,
        chunk_id: str,
        expansion_level: int = 1,
    ) -> str:
        """获取扩展上下文文本
        
        Args:
            chunk_id: 切片ID
            expansion_level: 扩展级别（0=仅当前切片，1=包含相邻切片，2=包含父切片）
            
        Returns:
            扩展后的上下文文本
        """
        chunk = self.get_chunk(chunk_id)
        if chunk is None:
            return ""
        
        if expansion_level == 0:
            return chunk.content
        
        elif expansion_level == 1:
            # 包含相邻切片
            adjacent = self.get_adjacent_chunks(chunk_id, num_before=1, num_after=1)
            return " ".join([c.content for c in adjacent])
        
        elif expansion_level >= 2:
            # 包含父切片
            parent = self.get_parent(chunk_id)
            if parent:
                return parent.content
            else:
                # 如果没有父切片，返回相邻切片
                adjacent = self.get_adjacent_chunks(chunk_id, num_before=1, num_after=1)
                return " ".join([c.content for c in adjacent])
        
        return chunk.content
    
    def get_chunks_by_doc(self, doc_id: str) -> List[Chunk]:
        """根据文档ID获取所有切片
        
        Args:
            doc_id: 文档ID
            
        Returns:
            切片列表
        """
        chunk_ids = self._doc_chunks.get(doc_id, [])
        return [
            self._chunks[cid] for cid in chunk_ids
            if cid in self._chunks
        ]
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """删除切片
        
        Args:
            chunk_id: 切片ID
            
        Returns:
            是否删除成功
        """
        if chunk_id not in self._chunks:
            return False
        
        chunk = self._chunks[chunk_id]
        mapping = self._mappings[chunk_id]
        
        # 从文档索引中删除
        doc_id = chunk.metadata.get("doc_id")
        if doc_id and doc_id in self._doc_chunks:
            self._doc_chunks[doc_id] = [
                cid for cid in self._doc_chunks[doc_id] if cid != chunk_id
            ]
        
        # 更新父切片的子切片列表
        if mapping.parent_id and mapping.parent_id in self._mappings:
            parent_mapping = self._mappings[mapping.parent_id]
            parent_mapping.children_ids = [
                cid for cid in parent_mapping.children_ids if cid != chunk_id
            ]
        
        # 更新兄弟关系
        for sibling_id in mapping.sibling_ids:
            if sibling_id in self._mappings:
                sibling_mapping = self._mappings[sibling_id]
                sibling_mapping.sibling_ids = [
                    sid for sid in sibling_mapping.sibling_ids if sid != chunk_id
                ]
        
        # 删除切片和映射
        del self._chunks[chunk_id]
        del self._mappings[chunk_id]
        
        return True
    
    def delete_chunks_by_doc(self, doc_id: str) -> int:
        """删除文档的所有切片
        
        Args:
            doc_id: 文档ID
            
        Returns:
            删除的切片数量
        """
        chunk_ids = self._doc_chunks.get(doc_id, [])
        count = 0
        
        for chunk_id in chunk_ids:
            if self.delete_chunk(chunk_id):
                count += 1
        
        return count
    
    def clear(self) -> None:
        """清空所有映射"""
        self._mappings.clear()
        self._chunks.clear()
        self._doc_chunks.clear()
    
    def count(self) -> int:
        """获取切片总数"""
        return len(self._chunks)
    
    def count_docs(self) -> int:
        """获取文档总数"""
        return len(self._doc_chunks)
    
    def save(self, path: str) -> None:
        """保存映射到文件
        
        Args:
            path: 保存路径
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存切片
        chunks_data = {
            chunk_id: chunk.to_dict()
            for chunk_id, chunk in self._chunks.items()
        }
        
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        # 保存映射
        mappings_data = {
            chunk_id: mapping.to_dict()
            for chunk_id, mapping in self._mappings.items()
        }
        
        with open(path / "mappings.json", "w", encoding="utf-8") as f:
            json.dump(mappings_data, f, ensure_ascii=False, indent=2)
        
        # 保存文档索引
        with open(path / "doc_chunks.json", "w", encoding="utf-8") as f:
            json.dump(self._doc_chunks, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str) -> None:
        """从文件加载映射
        
        Args:
            path: 加载路径
        """
        path = Path(path)
        
        # 清空现有数据
        self.clear()
        
        # 加载切片
        chunks_file = path / "chunks.json"
        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            
            for chunk_id, chunk_dict in chunks_data.items():
                self._chunks[chunk_id] = Chunk.from_dict(chunk_dict)
        
        # 加载映射
        mappings_file = path / "mappings.json"
        if mappings_file.exists():
            with open(mappings_file, "r", encoding="utf-8") as f:
                mappings_data = json.load(f)
            
            for chunk_id, mapping_dict in mappings_data.items():
                self._mappings[chunk_id] = ChunkMapping.from_dict(mapping_dict)
        
        # 加载文档索引
        doc_chunks_file = path / "doc_chunks.json"
        if doc_chunks_file.exists():
            with open(doc_chunks_file, "r", encoding="utf-8") as f:
                self._doc_chunks = json.load(f)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        # 统计有父切片的切片数量
        chunks_with_parent = sum(
            1 for m in self._mappings.values() if m.parent_id is not None
        )
        
        # 统计有子切片的切片数量
        chunks_with_children = sum(
            1 for m in self._mappings.values() if m.children_ids
        )
        
        return {
            "total_chunks": self.count(),
            "total_docs": self.count_docs(),
            "chunks_with_parent": chunks_with_parent,
            "chunks_with_children": chunks_with_children,
            "avg_chunks_per_doc": self.count() / max(1, self.count_docs()),
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"chunks={self.count()}, "
            f"docs={self.count_docs()})"
        )


if __name__ == "__main__":
    from src.storage.chunking_mapping.chunker import Chunk
    
    print("=" * 50)
    print("测试 Chunk Mapper 模块")
    print("=" * 50)
    
    # 测试 1: 创建映射器
    mapper = ChunkMapper()
    print(f"✓ 创建映射器: {mapper}")
    
    # 测试 2: 添加切片
    chunks = [
        Chunk(
            id="doc_001_chunk_0",
            content="这是第一个切片的内容。",
            metadata={"doc_id": "doc_001", "chunk_index": 0},
        ),
        Chunk(
            id="doc_001_chunk_1",
            content="这是第二个切片的内容。",
            metadata={"doc_id": "doc_001", "chunk_index": 1},
        ),
        Chunk(
            id="doc_001_chunk_2",
            content="这是第三个切片的内容。",
            metadata={"doc_id": "doc_001", "chunk_index": 2},
            parent_id="doc_001_parent",
        ),
    ]
    mapper.add_chunks(chunks)
    print(f"✓ 添加切片: count={mapper.count()}")
    
    # 测试 3: 获取切片
    chunk = mapper.get_chunk("doc_001_chunk_0")
    print(f"✓ 获取切片: id={chunk.id if chunk else 'None'}")
    
    # 测试 4: 获取相邻切片
    adjacent = mapper.get_adjacent_chunks("doc_001_chunk_1", num_before=1, num_after=1)
    print(f"✓ 获取相邻切片: {len(adjacent)} 个")
    
    # 测试 5: 获取上下文
    context = mapper.get_context("doc_001_chunk_1")
    print(f"✓ 获取上下文: adjacent={len(context.get('adjacent', []))}")
    
    # 测试 6: 获取文档切片
    doc_chunks = mapper.get_chunks_by_doc("doc_001")
    print(f"✓ 获取文档切片: doc_001 有 {len(doc_chunks)} 个切片")
    
    # 测试 7: 统计信息
    stats = mapper.get_stats()
    print(f"✓ 统计信息: chunks={stats['total_chunks']}, docs={stats['total_docs']}")
    
    # 测试 8: 删除切片
    deleted = mapper.delete_chunk("doc_001_chunk_0")
    print(f"✓ 删除切片: success={deleted}, remaining={mapper.count()}")
    
    print("\n所有测试通过!")
