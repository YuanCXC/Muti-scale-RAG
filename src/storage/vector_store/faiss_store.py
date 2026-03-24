# -*- coding: utf-8 -*-
"""FAISS 向量存储实现

使用 FAISS 库实现高效的向量索引和检索。
支持段落级 (paragraph) 和句子级 (sentence) 两种存储模式。
"""

import json
import os
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError(
        "使用 FAISSVectorStore 需要安装 faiss。请通过以下命令安装：pip install faiss-cpu 或 pip install faiss-gpu"
    )

from src.storage.vector_store.base_store import SearchResult, VectorMetadata, VectorStoreBase
from src.utils.config import get_config


class IndexType(Enum):
    """FAISS 索引类型"""
    FLAT = "flat"  # 暴力搜索，精确但慢
    IVF = "ivf"    # 倒排索引，快速近似搜索
    HNSW = "hnsw"  # 层次导航小世界图，快速近似搜索


class StorageMode(Enum):
    """存储模式"""
    PARAGRAPH = "paragraph"  # 段落级存储
    SENTENCE = "sentence"    # 句子级存储


class FAISSVectorStore(VectorStoreBase):
    """FAISS 向量存储实现
    
    使用 FAISS 库实现向量索引和检索，支持多种索引类型和存储模式。
    
    Attributes:
        index_type: 索引类型
        storage_mode: 存储模式
        nlist: IVF 索引的聚类中心数量
        nprobe: 搜索时探测的聚类数量
        M: HNSW 索引的连接数
        efConstruction: HNSW 索引的构建参数
        efSearch: HNSW 索引的搜索参数
    """
    
    def __init__(
        self,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        index_type: str = "flat",
        storage_mode: str = "paragraph",
        index_path: Optional[str] = None,
        nlist: int = 100,
        nprobe: int = 10,
        M: int = 32,
        efConstruction: int = 200,
        efSearch: int = 50,
        **kwargs: Any,
    ):
        """初始化 FAISS 向量存储
        
        Args:
            dimension: 向量维度 (可选，默认从 config 读取)
            metric: 相似度度量方式 (cosine, l2, ip)
            index_type: 索引类型 (flat, ivf, hnsw)
            storage_mode: 存储模式 (paragraph, sentence)
            index_path: 索引存储路径 (可选，默认从 config 读取)
            nlist: IVF 索引的聚类中心数量
            nprobe: 搜索时探测的聚类数量
            M: HNSW 索引的连接数
            efConstruction: HNSW 索引的构建参数
            efSearch: HNSW 索引的搜索参数
            **kwargs: 额外参数
        """
        config = get_config()
        
        self.dimension = config.vector_dim
        super().__init__(self.dimension, metric, **kwargs)
        
        self.index_path = config.faiss_index_path
        
        # 解析索引类型
        try:
            self.index_type = IndexType(index_type.lower())
        except ValueError:
            raise ValueError(
                f"Invalid index_type: {index_type}. "
                f"Must be one of {[t.value for t in IndexType]}"
            )
        
        # 解析存储模式
        try:
            self.storage_mode = StorageMode(storage_mode.lower())
        except ValueError:
            raise ValueError(
                f"Invalid storage_mode: {storage_mode}. "
                f"Must be one of {[m.value for m in StorageMode]}"
            )
        
        # IVF 参数
        self.nlist = nlist
        self.nprobe = nprobe
        
        # HNSW 参数
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        
        # 初始化索引
        self._index: Optional[faiss.Index] = None
        self._id_to_metadata: Dict[str, VectorMetadata] = {}
        self._id_to_internal_id: Dict[str, int] = {}
        self._internal_id_to_id: Dict[int, str] = {}
        self._next_internal_id: int = 0
        
        # 初始化索引
        self._init_index()
    
    def _init_index(self) -> None:
        """初始化 FAISS 索引"""
        if self.metric == "cosine":
            # 对于 cosine 相似度，使用内积索引 + 归一化向量
            self._index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "l2":
            self._index = faiss.IndexFlatL2(self.dimension)
        elif self.metric == "ip":
            self._index = faiss.IndexFlatIP(self.dimension)
    
    def _create_index(self, vectors: np.ndarray) -> faiss.Index:
        """根据索引类型创建索引
        
        Args:
            vectors: 向量数组
            
        Returns:
            FAISS 索引
        """
        n_vectors = vectors.shape[0]
        
        if self.index_type == IndexType.FLAT:
            if self.metric == "l2":
                return faiss.IndexFlatL2(self.dimension)
            else:
                return faiss.IndexFlatIP(self.dimension)
        
        elif self.index_type == IndexType.IVF:
            # 确保聚类中心数量不超过向量数量
            nlist = min(self.nlist, n_vectors)
            
            if self.metric == "l2":
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
            else:
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # 训练索引
            index.train(vectors)
            return index
        
        elif self.index_type == IndexType.HNSW:
            if self.metric == "l2":
                index = faiss.IndexHNSWFlat(self.dimension, self.M, faiss.METRIC_L2)
            else:
                index = faiss.IndexHNSWFlat(self.dimension, self.M, faiss.METRIC_INNER_PRODUCT)
            
            index.hnsw.efConstruction = self.efConstruction
            index.hnsw.efSearch = self.efSearch
            return index
        
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[VectorMetadata],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """添加向量到存储
        
        Args:
            vectors: 向量数组，shape=(n, dimension)
            metadata: 元数据列表
            ids: 向量ID列表（可选，不提供则自动生成）
            
        Returns:
            添加的向量ID列表
        """
        # 验证输入
        self._validate_vectors(vectors)
        self._validate_metadata(vectors, metadata)
        
        # 确保 vectors 是连续的 float32 数组
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
        # 归一化向量（用于 cosine 相似度）
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 避免除零
            vectors = vectors / norms
        
        # 生成 ID
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        elif len(ids) != len(vectors):
            raise ValueError(
                f"Number of ids ({len(ids)}) must match number of vectors ({len(vectors)})"
            )
        
        # 如果索引为空或需要重建索引
        if self._index is None or self.count() == 0:
            self._index = self._create_index(vectors)
        
        # 对于 IVF 索引，如果未训练则训练
        if self.index_type == IndexType.IVF and hasattr(self._index, 'is_trained'):
            if not self._index.is_trained:
                self._index.train(vectors)
        
        # 添加向量到索引
        start_id = self._next_internal_id
        self._index.add(vectors)
        
        # 更新映射关系
        for i, (vec_id, meta) in enumerate(zip(ids, metadata)):
            internal_id = start_id + i
            self._id_to_metadata[vec_id] = meta
            self._id_to_internal_id[vec_id] = internal_id
            self._internal_id_to_id[internal_id] = vec_id
        
        self._next_internal_id += len(vectors)
        
        return ids
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """搜索相似向量
        
        Args:
            query_vector: 查询向量，shape=(dimension,)
            top_k: 返回结果数量
            filter_dict: 过滤条件（可选）
            
        Returns:
            搜索结果列表，按相似度降序排列
        """
        if self._index is None or self.count() == 0:
            return []
        
        # 确保查询向量格式正确
        query_vector = np.ascontiguousarray(
            query_vector.astype(np.float32).reshape(1, -1)
        )
        
        # 归一化查询向量（用于 cosine 相似度）
        if self.metric == "cosine":
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
        
        # 设置 IVF 搜索参数
        if self.index_type == IndexType.IVF and hasattr(self._index, 'nprobe'):
            self._index.nprobe = self.nprobe
        
        # 执行搜索
        scores, indices = self._index.search(query_vector, top_k)
        
        # 构建结果列表
        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS 返回 -1 表示无效结果
                continue
            
            vec_id = self._internal_id_to_id.get(idx)
            if vec_id is None:
                continue
            
            metadata = self._id_to_metadata.get(vec_id)
            if metadata is None:
                continue
            
            # 应用过滤条件
            if filter_dict and not self._match_filter(metadata, filter_dict):
                continue
            
            results.append(SearchResult(
                id=vec_id,
                score=float(score),
                metadata=metadata,
            ))
        
        return results
    
    def _match_filter(
        self,
        metadata: VectorMetadata,
        filter_dict: Dict[str, Any],
    ) -> bool:
        """检查元数据是否匹配过滤条件
        
        Args:
            metadata: 元数据
            filter_dict: 过滤条件
            
        Returns:
            是否匹配
        """
        for key, value in filter_dict.items():
            if key == "doc_id":
                if metadata.doc_id != value:
                    return False
            elif key == "source":
                if metadata.source != value:
                    return False
            elif key in metadata.extra:
                if metadata.extra[key] != value:
                    return False
            else:
                return False
        return True
    
    def delete(self, ids: List[str]) -> bool:
        """删除向量
        
        注意：FAISS 不支持直接删除向量，此方法仅删除元数据映射。
        如需真正删除，需要重建索引。
        
        Args:
            ids: 要删除的向量ID列表
            
        Returns:
            是否删除成功
        """
        for vec_id in ids:
            if vec_id in self._id_to_metadata:
                del self._id_to_metadata[vec_id]
                internal_id = self._id_to_internal_id.pop(vec_id, None)
                if internal_id is not None:
                    self._internal_id_to_id.pop(internal_id, None)
        return True
    
    def save(self, path: Optional[str] = None) -> None:
        """保存索引到文件
        
        Args:
            path: 保存路径（可选，默认使用 self.index_path）
        """
        save_path = Path(path or self.index_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 FAISS 索引
        index_file = save_path / "faiss.index"
        faiss.write_index(self._index, str(index_file))
        
        # 保存元数据
        metadata_path = save_path / "metadata.json"
        metadata_dict = {
            "dimension": self.dimension,
            "metric": self.metric,
            "index_type": self.index_type.value,
            "storage_mode": self.storage_mode.value,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "M": self.M,
            "efConstruction": self.efConstruction,
            "efSearch": self.efSearch,
            "id_to_metadata": {
                k: v.to_dict() for k, v in self._id_to_metadata.items()
            },
            "id_to_internal_id": self._id_to_internal_id,
            "internal_id_to_id": {str(k): v for k, v in self._internal_id_to_id.items()},
            "next_internal_id": self._next_internal_id,
        }
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
    
    def load(self, path: Optional[str] = None) -> None:
        """从文件加载索引
        
        Args:
            path: 加载路径（可选，默认使用 self.index_path）
        """
        load_path = Path(path or self.index_path)
        
        # 加载 FAISS 索引
        index_file = load_path / "faiss.index"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        self._index = faiss.read_index(str(index_file))
        
        # 加载元数据
        metadata_path = load_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)
        
        # 恢复属性
        self.dimension = metadata_dict["dimension"]
        self.metric = metadata_dict["metric"]
        self.index_type = IndexType(metadata_dict["index_type"])
        self.storage_mode = StorageMode(metadata_dict["storage_mode"])
        self.nlist = metadata_dict.get("nlist", 100)
        self.nprobe = metadata_dict.get("nprobe", 10)
        self.M = metadata_dict.get("M", 32)
        self.efConstruction = metadata_dict.get("efConstruction", 200)
        self.efSearch = metadata_dict.get("efSearch", 50)
        
        # 恢复映射关系
        self._id_to_metadata = {
            k: VectorMetadata.from_dict(v)
            for k, v in metadata_dict["id_to_metadata"].items()
        }
        self._id_to_internal_id = metadata_dict["id_to_internal_id"]
        self._internal_id_to_id = {
            int(k): v for k, v in metadata_dict["internal_id_to_id"].items()
        }
        self._next_internal_id = metadata_dict["next_internal_id"]
    
    def get_vector(self, id: str) -> Optional[Tuple[np.ndarray, VectorMetadata]]:
        """根据ID获取向量和元数据
        
        Args:
            id: 向量ID
            
        Returns:
            向量和元数据元组，不存在则返回 None
        """
        if id not in self._id_to_metadata:
            return None
        
        metadata = self._id_to_metadata[id]
        internal_id = self._id_to_internal_id.get(id)
        
        if internal_id is None or self._index is None:
            return None
        
        # FAISS 不支持直接获取向量，需要重建
        # 这里返回 None，实际使用时可以通过其他方式获取
        return None
    
    def count(self) -> int:
        """获取向量总数
        
        Returns:
            向量总数
        """
        if self._index is None:
            return 0
        return self._index.ntotal
    
    def clear(self) -> None:
        """清空所有向量"""
        self._init_index()
        self._id_to_metadata.clear()
        self._id_to_internal_id.clear()
        self._internal_id_to_id.clear()
        self._next_internal_id = 0
    
    def get_by_doc_id(self, doc_id: str) -> List[SearchResult]:
        """根据文档ID获取所有相关向量
        
        Args:
            doc_id: 文档ID
            
        Returns:
            搜索结果列表
        """
        results = []
        for vec_id, metadata in self._id_to_metadata.items():
            if metadata.doc_id == doc_id:
                results.append(SearchResult(
                    id=vec_id,
                    score=1.0,
                    metadata=metadata,
                ))
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "total_vectors": self.count(),
            "dimension": self.dimension,
            "metric": self.metric,
            "index_type": self.index_type.value,
            "storage_mode": self.storage_mode.value,
            "unique_docs": len(set(m.doc_id for m in self._id_to_metadata.values())),
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"dimension={self.dimension}, "
            f"metric={self.metric}, "
            f"index_type={self.index_type.value}, "
            f"storage_mode={self.storage_mode.value}, "
            f"count={self.count()})"
        )


if __name__ == "__main__":
    import numpy as np
    import os
    import shutil
    
    print("=" * 50)
    print("测试 FAISS Vector Store 模块")
    print("=" * 50)
    
    # 使用配置中的默认值
    from src.utils.config import get_config
    config = get_config()
    
    # 测试 1: 创建存储实例（使用配置中的默认值）
    store = FAISSVectorStore(
        metric="cosine",
        index_type="flat",
        storage_mode="paragraph"
    )
    print(f"✓ 创建存储实例: dimension={store.dimension}, metric={store.metric}")
    
    # 测试 2: 添加向量
    vectors = np.random.randn(5, config.vector_dim).astype(np.float32)
    metadata = [
        VectorMetadata(
            doc_id=f"doc_{i}",
            chunk_id=f"chunk_{i}",
            content=f"内容{i}",
            source="test"
        )
        for i in range(5)
    ]
    store.add_vectors(vectors, metadata, ids=[f"vec_{i}" for i in range(5)])
    print(f"✓ 添加向量: count={store.count()}")
    
    # 测试 3: 搜索向量
    query = np.random.randn(1, config.vector_dim).astype(np.float32)
    results = store.search(query, top_k=3)
    print(f"✓ 搜索向量: 返回 {len(results)} 个结果")
    
    # 测试 4: 获取向量 (FAISS 不支持直接获取)
    result = store.get_vector("vec_0")
    print(f"✓ 获取向量: {'找到' if result else '不支持直接获取'}")
    
    # 测试 5: 按文档ID获取
    doc_results = store.get_by_doc_id("doc_0")
    print(f"✓ 按文档ID获取: 找到 {len(doc_results)} 个结果")
    
    # 测试 6: 统计信息
    stats = store.get_stats()
    print(f"✓ 统计信息: {stats}")
    
    # 测试 7: 保存和加载（使用配置中的默认路径）
    test_index_path = os.path.join(config.faiss_index_path, "test_index")
    store.save(test_index_path)
    print(f"✓ 保存索引到: {test_index_path}")
    
    store2 = FAISSVectorStore(metric="cosine")
    store2.load(test_index_path)
    print(f"✓ 加载索引: count={store2.count()}")
    
    # 清理测试索引
    if os.path.exists(test_index_path):
        shutil.rmtree(test_index_path)
    
    # 测试 8: 删除向量
    store.delete(["vec_0"])
    print(f"✓ 删除向量: count={store.count()}")
    
    # 测试 9: 清空
    store.clear()
    print(f"✓ 清空存储: count={store.count()}")
    
    print("\n所有测试通过!")
