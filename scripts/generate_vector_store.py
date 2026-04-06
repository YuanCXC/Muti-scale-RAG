# -*- coding: utf-8 -*-
"""生成向量存储脚本

从 valid_title_sentence.json 生成句子级和段落级向量存储
支持断点续跑，增量保存
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pickle

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage.vector_store.faiss_store import FAISSVectorStore, VectorMetadata
from src.llms.embedding_client import EmbeddingClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.pkl"
    
    def save(self, data: Dict[str, Any]) -> None:
        """保存检查点"""
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"检查点已保存: batch_idx={data.get('batch_idx', 0)}")
    
    def load(self) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"从检查点恢复: batch_idx={data.get('batch_idx', 0)}")
            return data
        return None
    
    def clear(self) -> None:
        """清除检查点"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("检查点已清除")


def load_source_data(file_path: str, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载源数据（支持限制数量）"""
    logger.info(f"加载数据: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_items:
        data = data[:max_items]
    
    logger.info(f"加载完成: {len(data)} 条数据")
    return data


def create_paragraph_documents(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """创建段落级文档列表"""
    documents = []
    
    for idx, item in enumerate(data):
        doc = {
            "doc_id": f"doc_{idx}",
            "content": item.get("sentence_total", item.get("sentence", "")),
            "metadata": {
                "title": item.get("title", ""),
                "subject": item.get("subject", ""),
                "source": "hotpotqa",
                "triplets": item.get("triplets", [])
            }
        }
        documents.append(doc)
    
    return documents


def create_sentence_documents(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """创建句子级文档列表"""
    documents = []
    
    for idx, item in enumerate(data):
        title = item.get("title", "")
        title_id = str(idx + 1)
        sentence_total = item.get("sentence_total", "")
        sentences = sentence_total.split(". ")
        subject = item.get("subject", "")
        triplets = item.get("triplets", [])
        
        for sent_idx, sentence in enumerate(sentences, start=1):
            if sentence.strip():
                sentence_id = f"{title_id}.{sent_idx}"
                doc = {
                    "doc_id": f"doc_{idx}_sent_{sent_idx}",
                    "content": sentence.strip(),
                    "metadata": {
                        "title_id": title_id,
                        "title": title,
                        "sentence_id": sentence_id,
                        "sentence": sentence.strip(),
                        "subject": subject,
                        "source": "hotpotqa",
                        "triplets": triplets
                    }
                }
                documents.append(doc)
    
    return documents


def generate_vector_store(
    documents: List[Dict[str, Any]],
    output_path: str,
    embedding_client: EmbeddingClient,
    storage_mode: str = "paragraph",
    batch_size: int = 32,
    checkpoint_interval: int = 100,
    save_interval: int = 500
) -> None:
    """生成向量存储（支持断点续跑，增量保存）"""
    
    checkpoint_manager = CheckpointManager(output_path)
    
    vector_store = FAISSVectorStore(
        metric="cosine",
        index_type="flat",
        storage_mode=storage_mode
    )
    
    start_batch = 0
    checkpoint_data = checkpoint_manager.load()
    
    if checkpoint_data:
        logger.info("发现检查点，正在恢复...")
        start_batch = checkpoint_data.get("batch_idx", 0)
        
        temp_index_path = os.path.join(output_path, "temp_index")
        if os.path.exists(temp_index_path):
            try:
                vector_store.load(temp_index_path)
                logger.info(f"已加载临时索引: {vector_store.count()} 个向量")
            except Exception as e:
                logger.warning(f"加载临时索引失败: {e}")
    
    total_docs = len(documents)
    total_batches = (total_docs + batch_size - 1) // batch_size
    
    logger.info(f"开始生成向量存储: {total_docs} 个文档, 模式: {storage_mode}")
    logger.info(f"总批次: {total_batches}, 起始批次: {start_batch}")
    
    for i in range(start_batch * batch_size, total_docs, batch_size):
        batch_idx = i // batch_size
        batch_docs = documents[i:i + batch_size]
        batch_texts = [doc["content"] for doc in batch_docs]
        
        logger.info(f"处理批次 {batch_idx + 1}/{total_batches}: {len(batch_docs)} 个文档")
        
        try:
            embeddings = embedding_client.embed(batch_texts, normalize=True)
            
            metadata_list = []
            ids = []
            
            for j, doc in enumerate(batch_docs):
                metadata = VectorMetadata(
                    doc_id=doc["doc_id"],
                    chunk_id=doc["doc_id"],
                    content=doc["content"],
                    source=doc["metadata"].get("source", "unknown"),
                    extra=doc["metadata"]
                )
                metadata_list.append(metadata)
                ids.append(doc["doc_id"])
            
            vectors = np.array(embeddings, dtype=np.float32)
            vector_store.add_vectors(vectors, metadata_list, ids)
            
            if (batch_idx + 1) % checkpoint_interval == 0:
                checkpoint_manager.save({"batch_idx": batch_idx + 1})
            
            if (batch_idx + 1) % save_interval == 0:
                temp_index_path = os.path.join(output_path, "temp_index")
                os.makedirs(temp_index_path, exist_ok=True)
                vector_store.save(temp_index_path)
                logger.info(f"临时索引已保存: {vector_store.count()} 个向量")
                
        except Exception as e:
            logger.error(f"批次 {batch_idx + 1} 处理失败: {e}")
            checkpoint_manager.save({"batch_idx": batch_idx})
            
            temp_index_path = os.path.join(output_path, "temp_index")
            os.makedirs(temp_index_path, exist_ok=True)
            vector_store.save(temp_index_path)
            raise
    
    logger.info(f"保存最终向量存储到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    vector_store.save(output_path)
    
    documents_json_path = os.path.join(output_path, "documents.json")
    with open(documents_json_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    temp_index_path = os.path.join(output_path, "temp_index")
    if os.path.exists(temp_index_path):
        import shutil
        shutil.rmtree(temp_index_path)
        logger.info("临时索引已清除")
    
    checkpoint_manager.clear()
    
    logger.info(f"向量存储生成完成: {vector_store.count()} 个向量")


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    
    source_file = project_root / "data" / "hotpotqa" / "valid_title_sentence.json"
    paragraph_output_dir = project_root / "data" / "hotpotqa" / "vector_stores" / "valid_title_sentence"
    sentence_output_dir = project_root / "data" / "hotpotqa" / "vector_stores" / "single_sentence"
    
    logger.info("=" * 60)
    logger.info("开始生成向量存储")
    logger.info("=" * 60)
    logger.info(f"源文件: {source_file}")
    
    data = load_source_data(str(source_file))
    
    logger.info("初始化 Embedding 客户端...")
    embedding_client = EmbeddingClient()
    
    """
    logger.info("\n" + "=" * 60)
    logger.info("生成段落级向量存储")
    logger.info("=" * 60)
    paragraph_docs = create_paragraph_documents(data)
    generate_vector_store(
        paragraph_docs, 
        str(paragraph_output_dir), 
        embedding_client,
        storage_mode="paragraph",
        batch_size=16,
        checkpoint_interval=50,
        save_interval=200
    )
    """
    logger.info("\n" + "=" * 60)
    logger.info("生成句子级向量存储")
    logger.info("=" * 60)
    sentence_docs = create_sentence_documents(data)
    generate_vector_store(
        sentence_docs,
        str(sentence_output_dir),
        embedding_client,
        storage_mode="sentence",
        batch_size=16,
        checkpoint_interval=50,
        save_interval=200
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("所有向量存储生成完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
