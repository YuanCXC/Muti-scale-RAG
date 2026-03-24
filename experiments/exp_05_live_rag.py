# -*- coding: utf-8 -*-
"""实验05: 动态路由 RAG (LiveRAG)

使用 BM25/Vector 双路检索 + 置信度路由决策。
根据 Top-1 分数的置信度动态选择最佳检索结果。

重构版本：使用 src/ 核心模块和 src/evaluation/ 评估系统。
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from experiment_base import (
    ExperimentConfig,
    load_knowledge_base,
    load_test_dataset,
    save_results,
    create_llm_client,
    create_embedding_client,
)

from src.storage.vector_store import FAISSVectorStore
from src.retrievers.vector_retriever import VectorRetriever
from src.retrievers.keyword_retriever import KeywordRetriever
from src.chains import HybridRAGChain
from src.evaluation import RAGEvaluator, EvaluationSample, WorkflowType
from src.evaluation.metrics import RetrievalMetrics, GenerationMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DynamicRouter:
    """动态路由器
    
    根据 BM25 和向量检索的置信度动态选择最佳检索策略。
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ):
        self.confidence_threshold = confidence_threshold
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.routing_stats = {
            "bm25_selected": 0,
            "vector_selected": 0,
            "hybrid_selected": 0,
        }
    
    def route(
        self,
        bm25_results: List[Any],
        vector_results: List[Any],
    ) -> Tuple[str, float]:
        """路由决策
        
        Args:
            bm25_results: BM25 检索结果
            vector_results: 向量检索结果
            
        Returns:
            (route_decision, confidence) 元组
        """
        if not bm25_results and not vector_results:
            return "none", 0.0
        
        bm25_top1_score = bm25_results[0].score if bm25_results else 0.0
        vector_top1_score = vector_results[0].score if vector_results else 0.0
        
        bm25_confidence = self._normalize_bm25_score(bm25_top1_score)
        vector_confidence = vector_top1_score
        
        if bm25_confidence > self.confidence_threshold and bm25_confidence > vector_confidence:
            self.routing_stats["bm25_selected"] += 1
            return "bm25", bm25_confidence
        
        elif vector_confidence > self.confidence_threshold and vector_confidence > bm25_confidence:
            self.routing_stats["vector_selected"] += 1
            return "vector", vector_confidence
        
        else:
            self.routing_stats["hybrid_selected"] += 1
            return "hybrid", max(bm25_confidence, vector_confidence)
    
    def _normalize_bm25_score(self, score: float) -> float:
        """归一化 BM25 分数到 [0, 1]"""
        return min(score / 10.0, 1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        total = sum(self.routing_stats.values())
        if total == 0:
            return self.routing_stats
        
        return {
            **self.routing_stats,
            "bm25_ratio": self.routing_stats["bm25_selected"] / total,
            "vector_ratio": self.routing_stats["vector_selected"] / total,
            "hybrid_ratio": self.routing_stats["hybrid_selected"] / total,
        }


def build_dual_index(
    knowledge_base: List[Dict[str, Any]],
    embedding_client: Any,
    vector_dim: int = 1024,
    batch_size: int = 32,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """构建双路索引（BM25 + 向量）
    
    Args:
        knowledge_base: 知识库文档列表
        embedding_client: Embedding 客户端
        vector_dim: 向量维度
        batch_size: 批处理大小
        
    Returns:
        (vector_store, keyword_retriever, doc_map) 元组
    """
    print("构建双路索引 (BM25 + 向量)...")
    
    vector_store = FAISSVectorStore(
        dimension=vector_dim,
        metric="cosine",
    )
    
    keyword_retriever = KeywordRetriever()
    
    documents = []
    vectors = []
    
    for i, chunk in enumerate(knowledge_base):
        doc_id = chunk.get("chunk_id", str(i))
        content = chunk.get("content", chunk.get("text", ""))
        metadata = chunk.get("metadata", {})
        
        documents.append({
            "doc_id": doc_id,
            "content": content,
            "metadata": metadata,
        })
    
    keyword_retriever.add_documents(documents)
    
    for i in tqdm(range(0, len(documents), batch_size), desc="向量化"):
        batch = documents[i:i + batch_size]
        texts = [doc["content"] for doc in batch]
        
        try:
            batch_vectors = embedding_client.embed(texts)
            if isinstance(batch_vectors, np.ndarray):
                batch_vectors = batch_vectors.tolist()
            vectors.extend(batch_vectors)
        except Exception as e:
            print(f"向量化批次 {i} 失败: {e}")
            continue
    
    if vectors:
        from src.storage.vector_store.base_store import VectorMetadata
        metadatas = [
            VectorMetadata(
                doc_id=doc["doc_id"],
                chunk_id=doc["doc_id"],
                content=doc["content"][:500],
                source=doc["metadata"].get("source", "unknown"),
                extra=doc["metadata"],
            )
            for doc in documents[:len(vectors)]
        ]
        vector_store.add_vectors(
            vectors=np.array(vectors, dtype=np.float32),
            metadata=metadatas,
            ids=[doc["doc_id"] for doc in documents[:len(vectors)]],
        )
    
    doc_map = {doc["doc_id"]: doc for doc in documents}
    
    print(f"双路索引构建完成: BM25={len(documents)} 文档, 向量={vector_store.count()} 文档")
    
    return vector_store, keyword_retriever, doc_map


def run_live_rag_experiment(
    config: ExperimentConfig,
    max_samples: int = 10,
    use_evaluator: bool = True,
) -> Dict[str, Any]:
    """运行 LiveRAG 实验
    
    Args:
        config: 实验配置
        max_samples: 最大样本数（0 表示全部）
        use_evaluator: 是否使用 RAGEvaluator（推荐）
        
    Returns:
        实验报告
    """
    print(f"\n{'='*60}")
    print(f"实验05: 动态路由 RAG (LiveRAG)")
    print(f"描述: BM25/Vector 双路检索 + 置信度路由决策")
    print(f"{'='*60}\n")
    
    llm_client = create_llm_client()
    embedding_client = create_embedding_client()
    
    knowledge_base = load_knowledge_base()
    print(f"知识库文档数: {len(knowledge_base)}")
    
    vector_store, keyword_retriever, doc_map = build_dual_index(
        knowledge_base, embedding_client, vector_dim=config.vector_dim
    )
    
    vector_retriever = VectorRetriever(
        vector_store=vector_store,
        llm_client=embedding_client,
        top_k=config.top_k,
    )
    
    router = DynamicRouter(
        confidence_threshold=0.7,
        bm25_weight=0.5,
        vector_weight=0.5,
    )
    
    test_data = load_test_dataset("rag_300_multihop.json")
    if max_samples > 0:
        test_data = test_data[:max_samples]
    
    print(f"测试样本数: {len(test_data)}")
    
    return _run_with_router(
        vector_retriever, keyword_retriever, router,
        test_data, config, llm_client, embedding_client, doc_map
    )


def _run_with_router(
    vector_retriever: Any,
    keyword_retriever: Any,
    router: DynamicRouter,
    test_data: List[Dict[str, Any]],
    config: ExperimentConfig,
    llm_client: Any,
    embedding_client: Any,
    doc_map: Dict[str, Any],
) -> Dict[str, Any]:
    """使用动态路由器运行评估"""
    retrieval_metrics = RetrievalMetrics(k_values=[1, 3, 5, 10])
    generation_metrics = GenerationMetrics()
    
    results = []
    success_count = 0
    failed_count = 0
    total_latency = 0.0
    
    for i, item in enumerate(tqdm(test_data, desc="处理中")):
        query = item.get("question", item.get("query", ""))
        ground_truth = item.get("ground_truth", item.get("answer", ""))
        relevant_docs = item.get("relevant_docs", item.get("doc_ids", []))
        
        result = {
            "id": item.get("id", str(i)),
            "question": query,
            "ground_truth": ground_truth,
            "predicted_answer": "",
            "retrieved_docs": [],
            "route_decision": "",
            "confidence": 0.0,
            "success": False,
            "latency": 0.0,
            "error": None,
        }
        
        try:
            start_time = time.time()
            
            vector_results = vector_retriever.retrieve(query, top_k=config.top_k * 2)
            bm25_results = keyword_retriever.retrieve(query, top_k=config.top_k * 2)
            
            route_decision, confidence = router.route(bm25_results, vector_results)
            result["route_decision"] = route_decision
            result["confidence"] = confidence
            
            if route_decision == "bm25":
                selected_docs = bm25_results[:config.top_k]
            elif route_decision == "vector":
                selected_docs = vector_results[:config.top_k]
            else:
                seen_ids = set()
                selected_docs = []
                for doc in vector_results[:config.top_k]:
                    if doc.doc_id not in seen_ids:
                        seen_ids.add(doc.doc_id)
                        selected_docs.append(doc)
                for doc in bm25_results[:config.top_k]:
                    if doc.doc_id not in seen_ids:
                        seen_ids.add(doc.doc_id)
                        selected_docs.append(doc)
                selected_docs = selected_docs[:config.top_k]
            
            context_docs = [
                {
                    "doc_id": doc.doc_id,
                    "content": doc.content,
                    "score": doc.score,
                    "metadata": doc.metadata,
                }
                for doc in selected_docs
            ]
            
            result["retrieved_docs"] = [
                {"doc_id": d["doc_id"], "score": d["score"]} for d in context_docs
            ]
            
            from src.llms.base_client import Message
            context = "\n\n".join([
                f"[文档 {i+1}]\n{doc['content']}"
                for i, doc in enumerate(context_docs)
            ])
            
            messages = [
                Message(role="system", content="你是一个有帮助的AI助手。请根据提供的上下文信息回答用户问题。"),
                Message(role="user", content=f"上下文信息：\n{context}\n\n用户问题：{query}\n\n请根据上下文信息回答问题："),
            ]
            
            response = llm_client.generate(messages)
            answer = response.content
            
            result["predicted_answer"] = answer
            result["success"] = True
            result["latency"] = time.time() - start_time
            
            retrieved_ids = [d["doc_id"] for d in context_docs]
            if relevant_docs:
                ret_result = retrieval_metrics.compute(retrieved_ids, relevant_docs)
                result["retrieval_metrics"] = ret_result.to_dict()
            
            gen_result = generation_metrics.compute(answer, ground_truth, compute_semantic=False)
            result["generation_metrics"] = gen_result.to_dict()
            
            success_count += 1
            total_latency += result["latency"]
            
        except Exception as e:
            result["error"] = str(e)
            result["latency"] = time.time() - start_time if 'start_time' in dir() else 0
            failed_count += 1
            print(f"\n样本 {i} 错误: {e}")
        
        results.append(result)
    
    avg_latency = total_latency / len(results) if results else 0
    routing_stats = router.get_stats()
    
    avg_retrieval = {}
    avg_generation = {}
    
    retrieval_results = [r.get("retrieval_metrics") for r in results if r.get("retrieval_metrics")]
    if retrieval_results:
        avg_retrieval = {
            "recall_at_k": {},
            "precision_at_k": {},
        }
        for k in [1, 3, 5, 10]:
            recalls = [r.get("recall_at_k", {}).get(k, 0) for r in retrieval_results]
            precisions = [r.get("precision_at_k", {}).get(k, 0) for r in retrieval_results]
            avg_retrieval["recall_at_k"][k] = sum(recalls) / len(recalls)
            avg_retrieval["precision_at_k"][k] = sum(precisions) / len(precisions)
        avg_retrieval["mrr"] = sum(r.get("mrr", 0) for r in retrieval_results) / len(retrieval_results)
        avg_retrieval["ndcg"] = sum(r.get("ndcg", 0) for r in retrieval_results) / len(retrieval_results)
    
    generation_results = [r.get("generation_metrics") for r in results if r.get("generation_metrics")]
    if generation_results:
        avg_generation = {
            "exact_match": sum(r.get("exact_match", 0) for r in generation_results) / len(generation_results),
            "f1_score": sum(r.get("f1_score", 0) for r in generation_results) / len(generation_results),
        }
    
    report = {
        "experiment_name": "动态路由 RAG",
        "experiment_id": config.experiment_id,
        "description": "BM25/Vector 双路检索 + 置信度路由决策",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(results),
        "successful_samples": success_count,
        "failed_samples": failed_count,
        "avg_latency": avg_latency,
        "avg_retrieval_metrics": avg_retrieval,
        "avg_generation_metrics": avg_generation,
        "routing_stats": routing_stats,
        "config": {
            "max_samples": max_samples,
            "top_k": config.top_k,
            "llm_model": config.llm_model,
            "embedding_model": config.embedding_model,
        },
    }
    
    save_results(results, config.get_results_path())
    save_results(report, config.get_metrics_path())
    
    print(f"\n{'='*60}")
    print(f"实验完成!")
    print(f"总样本: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"平均延迟: {avg_latency:.2f}s")
    print(f"\n路由统计:")
    print(f"  - BM25 选择: {routing_stats['bm25_selected']} ({routing_stats['bm25_ratio']:.1%})")
    print(f"  - Vector 选择: {routing_stats['vector_selected']} ({routing_stats['vector_ratio']:.1%})")
    print(f"  - Hybrid 选择: {routing_stats['hybrid_selected']} ({routing_stats['hybrid_ratio']:.1%})")
    print(f"{'='*60}\n")
    
    return report


def main():
    config = ExperimentConfig(
        experiment_name="动态路由 RAG",
        experiment_id="exp_05_live_rag",
        description="BM25/Vector 双路检索 + 置信度路由决策",
        test_samples=10,
    )
    
    report = run_live_rag_experiment(config, max_samples=10)
    
    print("结果已保存到:")
    print(f"  - {config.get_results_path()}")
    print(f"  - {config.get_metrics_path()}")


if __name__ == "__main__":
    main()
