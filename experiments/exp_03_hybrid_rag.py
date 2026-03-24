# -*- coding: utf-8 -*-
"""实验03: 混合检索 RAG (Hybrid RAG)

使用 BM25 + 向量检索 + RRF 融合算法，结合关键词和语义检索。

重构版本：使用 src/ 核心模块和 src/evaluation/ 评估系统。
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
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
from src.retrievers.hybrid_retriever import HybridRetriever
from src.chains import HybridRAGChain
from src.evaluation import RAGEvaluator, EvaluationSample, WorkflowType
from src.evaluation.metrics import RetrievalMetrics, GenerationMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_hybrid_index(
    knowledge_base: List[Dict[str, Any]],
    embedding_client: Any,
    vector_dim: int = 1024,
    batch_size: int = 32,
) -> tuple:
    """构建混合索引（向量 + 关键词）
    
    Args:
        knowledge_base: 知识库文档列表
        embedding_client: Embedding 客户端
        vector_dim: 向量维度
        batch_size: 批处理大小
        
    Returns:
        (vector_store, keyword_retriever, doc_map) 元组
    """
    print("构建混合索引 (向量 + BM25)...")
    
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
    print(f"BM25 索引构建完成: {len(documents)} 个文档")
    
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
    
    print(f"混合索引构建完成: BM25={len(documents)} 文档, 向量={vector_store.count()} 文档")
    
    return vector_store, keyword_retriever, doc_map


def run_hybrid_rag_experiment(
    config: ExperimentConfig,
    max_samples: int = 10,
    use_evaluator: bool = True,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
) -> Dict[str, Any]:
    """运行 Hybrid RAG 实验
    
    Args:
        config: 实验配置
        max_samples: 最大样本数（0 表示全部）
        use_evaluator: 是否使用 RAGEvaluator（推荐）
        vector_weight: 向量检索权重
        keyword_weight: 关键词检索权重
        
    Returns:
        实验报告
    """
    print(f"\n{'='*60}")
    print(f"实验03: 混合检索 RAG (Hybrid RAG)")
    print(f"描述: 使用 BM25 + 向量检索 + RRF 融合算法")
    print(f"权重: 向量={vector_weight}, 关键词={keyword_weight}")
    print(f"{'='*60}\n")
    
    llm_client = create_llm_client()
    embedding_client = create_embedding_client()
    
    knowledge_base = load_knowledge_base()
    print(f"知识库文档数: {len(knowledge_base)}")
    
    vector_store, keyword_retriever, doc_map = build_hybrid_index(
        knowledge_base, 
        embedding_client,
        vector_dim=config.vector_dim,
    )
    
    vector_retriever = VectorRetriever(
        vector_store=vector_store,
        llm_client=embedding_client,
        top_k=config.top_k,
    )
    
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        top_k=config.top_k,
    )
    
    chain = HybridRAGChain(
        retriever=hybrid_retriever,
        llm_client=llm_client,
        top_k=config.top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )
    
    test_data = load_test_dataset("rag_300_multihop.json")
    if max_samples > 0:
        test_data = test_data[:max_samples]
    
    print(f"测试样本数: {len(test_data)}")
    
    if use_evaluator:
        return _run_with_evaluator(chain, test_data, config, llm_client, embedding_client)
    else:
        return _run_manual(chain, test_data, config, doc_map)


def _run_with_evaluator(
    chain: HybridRAGChain,
    test_data: List[Dict[str, Any]],
    config: ExperimentConfig,
    llm_client: Any,
    embedding_client: Any,
) -> Dict[str, Any]:
    """使用 RAGEvaluator 运行评估"""
    samples = [
        EvaluationSample(
            query=item.get("question", item.get("query", "")),
            ground_truth=item.get("ground_truth", item.get("answer", "")),
            relevant_docs=item.get("relevant_docs", item.get("doc_ids", [])),
            metadata=item.get("metadata", {}),
        )
        for item in test_data
    ]
    
    evaluator = RAGEvaluator(
        workflow=chain,
        workflow_type=WorkflowType.HYBRID,
        embedding_client=embedding_client,
        llm_client=llm_client,
        checkpoint_path=str(config.get_checkpoint_path()),
        max_workers=1,
        k_values=[1, 3, 5, 10],
    )
    
    report = evaluator.evaluate(samples, resume=True, save_checkpoint_every=5)
    
    evaluator.save_report(report, str(config.get_metrics_path().with_suffix('')))
    
    print(f"\n{'='*60}")
    print(f"实验完成!")
    print(f"总样本: {report.total_samples}")
    print(f"成功: {report.successful_samples}")
    print(f"失败: {report.failed_samples}")
    print(f"平均延迟: {report.avg_latency:.2f}s")
    
    if report.avg_retrieval_metrics:
        print(f"\n检索指标:")
        for k, v in report.avg_retrieval_metrics.get("recall_at_k", {}).items():
            print(f"  Recall@{k}: {v:.4f}")
        for k, v in report.avg_retrieval_metrics.get("precision_at_k", {}).items():
            print(f"  Precision@{k}: {v:.4f}")
        print(f"  MRR: {report.avg_retrieval_metrics.get('mrr', 0):.4f}")
        print(f"  NDCG: {report.avg_retrieval_metrics.get('ndcg', 0):.4f}")
    
    if report.avg_generation_metrics:
        print(f"\n生成指标:")
        print(f"  EM: {report.avg_generation_metrics.get('exact_match', 0):.4f}")
        print(f"  F1: {report.avg_generation_metrics.get('f1_score', 0):.4f}")
        if "semantic_similarity" in report.avg_generation_metrics:
            print(f"  语义相似度: {report.avg_generation_metrics['semantic_similarity']:.4f}")
    
    print(f"{'='*60}\n")
    
    return report.to_dict()


def _run_manual(
    chain: HybridRAGChain,
    test_data: List[Dict[str, Any]],
    config: ExperimentConfig,
    doc_map: Dict[str, Any],
) -> Dict[str, Any]:
    """手动运行评估（不使用 RAGEvaluator）"""
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
            "success": False,
            "latency": 0.0,
            "error": None,
        }
        
        try:
            chain_result = chain.run(query)
            
            result["predicted_answer"] = chain_result.answer
            result["success"] = chain_result.success
            result["latency"] = chain_result.latency
            result["error"] = chain_result.error
            
            if chain_result.sources:
                result["retrieved_docs"] = [
                    {"doc_id": s.get("doc_id"), "score": s.get("score")}
                    for s in chain_result.sources
                ]
            
            retrieved_ids = [s.get("doc_id") for s in chain_result.sources if s.get("doc_id")]
            
            if relevant_docs:
                ret_result = retrieval_metrics.compute(retrieved_ids, relevant_docs)
                result["retrieval_metrics"] = ret_result.to_dict()
            
            gen_result = generation_metrics.compute(
                chain_result.answer,
                ground_truth,
                compute_semantic=False,
                compute_llm_based=False,
            )
            result["generation_metrics"] = gen_result.to_dict()
            
            if chain_result.success:
                success_count += 1
            else:
                failed_count += 1
            
            total_latency += chain_result.latency
            
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            failed_count += 1
            print(f"\n样本 {i} 错误: {e}")
        
        results.append(result)
    
    avg_latency = total_latency / len(results) if results else 0
    
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
        "experiment_name": "混合检索 RAG",
        "experiment_id": config.experiment_id,
        "description": "使用 BM25 + 向量检索 + RRF 融合算法",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(results),
        "successful_samples": success_count,
        "failed_samples": failed_count,
        "avg_latency": avg_latency,
        "avg_retrieval_metrics": avg_retrieval,
        "avg_generation_metrics": avg_generation,
        "config": {
            "max_samples": max_samples,
            "top_k": config.top_k,
            "vector_weight": chain.vector_weight,
            "keyword_weight": chain.keyword_weight,
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
    print(f"{'='*60}\n")
    
    return report


def main():
    config = ExperimentConfig(
        experiment_name="混合检索 RAG",
        experiment_id="exp_03_hybrid_rag",
        description="使用 BM25 + 向量检索 + RRF 融合算法",
        test_samples=10,
    )
    
    report = run_hybrid_rag_experiment(
        config, 
        max_samples=10, 
        use_evaluator=True,
        vector_weight=0.5,
        keyword_weight=0.5,
    )
    
    print("结果已保存到:")
    print(f"  - {config.get_results_path()}")
    print(f"  - {config.get_metrics_path()}")


if __name__ == "__main__":
    main()
