# -*- coding: utf-8 -*-
"""实验06: 完整系统 (Full System)

使用问题分类 + 差异化检索策略 + Rerank 重排序。
- Fact: 语义(10) + BM25(10) -> Rerank Top-5
- Explanation: 语义(10) + 图谱(3) -> Rerank Top-6
- Reasoning: 查询分解 + 多向量检索 + 图谱 -> Rerank Top-20 + CoT推理

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
    load_knowledge_graph,
    load_test_dataset,
    save_results,
    create_llm_client,
    create_embedding_client,
)

from src.storage.vector_store import FAISSVectorStore
from src.storage.graph_store import LocalGraphStore
from src.retrievers.vector_retriever import VectorRetriever
from src.retrievers.keyword_retriever import KeywordRetriever
from src.retrievers.graph_retriever import GraphRetriever
from src.retrievers.reranker import Reranker
from src.evaluation.metrics import RetrievalMetrics, GenerationMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QuestionClassifier:
    """问题分类器
    
    将问题分为三类：fact（事实型）、explanation（解释型）、reasoning（推理型）
    """
    
    def __init__(self, llm_client: Any):
        self.llm_client = llm_client
        
        self.fact_patterns = [
            r"什么是", r"定义", r"是谁", r"在哪", r"何时",
            r"多少", r"哪个", r"是否", r"有没有",
        ]
        
        self.explanation_patterns = [
            r"为什么", r"如何", r"怎样", r"原因", r"解释",
            r"区别", r"比较", r"优缺点", r"原理",
        ]
        
        self.reasoning_patterns = [
            r"如果.*那么", r"假设", r"推理", r"推断",
            r"综合", r"分析", r"评估", r"判断",
        ]
    
    def classify(self, query: str) -> str:
        """分类问题类型
        
        Args:
            query: 用户查询
            
        Returns:
            问题类型：fact/explanation/reasoning
        """
        import re
        
        for pattern in self.reasoning_patterns:
            if re.search(pattern, query):
                return "reasoning"
        
        for pattern in self.explanation_patterns:
            if re.search(pattern, query):
                return "explanation"
        
        for pattern in self.fact_patterns:
            if re.search(pattern, query):
                return "fact"
        
        return self._llm_classify(query)
    
    def _llm_classify(self, query: str) -> str:
        """使用 LLM 进行分类"""
        prompt = f"""请判断以下问题的类型，只返回一个类别：
- fact: 事实型问题，询问具体信息
- explanation: 解释型问题，需要解释原因或原理
- reasoning: 推理型问题，需要多步推理或综合分析

问题：{query}

类型："""

        from src.llms.base_client import Message
        messages = [Message(role="user", content=prompt)]
        
        try:
            response = self.llm_client.generate(messages)
            result = response.content.strip().lower()
            
            if "reasoning" in result:
                return "reasoning"
            elif "explanation" in result:
                return "explanation"
            else:
                return "fact"
        except Exception:
            return "fact"


def build_full_system_index(
    knowledge_base: List[Dict[str, Any]],
    knowledge_graph: Dict[str, Any],
    embedding_client: Any,
    vector_dim: int = 1024,
    batch_size: int = 32,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """构建完整系统索引
    
    Args:
        knowledge_base: 知识库文档列表
        knowledge_graph: 知识图谱数据
        embedding_client: Embedding 客户端
        vector_dim: 向量维度
        batch_size: 批处理大小
        
    Returns:
        (vector_store, keyword_retriever, graph_store, doc_map) 元组
    """
    print("构建完整系统索引...")
    
    vector_store = FAISSVectorStore(
        dimension=vector_dim,
        metric="cosine",
    )
    
    keyword_retriever = KeywordRetriever()
    graph_store = LocalGraphStore()
    
    if knowledge_graph:
        nodes = knowledge_graph.get("nodes", [])
        edges = knowledge_graph.get("edges", knowledge_graph.get("links", []))
        
        for node in nodes:
            node_id = node.get("id", node.get("node_id", ""))
            node_name = node.get("name", node_id)
            node_type = node.get("type", "entity")
            properties = node.get("properties", {})
            
            graph_store.add_node(
                node_id=node_id,
                node_type=node_type,
                properties={"name": node_name, **properties},
            )
        
        for edge in edges:
            source = edge.get("source", edge.get("source_id", ""))
            target = edge.get("target", edge.get("target_id", ""))
            relation = edge.get("relation", edge.get("type", "related_to"))
            
            if source and target:
                graph_store.add_edge(
                    source_id=source,
                    target_id=target,
                    relation=relation,
                )
    
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
    
    stats = graph_store.get_stats()
    print(
        f"索引构建完成: 向量={vector_store.count()}, "
        f"BM25={len(documents)}, 图谱节点={stats.get('node_count', 0)}"
    )
    
    return vector_store, keyword_retriever, graph_store, doc_map


def retrieve_by_strategy(
    query: str,
    question_type: str,
    vector_retriever: Any,
    keyword_retriever: Any,
    graph_store: Any,
    reranker: Any,
    llm_client: Any,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """根据问题类型使用差异化检索策略
    
    Args:
        query: 用户查询
        question_type: 问题类型
        vector_retriever: 向量检索器
        keyword_retriever: 关键词检索器
        graph_store: 图存储
        reranker: 重排序器
        llm_client: LLM 客户端
        top_k: 返回文档数量
        
    Returns:
        检索到的文档列表
    """
    candidates = []
    
    if question_type == "fact":
        vector_results = vector_retriever.retrieve(query, top_k=10)
        keyword_results = keyword_retriever.retrieve(query, top_k=10)
        
        seen_ids = set()
        for doc in vector_results:
            if doc.doc_id not in seen_ids:
                seen_ids.add(doc.doc_id)
                candidates.append({
                    "doc_id": doc.doc_id,
                    "content": doc.content,
                    "score": doc.score,
                    "source": "vector",
                })
        for doc in keyword_results:
            if doc.doc_id not in seen_ids:
                seen_ids.add(doc.doc_id)
                candidates.append({
                    "doc_id": doc.doc_id,
                    "content": doc.content,
                    "score": doc.score,
                    "source": "keyword",
                })
        
        candidates = reranker.rerank(query, candidates, top_k=top_k)
    
    elif question_type == "explanation":
        vector_results = vector_retriever.retrieve(query, top_k=10)
        
        for doc in vector_results:
            candidates.append({
                "doc_id": doc.doc_id,
                "content": doc.content,
                "score": doc.score,
                "source": "vector",
            })
        
        graph_results = graph_store.search_entities(query, limit=3)
        for entity in graph_results:
            entity_id = entity.get("id", entity.get("node_id", ""))
            entity_name = entity.get("name", entity_id)
            
            neighbors = graph_store.get_neighbors(entity_id, max_depth=1)
            neighbor_names = [n.get("name", n.get("id", "")) for n in neighbors[:5]]
            
            candidates.append({
                "doc_id": f"kg_{entity_id}",
                "content": f"实体: {entity_name}\n关联: {', '.join(neighbor_names)}",
                "score": 0.8,
                "source": "graph",
            })
        
        candidates = reranker.rerank(query, candidates, top_k=top_k + 1)
    
    else:  # reasoning
        sub_queries = decompose_query(query, llm_client)
        
        for sub_q in sub_queries[:3]:
            sub_results = vector_retriever.retrieve(sub_q, top_k=5)
            
            for doc in sub_results:
                candidates.append({
                    "doc_id": doc.doc_id,
                    "content": doc.content,
                    "score": doc.score,
                    "source": "vector",
                    "sub_query": sub_q,
                })
        
        graph_results = graph_store.search_entities(query, limit=3)
        for entity in graph_results:
            entity_id = entity.get("id", entity.get("node_id", ""))
            entity_name = entity.get("name", entity_id)
            
            neighbors = graph_store.get_neighbors(entity_id, max_depth=1)
            neighbor_names = [n.get("name", n.get("id", "")) for n in neighbors[:5]]
            
            candidates.append({
                "doc_id": f"kg_{entity_id}",
                "content": f"实体: {entity_name}\n关联: {', '.join(neighbor_names)}",
                "score": 0.8,
                "source": "graph",
            })
        
        candidates = reranker.rerank(query, candidates, top_k=min(top_k * 2, 10))
    
    return candidates[:top_k]


def decompose_query(query: str, llm_client: Any) -> List[str]:
    """分解复杂查询
    
    Args:
        query: 用户查询
        llm_client: LLM 客户端
        
    Returns:
        子查询列表
    """
    prompt = f"""请将以下复杂问题分解为2-3个简单的子问题，每个子问题一行：

问题：{query}

子问题："""

    from src.llms.base_client import Message
    messages = [Message(role="user", content=prompt)]
    
    try:
        response = llm_client.generate(messages)
        sub_queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        return sub_queries[:3]
    except Exception:
        return [query]


def generate_answer_with_cot(
    query: str,
    context_docs: List[Dict[str, Any]],
    question_type: str,
    llm_client: Any,
) -> str:
    """生成答案（支持 CoT 推理）
    
    Args:
        query: 用户查询
        context_docs: 上下文文档
        question_type: 问题类型
        llm_client: LLM 客户端
        
    Returns:
        生成的答案
    """
    context = "\n\n".join([
        f"[文档 {i+1}]\n{doc['content']}"
        for i, doc in enumerate(context_docs)
    ])
    
    if question_type == "reasoning":
        system_prompt = """你是一个有帮助的AI助手。请使用思维链(Chain-of-Thought)推理回答用户问题。
步骤：
1. 分析问题的关键要素
2. 从上下文中提取相关信息
3. 进行逐步推理
4. 给出最终答案

请清晰展示推理过程。"""
    else:
        system_prompt = """你是一个有帮助的AI助手。请根据提供的上下文信息回答用户问题。
如果上下文中没有相关信息，请明确说明。
回答要简洁、准确、有条理。"""

    user_prompt = f"""上下文信息：
{context}

用户问题：{query}

请根据上下文信息回答问题："""

    from src.llms.base_client import Message
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    
    response = llm_client.generate(messages)
    return response.content


def run_full_system_experiment(
    config: ExperimentConfig,
    max_samples: int = 10,
) -> Dict[str, Any]:
    """运行完整系统实验
    
    Args:
        config: 实验配置
        max_samples: 最大样本数（0 表示全部）
        
    Returns:
        实验报告
    """
    print(f"\n{'='*60}")
    print(f"实验06: 完整系统 (Full System)")
    print(f"描述: 问题分类 + 差异化检索策略 + Rerank")
    print(f"{'='*60}\n")
    
    llm_client = create_llm_client()
    embedding_client = create_embedding_client()
    
    knowledge_base = load_knowledge_base()
    knowledge_graph = load_knowledge_graph()
    
    print(f"知识库文档数: {len(knowledge_base)}")
    
    vector_store, keyword_retriever, graph_store, doc_map = build_full_system_index(
        knowledge_base, knowledge_graph, embedding_client, vector_dim=config.vector_dim
    )
    
    vector_retriever = VectorRetriever(
        vector_store=vector_store,
        llm_client=embedding_client,
        top_k=config.top_k,
    )
    
    classifier = QuestionClassifier(llm_client)
    reranker = Reranker()
    
    test_data = load_test_dataset("rag_300_multihop.json")
    if max_samples > 0:
        test_data = test_data[:max_samples]
    
    print(f"测试样本数: {len(test_data)}")
    
    retrieval_metrics = RetrievalMetrics(k_values=[1, 3, 5, 10])
    generation_metrics = GenerationMetrics()
    
    results = []
    success_count = 0
    failed_count = 0
    total_latency = 0.0
    type_stats = {"fact": 0, "explanation": 0, "reasoning": 0}
    
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
            "question_type": "",
            "success": False,
            "latency": 0.0,
            "error": None,
        }
        
        try:
            start_time = time.time()
            
            question_type = classifier.classify(query)
            result["question_type"] = question_type
            type_stats[question_type] = type_stats.get(question_type, 0) + 1
            
            retrieved_docs = retrieve_by_strategy(
                query=query,
                question_type=question_type,
                vector_retriever=vector_retriever,
                keyword_retriever=keyword_retriever,
                graph_store=graph_store,
                reranker=reranker,
                llm_client=llm_client,
                top_k=config.top_k,
            )
            
            result["retrieved_docs"] = [
                {"doc_id": doc["doc_id"], "score": doc.get("rerank_score", doc["score"])}
                for doc in retrieved_docs
            ]
            
            answer = generate_answer_with_cot(query, retrieved_docs, question_type, llm_client)
            
            result["predicted_answer"] = answer
            result["success"] = True
            result["latency"] = time.time() - start_time
            
            retrieved_ids = [doc["doc_id"] for doc in retrieved_docs]
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
        "experiment_name": "完整系统",
        "experiment_id": config.experiment_id,
        "description": "问题分类 + 差异化检索策略 + Rerank",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(results),
        "successful_samples": success_count,
        "failed_samples": failed_count,
        "avg_latency": avg_latency,
        "avg_retrieval_metrics": avg_retrieval,
        "avg_generation_metrics": avg_generation,
        "question_type_stats": type_stats,
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
    print(f"\n问题类型分布:")
    for qtype, count in type_stats.items():
        print(f"  - {qtype}: {count}")
    print(f"{'='*60}\n")
    
    return report


def main():
    config = ExperimentConfig(
        experiment_name="完整系统",
        experiment_id="exp_06_full_system",
        description="问题分类 + 差异化检索策略 + Rerank",
        test_samples=10,
    )
    
    report = run_full_system_experiment(config, max_samples=10)
    
    print("结果已保存到:")
    print(f"  - {config.get_results_path()}")
    print(f"  - {config.get_metrics_path()}")


if __name__ == "__main__":
    main()
