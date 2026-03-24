# -*- coding: utf-8 -*-
"""实验07: HotpotQA 多跳问答实验

适配 HotpotQA 数据格式，使用完整 RAG 流程：
- 加载已有向量存储
- 检索 + 父切片映射
- 多跳推理
"""

import json
import time
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from experiment_base import (
    ExperimentConfig,
    save_results,
    create_llm_client,
    create_embedding_client,
)


def load_existing_vector_store(store_path: Path) -> Tuple[Any, Dict[str, Any]]:
    print(f"加载已有向量存储: {store_path}")
    
    index_path = store_path / "faiss.index"
    docs_path = store_path / "documents.json"
    
    if not index_path.exists() or not docs_path.exists():
        raise FileNotFoundError(f"向量存储文件不存在: {store_path}")
    
    index = faiss.read_index(str(index_path))
    
    with open(docs_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    doc_map = {}
    for doc in documents:
        doc_id = doc.get("doc_id", doc.get("id", ""))
        doc_map[doc_id] = doc
    
    print(f"向量存储加载完成: {index.ntotal} 个向量, {len(doc_map)} 个文档")
    
    return index, doc_map


def load_hotpotqa_queries(data_dir: Path) -> List[Dict]:
    print("加载 HotpotQA 查询数据...")
    
    parquet_path = data_dir / "validation-00000-of-00001.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        queries = df.to_dict('records')
        print(f"加载查询数据: {len(queries)} 条")
    else:
        queries = []
        print("未找到查询数据")
    
    return queries


def retrieve_with_faiss(
    query: str,
    index: Any,
    doc_map: Dict[str, Any],
    embedding_client: Any,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    query_vector = embedding_client.embed([query])
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.astype(np.float32)
    else:
        query_vector = np.array(query_vector, dtype=np.float32)
    
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    
    distances, indices = index.search(query_vector, top_k)
    
    retrieved = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0:
            continue
        
        doc_id = str(idx)
        doc = None
        
        for did, d in doc_map.items():
            if str(d.get("index", d.get("id", ""))) == doc_id:
                doc = d
                break
        
        if doc is None and idx < len(doc_map):
            doc = list(doc_map.values())[idx]
        
        if doc:
            retrieved.append({
                "doc_id": doc.get("doc_id", doc.get("id", str(idx))),
                "title": doc.get("title", ""),
                "content": doc.get("content", doc.get("text", "")),
                "score": float(1.0 / (1.0 + dist)),
            })
    
    return retrieved


def generate_multihop_answer(
    query: str,
    context_docs: List[Dict[str, Any]],
    llm_client: Any,
) -> str:
    context = "\n\n".join([
        f"[文档 {i+1}] 标题: {doc.get('title', 'N/A')}\n{doc['content'][:500]}..."
        for i, doc in enumerate(context_docs)
    ])
    
    system_prompt = """你是一个专业的问答助手。请根据提供的文档回答用户的多跳问题。

多跳问题需要综合多个文档的信息才能回答。请：
1. 识别问题中涉及的关键实体和概念
2. 从各个文档中提取相关信息
3. 进行推理和综合
4. 给出完整的答案

如果需要多个文档的信息，请明确说明推理过程。"""

    user_prompt = f"""参考文档：
{context}

问题：{query}

请回答问题，并说明你的推理过程："""

    from src.llms.base_client import Message
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    
    response = llm_client.generate(messages)
    return response.content


def run_hotpotqa_experiment(
    config: ExperimentConfig,
    max_samples: int = 10,
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"实验07: HotpotQA 多跳问答实验")
    print(f"描述: 适配 HotpotQA 数据格式，支持多跳推理")
    print(f"{'='*60}\n")
    
    llm_client = create_llm_client()
    embedding_client = create_embedding_client()
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "hotpotqa"
    vector_store_path = data_dir / "vector_stores" / "valid_title_sentence"
    
    try:
        index, doc_map = load_existing_vector_store(vector_store_path)
    except FileNotFoundError:
        print("警告: 未找到已有向量存储，尝试使用 single_sentence 存储")
        vector_store_path = data_dir / "vector_stores" / "single_sentence"
        index, doc_map = load_existing_vector_store(vector_store_path)
    
    queries = load_hotpotqa_queries(data_dir)
    
    if max_samples > 0:
        queries = queries[:max_samples]
    
    print(f"测试样本数: {len(queries)}")
    
    results = []
    success_count = 0
    failed_count = 0
    total_latency = 0.0
    
    for i, item in enumerate(tqdm(queries, desc="处理中")):
        query = item.get("question", "")
        ground_truth = item.get("answer", "")
        supporting_facts = item.get("supporting_facts", [])
        
        result = {
            "id": item.get("id", str(i)),
            "question": query,
            "ground_truth": ground_truth,
            "predicted_answer": "",
            "retrieved_docs": [],
            "supporting_facts": supporting_facts,
            "success": False,
            "latency": 0.0,
            "error": None,
        }
        
        start_time = time.time()
        
        try:
            retrieved_docs = retrieve_with_faiss(
                query=query,
                index=index,
                doc_map=doc_map,
                embedding_client=embedding_client,
                top_k=config.top_k,
            )
            
            result["retrieved_docs"] = [
                {"doc_id": doc["doc_id"], "title": doc["title"], "score": doc["score"]}
                for doc in retrieved_docs
            ]
            
            answer = generate_multihop_answer(query, retrieved_docs, llm_client)
            
            result["predicted_answer"] = answer
            result["success"] = True
            result["latency"] = time.time() - start_time
            
            success_count += 1
            total_latency += result["latency"]
            
        except Exception as e:
            result["error"] = str(e)
            result["latency"] = time.time() - start_time
            failed_count += 1
            print(f"\n样本 {i} 错误: {e}")
        
        results.append(result)
    
    avg_latency = total_latency / len(results) if results else 0
    
    report = {
        "experiment_name": "HotpotQA 多跳问答",
        "experiment_id": config.experiment_id,
        "description": "适配 HotpotQA 数据格式，支持多跳推理",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(results),
        "successful_samples": success_count,
        "failed_samples": failed_count,
        "avg_latency": avg_latency,
        "config": {
            "max_samples": max_samples,
            "top_k": config.top_k,
            "llm_model": config.llm_model,
            "embedding_model": config.embedding_model,
            "vector_store_path": str(vector_store_path),
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
        experiment_name="HotpotQA 多跳问答",
        experiment_id="exp_07_hotpotqa",
        description="适配 HotpotQA 数据格式，支持多跳推理",
        test_samples=10,
    )
    
    report = run_hotpotqa_experiment(config, max_samples=10)
    
    print("结果已保存到:")
    print(f"  - {config.get_results_path()}")
    print(f"  - {config.get_metrics_path()}")


if __name__ == "__main__":
    main()
