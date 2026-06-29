# -*- coding: utf-8 -*-
"""实验1.4：KG-RAG

方法：向量 + 关键词 + 重排序
"""
from __future__ import annotations
import argparse
import json
import re
import threading
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Iterable

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrievers.reranker import RerankSearchResult


@dataclass
class RetrievalConfig:
    test_data_path: Path = Path("data/hotpotqa/validation-00000-of-00001.parquet")
    documents_path: Path = Path("data/hotpotqa/valid_title_sentence.json")
    paragraph_store_path: Path = Path("data/hotpotqa/vector_stores/valid_title_sentence")
    sentence_store_path: Path = Path("data/hotpotqa/vector_stores/single_sentence")
    output_dir: Path = Path("new_experiments/results")
    sample_size: int = 1000
    random_seed: int = 42
    k1: int = 10
    k2: int = 20
    k3: int = 7
    hmax: int = 2
    context_budget: int = 3600
    complexity_threshold: float = 0.80
    parent_threshold: float = 0.60
    fragmentation_threshold: float = 0.65
    max_graph_neighbors: int = 10
    max_context_units: int = 20
    run_generation: bool = True
    run_judge: bool = True
    use_api_reranker: bool = True
    require_neo4j: bool = True
    max_workers: int = 250


@dataclass
class EvidenceUnit:
    id: str
    title: str
    content: str
    score: float = 0.0
    source: str = "unknown"
    granularity: str = "sentence"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodResult:
    units: List[EvidenceUnit]
    stats: Dict[str, Any]


@dataclass
class ExperimentMetrics:
    recall: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    map_score: float = 0.0
    avg_len: float = 0.0
    time_ms: float = 0.0
    expanded_nodes: float = 0.0


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_chars = len(text) - chinese_chars
    return max(1, int(chinese_chars * 1.5 + other_chars / 4.0))


def context_len(units: Sequence[EvidenceUnit]) -> int:
    return sum(estimate_tokens(u.content) for u in units)


def compute_retrieval_metrics(
    retrieved_titles: Sequence[str],
    relevant_titles: Set[str],
    avg_context_len: float,
    latency_ms: float,
    expanded_nodes: float = 0,
) -> ExperimentMetrics:
    retrieved = list(retrieved_titles)
    relevant = set(relevant_titles)
    
    hits = [1 if t in relevant else 0 for t in retrieved]
    recall = sum(hits) / len(relevant) if relevant else 0.0
    precision = sum(hits) / len(retrieved) if retrieved else 0.0
    
    mrr = 0.0
    for idx, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / idx
            break
    
    dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))
    ideal_hits = [1] * min(len(relevant), len(retrieved))
    ideal_dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(ideal_hits))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    
    precisions = []
    running_hits = 0
    for idx, hit in enumerate(hits, start=1):
        if hit:
            running_hits += 1
            precisions.append(running_hits / idx)
    map_score = sum(precisions) / len(relevant) if relevant else 0.0
    
    return ExperimentMetrics(
        recall=recall,
        precision=precision,
        mrr=mrr,
        ndcg=ndcg,
        map_score=map_score,
        avg_len=avg_context_len,
        time_ms=latency_ms,
        expanded_nodes=expanded_nodes,
    )


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return sum(nums) / len(nums) if nums else None


def round2(value: Optional[float]) -> Optional[float]:
    if value is None or not isinstance(value, (int, float)) or math.isnan(float(value)):
        return None
    return round(float(value), 2)


class KGRAGExperiment:
    """KG-RAG实验"""
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.llm_client: Optional[Any] = None
        self.reranker: Optional[Any] = None
        self.sentence_store: Optional[Any] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.documents: List[Dict[str, Any]] = []
        self.title_to_content: Dict[str, str] = {}
        self._rerank_semaphore = threading.Semaphore(5)
        self._rerank_cache: Dict[str, List[EvidenceUnit]] = {}
    
    def load_resources(self):
        """加载所有资源"""
        print("加载资源...")
        
        with open(self.config.documents_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        self.title_to_content = {
            doc.get("title", ""): doc.get("sentence_total", doc.get("content", ""))
            for doc in self.documents
            if doc.get("title")
        }
        print(f"  文档: {len(self.title_to_content)} 个标题")
        
        self.test_data = pd.read_parquet(self.config.test_data_path)
        if self.config.sample_size < len(self.test_data):
            self.test_data = self.test_data.sample(
                n=self.config.sample_size, random_state=self.config.random_seed
            )
        print(f"  测试数据: {len(self.test_data)} 个样本")
        
        from src.storage.vector_store.faiss_store import FAISSVectorStore
        self.sentence_store = FAISSVectorStore(persist_path=str(self.config.sentence_store_path))
        print("  向量存储: 已加载")
        
        if self.config.use_api_reranker:
            try:
                from src.retrievers.reranker import BGEReranker
                self.reranker = BGEReranker()
                print("  重排序器: BGE API")
            except Exception as e:
                print(f"  重排序器: 加载失败 ({e})，使用词汇回退")
                self.reranker = None
        
        if self.config.run_generation or self.config.run_judge:
            try:
                from src.llms.deepseek_client import DeepSeekClient
                self.llm_client = DeepSeekClient()
                print("  LLM客户端: DeepSeek")
            except Exception as e:
                print(f"  LLM客户端: 加载失败 ({e})")
    
    def get_relevant_titles(self, sample: pd.Series) -> Set[str]:
        supporting_facts = sample.get("supporting_facts", [])
        titles = set()
        for fact in supporting_facts:
            if isinstance(fact, list) and len(fact) > 0:
                titles.add(fact[0])
            elif isinstance(fact, dict):
                titles.add(fact.get("title", ""))
        return titles
    
    def vector_retrieve(self, query: str, k: int = 10) -> List[EvidenceUnit]:
        if self.sentence_store is None:
            return []
        try:
            results = self.sentence_store.search(query, top_k=k)
            return [
                EvidenceUnit(
                    id=f"vec::{r.get('id', i)}",
                    title=r.get("title", ""),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                    source="vector",
                )
                for i, r in enumerate(results)
            ]
        except Exception as e:
            print(f"向量检索失败: {e}")
            return []
    
    def keyword_retrieve(self, query: str, k: int = 10) -> List[EvidenceUnit]:
        query_terms = set(query.lower().split())
        scored: List[Tuple[float, EvidenceUnit]] = []
        
        for title, content in self.title_to_content.items():
            content_terms = set(content.lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                scored.append((
                    overlap / max(len(query_terms), 1),
                    EvidenceUnit(
                        id=f"kw::{title}",
                        title=title,
                        content=content,
                        score=overlap / max(len(query_terms), 1),
                        source="keyword",
                        granularity="paragraph",
                    )
                ))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [unit for _, unit in scored[:k]]
    
    def rerank_units(self, query: str, units: Sequence[EvidenceUnit], top_k: Optional[int] = None) -> List[EvidenceUnit]:
        if not units:
            return []
        limit = top_k or self.config.k3
        
        cache_key = f"{query}::{','.join(u.id for u in units)}::{limit}"
        if cache_key in self._rerank_cache:
            return self._rerank_cache[cache_key]
        
        if self.reranker is not None:
            try:
                with self._rerank_semaphore:
                    search_results = [
                        RerankSearchResult(doc_id=u.id, content=u.content, score=u.score, metadata={"unit": u})
                        for u in units
                    ]
                    reranked = self.reranker.rerank(query, search_results, top_k=limit)
                result = [r.metadata["unit"] for r in reranked]
                self._rerank_cache[cache_key] = result
                return result
            except Exception as exc:
                print(f"API rerank failed: {exc}")
        
        query_terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", query))
        scored: List[Tuple[float, EvidenceUnit]] = []
        for unit in units:
            content_terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", unit.content))
            title_terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", unit.title))
            overlap = len(query_terms & (content_terms | title_terms)) / max(len(query_terms), 1)
            score = 0.7 * unit.score + 0.3 * overlap
            unit.score = score
            scored.append((score, unit))
        scored.sort(key=lambda item: item[0], reverse=True)
        result = [unit for _, unit in scored[:limit]]
        self._rerank_cache[cache_key] = result
        return result
    
    def generate_answer(self, question: str, units: List[EvidenceUnit]) -> str:
        if not self.llm_client:
            return ""
        context = "\n".join(f"[{i+1}] {u.content}" for i, u in enumerate(units[:7]))
        prompt = f"Based on the following context, answer the question briefly.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        try:
            from src.llms.base_client import Message
            response = self.llm_client.generate([Message(role="user", content=prompt)], max_tokens=100)
            return response.content.strip()
        except Exception as e:
            print(f"生成答案失败: {e}")
            return ""
    
    def judge_answer(self, question: str, ground_truth: str, answer: str, units: List[EvidenceUnit]) -> Dict[str, Any]:
        if not self.llm_client:
            return {}
        context = "\n".join(f"[{i+1}] {u.content}" for i, u in enumerate(units[:5]))
        prompt = f"""Evaluate the answer quality. Return JSON with scores 0-1.

Question: {question}
Reference Answer: {ground_truth}
Generated Answer: {answer}
Context: {context}

Return JSON: {{"correctness": 0-1, "faithfulness": 0-1, "answer_relevance": 0-1, "context_relevance": 0-1}}"""
        try:
            from src.llms.base_client import Message
            response = self.llm_client.generate([Message(role="user", content=prompt)], max_tokens=200)
            return json.loads(response.content.strip())
        except Exception as e:
            print(f"评估答案失败: {e}")
            return {}
    
    def retrieve_method(self, query: str) -> MethodResult:
        """向量 + 关键词 + 重排序"""
        start = time.time()
        vec_results = self.vector_retrieve(query, k=self.config.k1)
        kw_results = self.keyword_retrieve(query, k=self.config.k2)
        combined = list(vec_results) + kw_results
        units = self.rerank_units(query, combined)
        elapsed = (time.time() - start) * 1000
        return MethodResult(units=units, stats={"time_ms": elapsed, "avg_len": context_len(units), "expanded_nodes": 0, "route": "kg_rag"})
    
    def evaluate_methods(self, method_fns: Dict[str, Callable[[str], MethodResult]], desc: str = "评估") -> Dict[str, List[Dict[str, Any]]]:
        assert self.test_data is not None
        rows_by_method: Dict[str, List[Dict[str, Any]]] = {name: [] for name in method_fns}
        rows_lock = threading.Lock()
        
        samples = list(self.test_data.iterrows())
        method_names = list(method_fns.keys())
        
        def process_one(idx: int, sample: pd.Series, method_name: str) -> Tuple[str, Dict[str, Any]]:
            fn = method_fns[method_name]
            question = str(sample.get("question", ""))
            ground_truth = str(sample.get("answer", ""))
            relevant_titles = self.get_relevant_titles(sample)
            
            for attempt in range(3):
                try:
                    method_result = fn(question)
                    break
                except Exception as exc:
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    print(f"{method_name} failed: {exc}")
                    method_result = MethodResult(units=[], stats={"error": str(exc), "time_ms": 0.0, "avg_len": 0.0, "expanded_nodes": 0})
            
            retrieved_titles = [u.title for u in method_result.units]
            metrics = compute_retrieval_metrics(
                retrieved_titles=retrieved_titles,
                relevant_titles=relevant_titles,
                avg_context_len=method_result.stats.get("avg_len", context_len(method_result.units)),
                latency_ms=method_result.stats.get("time_ms", 0.0),
                expanded_nodes=method_result.stats.get("expanded_nodes", 0),
            )
            
            answer = self.generate_answer(question, method_result.units) if self.config.run_generation else ""
            semantic = self.judge_answer(question, ground_truth, answer, method_result.units) if self.config.run_judge else {}
            
            result = {
                "id": sample.get("id"),
                "question": question,
                "retrieval_metrics": metrics,
                "semantic_metrics": semantic,
                "stats": method_result.stats,
            }
            return method_name, result
        
        total_tasks = len(samples) * len(method_names)
        
        if self.config.max_workers > 1:
            pbar = tqdm(total=total_tasks, desc=desc)
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {}
                for idx, sample in samples:
                    for method_name in method_names:
                        future = executor.submit(process_one, idx, sample, method_name)
                        futures[future] = (idx, method_name)
                
                for future in as_completed(futures):
                    try:
                        method_name, result = future.result()
                        with rows_lock:
                            rows_by_method[method_name].append(result)
                    except Exception as exc:
                        print(f"异常: {exc}")
                    pbar.update(1)
            pbar.close()
        else:
            for idx, sample in tqdm(samples, desc=desc):
                for method_name in method_names:
                    method_name, result = process_one(idx, sample, method_name)
                    rows_by_method[method_name].append(result)
        
        return rows_by_method
    
    def summarize_methods(self, rows_by_method: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return [
            {"Method": method_name, **self.aggregate_method_rows(rows)}
            for method_name, rows in rows_by_method.items()
        ]
    
    def aggregate_method_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = [row["retrieval_metrics"] for row in rows]
        semantic = [row.get("semantic_metrics", {}) for row in rows]
        return {
            "Recall": round2(mean_or_none(m.recall for m in metrics)) or 0.0,
            "MRR": round2(mean_or_none(m.mrr for m in metrics)) or 0.0,
            "NDCG": round2(mean_or_none(m.ndcg for m in metrics)) or 0.0,
            "Avg Token": round2(mean_or_none(m.avg_len for m in metrics)) or 0.0,
            "correctness": round2(mean_or_none(s.get("correctness") for s in semantic)),
            "faithfulness": round2(mean_or_none(s.get("faithfulness") for s in semantic)),
        }
    
    def write_csv(self, path: Path, rows: List[Dict[str, Any]]):
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(f"  保存: {path}")
    
    def run(self) -> Dict[str, Any]:
        from datetime import datetime
        self.load_resources()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.output_dir / f"exp1_4_kg_rag_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n运行 KG-RAG 实验...")
        rows_by_method = self.evaluate_methods({"KG-RAG": self.retrieve_method}, desc="KG-RAG")
        summary = self.summarize_methods(rows_by_method)
        
        self.write_csv(run_dir / "result.csv", summary)
        
        with open(run_dir / "details.json", "w", encoding="utf-8") as f:
            json.dump(rows_by_method, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n实验完成。结果目录: {run_dir}")
        return {"run_dir": str(run_dir), "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="KG-RAG")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--max-workers", type=int, default=250)
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()
    
    config = RetrievalConfig(
        sample_size=args.sample_size,
        max_workers=args.max_workers,
        run_generation=not args.no_generation,
        run_judge=not args.no_judge,
    )
    
    result = KGRAGExperiment(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
