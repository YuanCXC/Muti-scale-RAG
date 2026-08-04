# -*- coding: utf-8 -*-
"""实验三：父切片映射实验

所有命中的细粒度候选统一映射到父段落。
这是实验6的对比实验，评估父切片映射策略的效果。

评估指标：证据强度分、更新触发率、EM/F1 提升

流程：
1. 句子级向量检索 (FAISS, k1=20)
2. 父切片映射 (句子到段落映射)
3. LLM 证据强度打分 (0-1 分)
4. 低分文档更新 (查询完整内容)
5. LLM 生成最终答案
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics.retrieval_metrics import RetrievalMetrics, RetrievalResult
from src.evaluation.metrics.generation_metrics import GenerationMetrics, GenerationResult
from src.llms.embedding_client import EmbeddingClient
from src.llms.deepseek_client import DeepSeekClient
from src.llms.base_client import Message, LLMResponse
from src.storage.vector_store.faiss_store import FAISSVectorStore
from src.storage.vector_store.base_store import VectorMetadata
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ParentChunkMappingExperiment:
    """父切片映射实验类
    
    将句子级检索结果统一映射到父段落，评估父切片映射策略的效果。
    """
    
    CHECKPOINT_FILE = "checkpoint.json"
    
    def __init__(
        self,
        test_data_path: str,
        output_dir: str,
        documents_path: Optional[str] = None,
        sentence_vector_store_path: Optional[str] = None,
        k1: int = 20,
        evidence_threshold: float = 0.8,
        checkpoint_interval: int = 10,
    ):
        """初始化实验
        
        Args:
            test_data_path: 测试数据路径
            output_dir: 输出目录
            documents_path: 文档数据路径（段落级）
            sentence_vector_store_path: 句子级向量存储路径
            k1: 句子级向量检索返回数量（评估使用的K值）
            evidence_threshold: 证据强度阈值（低于此值触发更新）
            checkpoint_interval: 检查点保存间隔
        """
        self.test_data_path = test_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.documents_path = documents_path
        self.sentence_vector_store_path = sentence_vector_store_path or str(
            Path(test_data_path).parent / "vector_stores" / "single_sentence"
        )
        
        self.k1 = k1
        self.evidence_threshold = evidence_threshold
        self.metrics_calculator = RetrievalMetrics(k_values=[self.k1])
        self.generation_metrics = GenerationMetrics()
        self.checkpoint_interval = checkpoint_interval
        
        self.config = get_config()
        
        self.sentence_vector_store: Optional[FAISSVectorStore] = None
        self.embedding_client: Optional[EmbeddingClient] = None
        self.llm_client: Optional[DeepSeekClient] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.documents: List[Dict] = []
        self.title_to_content: Dict[str, str] = {}
        self.all_titles: Set[str] = set()
        
    def load_documents(self) -> None:
        """加载文档数据（段落级）"""
        if not self.documents_path:
            logger.warning("未提供文档路径，跳过文档加载")
            return
        
        logger.info(f"加载文档数据: {self.documents_path}")
        
        with open(self.documents_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        
        logger.info(f"文档加载完成: {len(self.documents)} 个文档")
        
        self._build_title_index()
    
    def _build_title_index(self) -> None:
        """构建标题索引"""
        logger.info("构建标题索引...")
        
        for doc in self.documents:
            title = doc.get("title", "")
            if title:
                self.all_titles.add(title)
                self.title_to_content[title] = doc.get("sentence_total", doc.get("content", ""))
        
        logger.info(f"标题索引构建完成: {len(self.all_titles)} 个唯一标题")
    
    def load_sentence_vector_store(self) -> None:
        """加载句子级向量存储"""
        logger.info(f"加载句子级向量存储: {self.sentence_vector_store_path}")
        try:
            self.sentence_vector_store = FAISSVectorStore()
            self.sentence_vector_store.load(self.sentence_vector_store_path)
            logger.info(f"句子级向量存储加载成功: vectors={self.sentence_vector_store.count()}")
        except FileNotFoundError:
            logger.warning(f"句子级向量存储文件不存在: {self.sentence_vector_store_path}")
            self.sentence_vector_store = None
    
    def load_embedding_client(self) -> None:
        """加载嵌入客户端"""
        logger.info("加载嵌入客户端...")
        self.embedding_client = EmbeddingClient()
        logger.info(f"嵌入客户端加载完成: dimension={self.embedding_client.dimension}")
    
    def load_llm_client(self) -> None:
        """加载 LLM 客户端"""
        logger.info("加载 LLM 客户端...")
        self.llm_client = DeepSeekClient()
        logger.info(f"LLM 客户端加载完成: model={self.llm_client.model}")
    
    def load_test_data(self) -> None:
        """加载测试数据"""
        logger.info(f"加载测试数据: {self.test_data_path}")
        self.test_data = pd.read_parquet(self.test_data_path)
        logger.info(f"测试数据加载完成: {len(self.test_data)} 条记录")
    
    def step1_sentence_retrieval(self, query: str) -> List[Dict]:
        """Step 1: 句子级向量检索
        
        Args:
            query: 查询文本
            
        Returns:
            检索结果列表（句子级）
        """
        if not self.sentence_vector_store:
            return []
        
        query_vector = self.embedding_client.embed(query)
        results = self.sentence_vector_store.search(query_vector, top_k=self.k1)
        
        all_results = []
        for result in results:
            title = result.metadata.extra.get("title", result.metadata.doc_id)
            content = result.metadata.content
            sentence_id = result.metadata.extra.get("sentence_id", "")
            
            all_results.append({
                "id": result.id,
                "title": title,
                "content": content,
                "score": result.score,
                "source": "sentence_vector",
                "sentence_id": sentence_id,
                "is_sentence_level": True,
            })
        
        return all_results
    
    def step2_parent_chunk_mapping(self, sentence_results: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Step 2: 父切片映射（句子到段落映射）
        
        将句子级检索结果映射回完整段落：
        1. 识别句子级检索结果
        2. 从 title_to_content 获取完整段落
        3. 合并同一标题下的多个句子
        
        Args:
            sentence_results: 句子级检索结果列表
            
        Returns:
            (所有标题列表, 映射后的结果列表)
        """
        merged_dict = {}
        
        for item in sentence_results:
            title = item.get("title", "")
            content = item.get("content", "")
            
            if not title:
                continue
            
            if title not in merged_dict:
                if title in self.title_to_content:
                    full_content = self.title_to_content[title]
                else:
                    full_content = content
                
                merged_dict[title] = {
                    "title": title,
                    "content": full_content,
                    "source": "parent_mapping",
                    "sentence_ids": [item.get("sentence_id", "")],
                    "sentence_count": 1,
                    "original_sentences": [content],
                }
            else:
                merged_dict[title]["sentence_ids"].append(item.get("sentence_id", ""))
                merged_dict[title]["sentence_count"] += 1
                if content not in merged_dict[title]["original_sentences"]:
                    merged_dict[title]["original_sentences"].append(content)
        
        all_titles = []
        final_results = []
        for v in merged_dict.values():
            final_results.append(v)
            all_titles.append(v["title"])
        
        return all_titles, final_results
    
    def step3_llm_scoring(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Step 3: LLM 证据强度打分
        
        对每个上下文块进行证据强度评估，返回 0-1 分。
        
        Args:
            query: 查询文本
            candidates: 候选文档列表
            
        Returns:
            打分后的结果列表
        """
        if not candidates:
            return []
        
        context_chunks = ""
        for i, candidate in enumerate(candidates, 1):
            content = candidate.get("content", "")[:800]
            context_chunks += f"[Context {i}]\n{content}\n\n"
        
        prompt = f"""Your task is to evaluate the strength of evidential support that each given context chunk provides for answering the query, so as to determine whether the current chunk has achieved evidence completeness.
You may be given one or more context chunks. Evaluate only the context chunks that are explicitly provided.

Query:
{query}

Context Chunks:
{context_chunks}

Please evaluate each provided context chunk independently. Do not combine information from different context chunks when assigning scores.
For each context chunk, focus on whether it provides direct, key, and effective evidential support for the query, rather than only judging whether it is sufficient on its own to fully cover the complete answer.

Important considerations:
- Even if a context chunk cannot independently support the complete answer, it should still receive a relatively high score as long as it provides accurate, direct, and important core evidence.
- Do not unduly lower the score of the current chunk simply because the complete answer may still depend on other context chunks.
- A low score should be assigned only when the chunk is weakly related to the query, provides limited evidence, or contains only vague background information.

Please assign a score from 0 to 1, in increments of 0.1, for each context chunk based on relevance, evidence importance, and evidence sufficiency.

Scoring criteria:
- 0.0-0.3: Little or no valid evidence
- 0.4-0.5: Contains some relevant information, but the evidence is weak, limited, or mainly local/background content
- 0.6-0.7: Provides relatively clear and useful evidence and offers strong support for the query, but it is still not sufficiently key or sufficient
- 0.8-1.0: Provides direct, key, and high-value evidential support; even if the complete answer may still require additional context, this chunk itself already has high support value

Important rule:
If a context chunk provides accurate, direct, and important evidence for a key aspect of the query, its score should be 0.8 or higher, even if the complete answer may still require supplementation from other context chunks.

Notes:
- Evaluate each context chunk independently.
- Do not infer information that is not explicitly stated in the text.
- Keep the judgment objective and concise.

Output format:
Return only a JSON array of scores.
Each score must correspond to the provided context chunks in the same order.
Do not return any extra text, explanation, key, or markdown.
"""
        
        try:
            response = self.llm_client.generate([Message(role="user", content=prompt)])
            if response.success:
                scores_raw = response.content.strip()
                
                try:
                    scores = json.loads(scores_raw)
                    if not isinstance(scores, list):
                        scores = [float(s) for s in str(scores).split() if s.replace('.', '').isdigit()]
                except json.JSONDecodeError:
                    match = re.search(r'\[[\d\.\,\s]+\]', scores_raw)
                    if match:
                        scores = json.loads(match.group())
                    else:
                        score_lines = scores_raw.strip().split('\n')
                        scores = []
                        for line in score_lines:
                            line = line.strip()
                            if line:
                                try:
                                    score = float(line)
                                    scores.append(max(0.0, min(1.0, score)))
                                except ValueError:
                                    continue
                
                for i, candidate in enumerate(candidates):
                    if i < len(scores):
                        candidate["evidence_score"] = max(0.0, min(1.0, float(scores[i])))
                    else:
                        candidate["evidence_score"] = 0.5
            else:
                for candidate in candidates:
                    candidate["evidence_score"] = 0.5
        except Exception as e:
            logger.warning(f"LLM 打分失败: {e}")
            for candidate in candidates:
                candidate["evidence_score"] = 0.5
        
        return candidates
    
    def step4_update_low_score_docs(self, query: str, candidates: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Step 4: 低分文档更新
        
        对证据强度低于阈值的文档，从 title_to_content 获取完整内容。
        
        Args:
            query: 查询文本
            candidates: 候选文档列表
            
        Returns:
            (更新后的结果列表, 更新统计信息)
        """
        update_stats = {
            "total_candidates": len(candidates),
            "low_score_count": 0,
            "updated_count": 0,
            "update_triggered": False,
        }
        
        for candidate in candidates:
            evidence_score = candidate.get("evidence_score", 0)
            if evidence_score < self.evidence_threshold:
                update_stats["low_score_count"] += 1
                title = candidate.get("title", "")
                
                if title in self.title_to_content:
                    full_content = self.title_to_content[title]
                    if full_content != candidate.get("content", ""):
                        candidate["content"] = full_content
                        candidate["updated"] = True
                        candidate["update_reason"] = "low_score_enhanced"
                        update_stats["updated_count"] += 1
                    else:
                        candidate["updated"] = False
                        candidate["update_reason"] = "already_full_content"
                else:
                    candidate["updated"] = False
                    candidate["update_reason"] = "no_full_content_available"
            else:
                candidate["updated"] = False
                candidate["update_reason"] = "score_above_threshold"
        
        update_stats["update_triggered"] = update_stats["updated_count"] > 0
        
        return candidates, update_stats
    
    def step5_generate_answer(self, query: str, candidates: List[Dict]) -> str:
        """Step 5: LLM 生成最终答案
        
        Args:
            query: 查询文本
            candidates: 候选文档列表
            
        Returns:
            生成的答案
        """
        context_parts = []
        for i, c in enumerate(candidates[:3]):
            content = c.get("content", "")
            context_parts.append(f"[文档{i+1}] {content[:300]}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""基于以下文档内容回答问题。如果文档中没有相关信息，请说明。

问题: {query}

文档内容:
{context}

答案:"""
        
        try:
            response = self.llm_client.generate([Message(role="user", content=prompt)])
            if response.success:
                return response.content
        except Exception as e:
            logger.warning(f"答案生成失败: {e}")
        
        return ""
    
    def run_parent_chunk_mapping_retrieval(self, query: str) -> Tuple[List[str], Dict]:
        """运行完整的父切片映射检索流程
        
        Args:
            query: 查询文本
            
        Returns:
            (检索到的标题列表, 流程统计信息)
        """
        stats = {
            "sentence_results": 0,
            "mapped_results": 0,
            "final_results": 0,
            "latency": {},
            "evidence_scores": [],
            "update_stats": {},
        }
        
        start_time = time.time()
        sentence_results = self.step1_sentence_retrieval(query)
        stats["latency"]["sentence_retrieval"] = time.time() - start_time
        stats["sentence_results"] = len(sentence_results)
        
        start_time = time.time()
        all_titles, mapped = self.step2_parent_chunk_mapping(sentence_results)
        stats["latency"]["mapping"] = time.time() - start_time
        stats["mapped_results"] = len(mapped)
        
        start_time = time.time()
        scored = self.step3_llm_scoring(query, mapped)
        stats["latency"]["scoring"] = time.time() - start_time
        stats["evidence_scores"] = [c.get("evidence_score", 0) for c in scored]
        
        start_time = time.time()
        updated, update_stats = self.step4_update_low_score_docs(query, scored)
        stats["latency"]["update"] = time.time() - start_time
        stats["update_stats"] = update_stats
        
        stats["final_results"] = len(updated)
        
        retrieved_titles = [c.get("title", "") for c in updated if c.get("title")]
        
        return retrieved_titles, stats
    
    def get_relevant_titles(self, row: pd.Series) -> Set[str]:
        """获取相关文档标题"""
        supporting_facts = row.get("supporting_facts", {})
        titles = supporting_facts.get("title", [])
        
        if isinstance(titles, np.ndarray):
            titles = titles.tolist()
        
        return set(titles)
    
    def evaluate_single_query(
        self,
        query: str,
        ground_truth: str,
        relevant_titles: Set[str],
        max_k: int,
    ) -> Dict[str, Any]:
        """评估单个查询
        
        Args:
            query: 查询文本
            ground_truth: 标准答案
            relevant_titles: 相关标题集合
            max_k: 最大K值
            
        Returns:
            评估结果字典
        """
        retrieved_titles, stats = self.run_parent_chunk_mapping_retrieval(query)
        
        retrieved_list = retrieved_titles[:max_k]
        
        result = self.metrics_calculator.compute(retrieved_list, list(relevant_titles))
        
        retrieved_set = set(retrieved_titles)
        title_recall = len(retrieved_set & relevant_titles) / len(relevant_titles) if relevant_titles else 0
        title_precision = len(retrieved_set & relevant_titles) / len(retrieved_set) if retrieved_set else 0
        
        generated_answer = self.step5_generate_answer(query, [{"title": t, "content": self.title_to_content.get(t, t)} for t in retrieved_titles[:5]])
        
        gen_result = self.generation_metrics.compute(
            predicted=generated_answer,
            ground_truth=ground_truth,
            compute_semantic=True,
            embedding_client=self.embedding_client,
        )
        
        evidence_scores = stats.get("evidence_scores", [])
        avg_evidence_score = np.mean(evidence_scores) if evidence_scores else 0
        min_evidence_score = np.min(evidence_scores) if evidence_scores else 0
        
        update_stats = stats.get("update_stats", {})
        update_trigger_rate = update_stats.get("updated_count", 0) / update_stats.get("total_candidates", 1) if update_stats.get("total_candidates", 0) > 0 else 0
        
        return {
            "retrieved_titles": retrieved_titles[:20],
            "relevant_titles": list(relevant_titles),
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "metrics": result.to_dict(),
            "generation_metrics": gen_result.to_dict(),
            "title_recall": title_recall,
            "title_precision": title_precision,
            "stats": stats,
            "avg_evidence_score": avg_evidence_score,
            "min_evidence_score": min_evidence_score,
            "update_trigger_rate": update_trigger_rate,
        }
    
    def _save_checkpoint(
        self,
        processed_indices: List[int],
        all_results: List[Dict],
        aggregated_metrics: Dict,
        sample_size: Optional[int],
    ) -> None:
        """保存检查点"""
        checkpoint = {
            "processed_indices": processed_indices,
            "all_results": all_results,
            "aggregated_metrics": aggregated_metrics,
            "sample_size": sample_size,
            "timestamp": datetime.now().isoformat(),
        }
        
        checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False)
        
        logger.info(f"检查点已保存: 已处理 {len(processed_indices)} 条")
    
    def _load_checkpoint(self) -> Optional[Dict]:
        """加载检查点"""
        checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            
            if "aggregated_metrics" in checkpoint:
                agg_metrics = checkpoint["aggregated_metrics"]
                if "recall_at_k" in agg_metrics:
                    agg_metrics["recall_at_k"] = {int(k): v for k, v in agg_metrics["recall_at_k"].items()}
                if "precision_at_k" in agg_metrics:
                    agg_metrics["precision_at_k"] = {int(k): v for k, v in agg_metrics["precision_at_k"].items()}
            
            logger.info(f"从检查点恢复: 已处理 {len(checkpoint['processed_indices'])} 条")
            return checkpoint
        except Exception as e:
            logger.warning(f"加载检查点失败: {e}")
            return None
    
    def _clear_checkpoint(self) -> None:
        """清除检查点"""
        checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("检查点已清除")
    
    def run_experiment(
        self,
        sample_size: Optional[int] = None,
        save_details: bool = True,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """运行实验"""
        logger.info("开始运行父切片映射实验...")
        
        self.load_documents()
        self.load_sentence_vector_store()
        self.load_embedding_client()
        self.load_llm_client()
        self.load_test_data()
        
        test_data = self.test_data
        if sample_size:
            test_data = test_data.sample(n=sample_size, random_state=42)
            logger.info(f"采样 {sample_size} 条数据进行测试")
        
        max_k = self.k1
        test_indices = test_data.index.tolist()
        
        processed_indices: List[int] = []
        all_results: List[Dict] = []
        aggregated_metrics = {
            "recall_at_k": [],
            "precision_at_k": [],
            "mrr": [],
            "ndcg": [],
            "map_score": [],
            "hit_rate": [],
            "title_recall": [],
            "title_precision": [],
            "exact_match": [],
            "f1_score": [],
            "semantic_similarity": [],
            "avg_evidence_score": [],
            "min_evidence_score": [],
            "update_trigger_rate": [],
            "sentence_retrieval_latency": [],
            "mapping_latency": [],
            "scoring_latency": [],
            "update_latency": [],
            "total_latency": [],
        }
        
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                if checkpoint.get("sample_size") == sample_size:
                    processed_indices = checkpoint["processed_indices"]
                    all_results = checkpoint["all_results"]
                    aggregated_metrics = checkpoint["aggregated_metrics"]
                    logger.info(f"从检查点恢复成功，跳过 {len(processed_indices)} 条已处理数据")
                else:
                    logger.info("采样数量不匹配，从头开始")
        
        remaining_indices = [idx for idx in test_indices if idx not in processed_indices]
        
        if not remaining_indices:
            logger.info("所有数据已处理完成")
        else:
            pbar = tqdm(remaining_indices, desc="评估进度", initial=len(processed_indices), total=len(test_indices))
            
            for idx in remaining_indices:
                row = test_data.loc[idx]
                query = row["question"]
                ground_truth = row.get("answer", "")
                relevant_titles = self.get_relevant_titles(row)
                
                try:
                    result = self.evaluate_single_query(query, ground_truth, relevant_titles, max_k)
                    
                    metrics = result["metrics"]
                    aggregated_metrics["recall_at_k"].append(
                        metrics["recall_at_k"].get(self.k1, 0)
                    )
                    aggregated_metrics["precision_at_k"].append(
                        metrics["precision_at_k"].get(self.k1, 0)
                    )
                    
                    aggregated_metrics["mrr"].append(metrics["mrr"])
                    aggregated_metrics["ndcg"].append(metrics["ndcg"])
                    aggregated_metrics["map_score"].append(metrics["map_score"])
                    aggregated_metrics["hit_rate"].append(metrics["hit_rate"])
                    aggregated_metrics["title_recall"].append(result["title_recall"])
                    aggregated_metrics["title_precision"].append(result["title_precision"])
                    
                    gen_metrics = result["generation_metrics"]
                    aggregated_metrics["exact_match"].append(gen_metrics["exact_match"])
                    aggregated_metrics["f1_score"].append(gen_metrics["f1_score"])
                    aggregated_metrics["semantic_similarity"].append(gen_metrics["semantic_similarity"])
                    
                    aggregated_metrics["avg_evidence_score"].append(result["avg_evidence_score"])
                    aggregated_metrics["min_evidence_score"].append(result["min_evidence_score"])
                    aggregated_metrics["update_trigger_rate"].append(result["update_trigger_rate"])
                    
                    stats = result.get("stats", {})
                    latency = stats.get("latency", {})
                    aggregated_metrics["sentence_retrieval_latency"].append(
                        latency.get("sentence_retrieval", 0)
                    )
                    aggregated_metrics["mapping_latency"].append(
                        latency.get("mapping", 0)
                    )
                    aggregated_metrics["scoring_latency"].append(
                        latency.get("scoring", 0)
                    )
                    aggregated_metrics["update_latency"].append(
                        latency.get("update", 0)
                    )
                    aggregated_metrics["total_latency"].append(
                        sum(latency.values())
                    )
                    
                    if save_details:
                        all_results.append({
                            "id": row.get("id", idx),
                            "question": query,
                            "answer": ground_truth,
                            "generated_answer": result["generated_answer"],
                            "relevant_titles": list(relevant_titles),
                            "retrieved_titles": result["retrieved_titles"],
                            "metrics": metrics,
                            "generation_metrics": gen_metrics,
                            "title_recall": result["title_recall"],
                            "title_precision": result["title_precision"],
                            "avg_evidence_score": result["avg_evidence_score"],
                            "min_evidence_score": result["min_evidence_score"],
                            "update_trigger_rate": result["update_trigger_rate"],
                            "stats": stats,
                        })
                    
                    processed_indices.append(idx)
                    
                    if len(processed_indices) % self.checkpoint_interval == 0:
                        self._save_checkpoint(
                            processed_indices, all_results, aggregated_metrics, sample_size
                        )
                        
                except Exception as e:
                    logger.error(f"处理查询失败 (idx={idx}): {e}")
                    continue
                
                pbar.update(1)
            
            pbar.close()
        
        final_metrics = {
            "recall_at_k": {
                k: np.mean(v) for k, v in aggregated_metrics["recall_at_k"].items()
            },
            "precision_at_k": {
                k: np.mean(v) for k, v in aggregated_metrics["precision_at_k"].items()
            },
            "mrr": np.mean(aggregated_metrics["mrr"]),
            "ndcg": np.mean(aggregated_metrics["ndcg"]),
            "map_score": np.mean(aggregated_metrics["map_score"]),
            "hit_rate": np.mean(aggregated_metrics["hit_rate"]),
            "title_recall": np.mean(aggregated_metrics["title_recall"]),
            "title_precision": np.mean(aggregated_metrics["title_precision"]),
            "exact_match": np.mean(aggregated_metrics["exact_match"]),
            "f1_score": np.mean(aggregated_metrics["f1_score"]),
            "semantic_similarity": np.nanmean(aggregated_metrics["semantic_similarity"]),
            "avg_evidence_score": np.mean(aggregated_metrics["avg_evidence_score"]),
            "min_evidence_score": np.mean(aggregated_metrics["min_evidence_score"]),
            "update_trigger_rate": np.mean(aggregated_metrics["update_trigger_rate"]),
            "avg_sentence_retrieval_latency": np.mean(aggregated_metrics["sentence_retrieval_latency"]),
            "avg_mapping_latency": np.mean(aggregated_metrics["mapping_latency"]),
            "avg_scoring_latency": np.mean(aggregated_metrics["scoring_latency"]),
            "avg_update_latency": np.mean(aggregated_metrics["update_latency"]),
            "avg_total_latency": np.mean(aggregated_metrics["total_latency"]),
        }
        
        baseline_recall = 0.1
        recall_improvement = {
            k: (v - baseline_recall) / baseline_recall * 100 if baseline_recall > 0 else 0
            for k, v in final_metrics["recall_at_k"].items()
        }
        final_metrics["recall_improvement"] = recall_improvement
        
        experiment_result = {
            "experiment_name": "parent_chunk_mapping",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "test_data_path": self.test_data_path,
                "documents_path": self.documents_path,
                "sentence_vector_store_path": self.sentence_vector_store_path,
                "k1": self.k1,
                "evidence_threshold": self.evidence_threshold,
                "sample_size": sample_size or len(test_data),
                "total_test_samples": len(test_data),
            },
            "metrics": final_metrics,
            "details": all_results if save_details else None,
        }
        
        self._save_results(experiment_result, save_details)
        
        self._clear_checkpoint()
        
        return experiment_result
    
    def _save_results(self, results: Dict[str, Any], save_details: bool) -> None:
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary_path = self.output_dir / f"experiment_summary_{timestamp}.json"
        summary = {k: v for k, v in results.items() if k != "details"}
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"实验摘要已保存: {summary_path}")
        
        if save_details and results.get("details"):
            details_path = self.output_dir / f"experiment_details_{timestamp}.json"
            with open(details_path, "w", encoding="utf-8") as f:
                json.dump(results["details"], f, ensure_ascii=False, indent=2)
            logger.info(f"详细结果已保存: {details_path}")
        
        metrics = results["metrics"]
        metrics_rows = [
            {
                "Metric": "Recall@K",
                **{f"K={k}": v for k, v in metrics["recall_at_k"].items()},
            },
            {
                "Metric": "Precision@K",
                **{f"K={k}": v for k, v in metrics["precision_at_k"].items()},
            },
            {
                "Metric": "Recall提升率(%)",
                **{f"K={k}": v for k, v in metrics["recall_improvement"].items()},
            },
            {
                "Metric": "MRR",
                "Value": metrics["mrr"],
            },
            {
                "Metric": "NDCG",
                "Value": metrics["ndcg"],
            },
            {
                "Metric": "MAP",
                "Value": metrics["map_score"],
            },
            {
                "Metric": "Hit Rate",
                "Value": metrics["hit_rate"],
            },
            {
                "Metric": "Title Recall",
                "Value": metrics["title_recall"],
            },
            {
                "Metric": "Title Precision",
                "Value": metrics["title_precision"],
            },
            {
                "Metric": "Exact Match",
                "Value": metrics["exact_match"],
            },
            {
                "Metric": "F1 Score",
                "Value": metrics["f1_score"],
            },
            {
                "Metric": "Semantic Similarity",
                "Value": metrics["semantic_similarity"],
            },
            {
                "Metric": "平均证据强度分",
                "Value": metrics["avg_evidence_score"],
            },
            {
                "Metric": "最低证据强度分",
                "Value": metrics["min_evidence_score"],
            },
            {
                "Metric": "更新触发率",
                "Value": metrics["update_trigger_rate"],
            },
            {
                "Metric": "平均句子检索延迟(s)",
                "Value": metrics["avg_sentence_retrieval_latency"],
            },
            {
                "Metric": "平均映射延迟(s)",
                "Value": metrics["avg_mapping_latency"],
            },
            {
                "Metric": "平均打分延迟(s)",
                "Value": metrics["avg_scoring_latency"],
            },
            {
                "Metric": "平均更新延迟(s)",
                "Value": metrics["avg_update_latency"],
            },
            {
                "Metric": "平均总延迟(s)",
                "Value": metrics["avg_total_latency"],
            },
        ]
        
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = self.output_dir / f"metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"指标表格已保存: {metrics_path}")
        
        self._print_results(results)
    
    def _print_results(self, results: Dict[str, Any]) -> None:
        """打印实验结果"""
        print("\n" + "=" * 60)
        print("实验结果 - 父切片映射实验")
        print("=" * 60)
        
        config = results["config"]
        metrics = results["metrics"]
        
        print(f"\n【配置参数】")
        print(f"  句子检索 K1: {config['k1']}")
        print(f"  证据强度阈值: {config['evidence_threshold']}")
        
        print("\n【Recall@K】")
        for k, v in metrics["recall_at_k"].items():
            print(f"  Recall@{k}: {v:.4f}")
        
        print("\n【Precision@K】")
        for k, v in metrics["precision_at_k"].items():
            print(f"  Precision@{k}: {v:.4f}")
        
        print("\n【Recall提升率】")
        for k, v in metrics["recall_improvement"].items():
            print(f"  Recall@{k} 提升: {v:.2f}%")
        
        print("\n【综合指标】")
        print(f"  MRR: {metrics['mrr']:.4f}")
        print(f"  NDCG: {metrics['ndcg']:.4f}")
        print(f"  MAP: {metrics['map_score']:.4f}")
        print(f"  Hit Rate: {metrics['hit_rate']:.4f}")
        
        print("\n【标题级别指标】")
        print(f"  Title Recall: {metrics['title_recall']:.4f}")
        print(f"  Title Precision: {metrics['title_precision']:.4f}")
        
        print("\n【生成指标】")
        print(f"  Exact Match: {metrics['exact_match']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Semantic Similarity: {metrics['semantic_similarity']:.4f}")
        
        print("\n【证据强度指标】")
        print(f"  平均证据强度分: {metrics['avg_evidence_score']:.4f}")
        print(f"  最低证据强度分: {metrics['min_evidence_score']:.4f}")
        print(f"  更新触发率: {metrics['update_trigger_rate']:.4f}")
        
        print("\n【延迟指标】")
        print(f"  平均句子检索延迟: {metrics['avg_sentence_retrieval_latency']:.4f}s")
        print(f"  平均映射延迟: {metrics['avg_mapping_latency']:.4f}s")
        print(f"  平均打分延迟: {metrics['avg_scoring_latency']:.4f}s")
        print(f"  平均更新延迟: {metrics['avg_update_latency']:.4f}s")
        print(f"  平均总延迟: {metrics['avg_total_latency']:.4f}s")
        
        print("\n" + "=" * 60)


def main():
    """主函数"""
    test_data_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/validation-00000-of-00001.parquet"
    documents_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/valid_title_sentence.json"
    sentence_vector_store_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/vector_stores/single_sentence"
    output_dir = "e:/Code_Personal/Subject/test02/experiments/exp3_unified_chunking"
    
    experiment = ParentChunkMappingExperiment(
        test_data_path=test_data_path,
        output_dir=output_dir,
        documents_path=documents_path,
        sentence_vector_store_path=sentence_vector_store_path,
        k1=7,
        evidence_threshold=0.8,
        checkpoint_interval=10,
    )
    
    results = experiment.run_experiment(
        sample_size=None,
        save_details=True,
        resume=True,
    )


if __name__ == "__main__":
    main()
