# -*- coding: utf-8 -*-
"""实验二：细粒度向量检索（句子级别）

使用 FAISS 向量存储进行句子级别的细粒度检索实验。
评估指标：Recall, Precision, MRR, NDCG, MAP, EM, F1 Score, Semantic Similarity
支持断点续跑
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics.retrieval_metrics import RetrievalMetrics, RetrievalResult
from src.evaluation.metrics.generation_metrics import GenerationMetrics, GenerationResult
from src.llms.deepseek_client import DeepSeekClient
from src.llms.base_client import Message
from src.llms.embedding_client import EmbeddingClient
from src.storage.vector_store.faiss_store import FAISSVectorStore
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FineGrainedVectorRetrievalExperiment:
    """细粒度向量检索实验类（句子级别）"""
    
    CHECKPOINT_FILE = "checkpoint.json"
    
    def __init__(
        self,
        vector_store_path: str,
        test_data_path: str,
        output_dir: str,
        top_k_values: Optional[List[int]] = None,
        checkpoint_interval: int = 10,
    ):
        """初始化实验
        
        Args:
            vector_store_path: 向量存储路径
            test_data_path: 测试数据路径
            output_dir: 输出目录
            top_k_values: 评估的 K 值列表
            checkpoint_interval: 检查点保存间隔
        """
        self.vector_store_path = vector_store_path
        self.test_data_path = test_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.top_k_values = top_k_values or [1, 3, 5, 10, 20, 50]
        self.metrics_calculator = RetrievalMetrics(k_values=self.top_k_values)
        self.checkpoint_interval = checkpoint_interval
        
        self.config = get_config()
        self.vector_store: Optional[FAISSVectorStore] = None
        self.embedding_client: Optional[EmbeddingClient] = None
        self.llm_client: Optional[DeepSeekClient] = None
        self.generation_metrics = GenerationMetrics()
        self.test_data: Optional[pd.DataFrame] = None
        self.title_to_doc_ids: Dict[str, Set[str]] = {}
        self.title_to_content: Dict[str, str] = {}
        
    def load_vector_store(self) -> None:
        """加载向量存储"""
        logger.info(f"加载向量存储: {self.vector_store_path}")
        
        self.vector_store = FAISSVectorStore()
        self.vector_store.load(self.vector_store_path)
        
        stats = self.vector_store.get_stats()
        logger.info(f"向量存储统计: {stats}")
        
        self._build_title_index()
        
    def _build_title_index(self) -> None:
        """构建标题到文档ID的索引"""
        logger.info("构建标题索引...")
        
        for vec_id, metadata in self.vector_store._id_to_metadata.items():
            title = metadata.extra.get("title", "")
            if title:
                if title not in self.title_to_doc_ids:
                    self.title_to_doc_ids[title] = set()
                self.title_to_doc_ids[title].add(vec_id)
                self.title_to_content[title] = metadata.content
        
        logger.info(f"标题索引构建完成: {len(self.title_to_doc_ids)} 个唯一标题")
    
    def load_embedding_client(self) -> None:
        """加载嵌入客户端"""
        logger.info("加载嵌入客户端...")
        self.embedding_client = EmbeddingClient()
        logger.info(f"嵌入客户端加载完成: dimension={self.embedding_client.dimension}")
    
    def load_llm_client(self) -> None:
        """加载LLM客户端"""
        logger.info("加载LLM客户端...")
        self.llm_client = DeepSeekClient()
        logger.info("LLM客户端加载完成")
    
    def load_test_data(self) -> None:
        """加载测试数据"""
        logger.info(f"加载测试数据: {self.test_data_path}")
        
        self.test_data = pd.read_parquet(self.test_data_path)
        logger.info(f"测试数据加载完成: {len(self.test_data)} 条记录")
        
    def retrieve(self, query: str, top_k: int) -> List[str]:
        """执行向量检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索到的文档ID列表
        """
        query_embedding = self.embedding_client.embed(query)
        
        if len(query_embedding.shape) == 2:
            query_embedding = query_embedding[0]
        
        results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
        )
        
        return [r.id for r in results]
    
    def get_relevant_titles(self, row: pd.Series) -> Set[str]:
        """获取相关文档标题
        
        Args:
            row: 测试数据行
            
        Returns:
            相关文档标题集合
        """
        supporting_facts = row.get("supporting_facts", {})
        titles = supporting_facts.get("title", [])
        
        if isinstance(titles, np.ndarray):
            titles = titles.tolist()
        
        return set(titles)
    
    def get_relevant_doc_ids(self, relevant_titles: Set[str]) -> Set[str]:
        """根据标题获取相关文档ID
        
        Args:
            relevant_titles: 相关标题集合
            
        Returns:
            相关文档ID集合
        """
        doc_ids = set()
        for title in relevant_titles:
            if title in self.title_to_doc_ids:
                doc_ids.update(self.title_to_doc_ids[title])
        return doc_ids
    
    def generate_answer(self, query: str, retrieved_titles: List[str]) -> str:
        """基于检索结果生成答案
        
        Args:
            query: 查询文本
            retrieved_titles: 检索到的标题列表
            
        Returns:
            生成的答案
        """
        contexts = []
        for title in retrieved_titles[:5]:
            if title in self.title_to_content:
                contexts.append(f"【{title}】\n{self.title_to_content[title]}")
        
        if not contexts:
            return ""
        
        context_text = "\n\n".join(contexts)
        
        messages = [
            Message(
                role="system",
                content="你是一个专业的问答助手。请根据提供的上下文信息回答用户问题。答案应该简洁准确。"
            ),
            Message(
                role="user",
                content=f"上下文信息：\n{context_text}\n\n问题：{query}\n\n请根据上下文信息回答问题："
            )
        ]
        
        try:
            response = self.llm_client.generate(messages)
            if response.success:
                return response.content
            return ""
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return ""
    
    def evaluate_single_query(
        self,
        query: str,
        relevant_titles: Set[str],
        max_k: int,
        ground_truth: str = "",
    ) -> Dict[str, Any]:
        """评估单个查询
        
        Args:
            query: 查询文本
            relevant_titles: 相关标题集合
            max_k: 最大K值
            ground_truth: 真实答案
            
        Returns:
            评估结果字典
        """
        retrieved_ids = self.retrieve(query, max_k)
        
        relevant_doc_ids = self.get_relevant_doc_ids(relevant_titles)
        
        result = self.metrics_calculator.compute(retrieved_ids, list(relevant_doc_ids))
        
        retrieved_titles = set()
        for vec_id in retrieved_ids:
            metadata = self.vector_store._id_to_metadata.get(vec_id)
            if metadata:
                title = metadata.extra.get("title", "")
                if title:
                    retrieved_titles.add(title)
        
        title_recall = len(retrieved_titles & relevant_titles) / len(relevant_titles) if relevant_titles else 0
        title_precision = len(retrieved_titles & relevant_titles) / len(retrieved_titles) if retrieved_titles else 0
        
        generated_answer = ""
        generation_metrics = {}
        
        if self.llm_client and ground_truth:
            generated_answer = self.generate_answer(query, list(retrieved_titles))
            
            if generated_answer:
                gen_result = self.generation_metrics.compute(
                    predicted=generated_answer,
                    ground_truth=ground_truth,
                    compute_semantic=True,
                    embedding_client=self.embedding_client,
                )
                generation_metrics = gen_result.to_dict()
        
        return {
            "retrieved_ids": retrieved_ids[:10],
            "relevant_titles": list(relevant_titles),
            "retrieved_titles": list(retrieved_titles),
            "metrics": result.to_dict(),
            "title_recall": title_recall,
            "title_precision": title_precision,
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "generation_metrics": generation_metrics,
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
            
            # 修复 JSON 序列化导致的整数键变成字符串键的问题
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
        """运行实验
        
        Args:
            sample_size: 采样数量（None表示全部）
            save_details: 是否保存详细结果
            resume: 是否从检查点恢复
            
        Returns:
            实验结果字典
        """
        logger.info("开始运行实验...")
        
        self.load_vector_store()
        self.load_embedding_client()
        self.load_llm_client()
        self.load_test_data()
        
        test_data = self.test_data
        if sample_size:
            test_data = test_data.sample(n=sample_size, random_state=42)
            logger.info(f"采样 {sample_size} 条数据进行测试")
        
        max_k = max(self.top_k_values)
        test_indices = test_data.index.tolist()
        
        processed_indices: List[int] = []
        all_results: List[Dict] = []
        aggregated_metrics = {
            "recall_at_k": {k: [] for k in self.top_k_values},
            "precision_at_k": {k: [] for k in self.top_k_values},
            "mrr": [],
            "ndcg": [],
            "map_score": [],
            "hit_rate": [],
            "title_recall": [],
            "title_precision": [],
            "exact_match": [],
            "f1_score": [],
            "semantic_similarity": [],
        }
        
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                if checkpoint.get("sample_size") == sample_size:
                    processed_indices = checkpoint["processed_indices"]
                    all_results = checkpoint["all_results"]
                    aggregated_metrics = checkpoint["aggregated_metrics"]
                    
                    for k in self.top_k_values:
                        if k not in aggregated_metrics["recall_at_k"]:
                            aggregated_metrics["recall_at_k"][k] = []
                        if k not in aggregated_metrics["precision_at_k"]:
                            aggregated_metrics["precision_at_k"][k] = []
                    
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
                relevant_titles = self.get_relevant_titles(row)
                ground_truth = row.get("answer", "")
                
                try:
                    result = self.evaluate_single_query(query, relevant_titles, max_k, ground_truth)
                    
                    metrics = result["metrics"]
                    for k in self.top_k_values:
                        aggregated_metrics["recall_at_k"][k].append(
                            metrics["recall_at_k"].get(k, 0)
                        )
                        aggregated_metrics["precision_at_k"][k].append(
                            metrics["precision_at_k"].get(k, 0)
                        )
                    
                    aggregated_metrics["mrr"].append(metrics["mrr"])
                    aggregated_metrics["ndcg"].append(metrics["ndcg"])
                    aggregated_metrics["map_score"].append(metrics["map_score"])
                    aggregated_metrics["hit_rate"].append(metrics["hit_rate"])
                    aggregated_metrics["title_recall"].append(result["title_recall"])
                    aggregated_metrics["title_precision"].append(result["title_precision"])
                    
                    if result["generation_metrics"]:
                        aggregated_metrics["exact_match"].append(
                            result["generation_metrics"].get("exact_match", 0)
                        )
                        aggregated_metrics["f1_score"].append(
                            result["generation_metrics"].get("f1_score", 0)
                        )
                        aggregated_metrics["semantic_similarity"].append(
                            result["generation_metrics"].get("semantic_similarity", 0)
                        )
                    
                    if save_details:
                        all_results.append({
                            "id": row.get("id", idx),
                            "question": query,
                            "answer": ground_truth,
                            "relevant_titles": list(relevant_titles),
                            "retrieved_titles": result["retrieved_titles"],
                            "metrics": metrics,
                            "title_recall": result["title_recall"],
                            "title_precision": result["title_precision"],
                            "generated_answer": result["generated_answer"],
                            "generation_metrics": result["generation_metrics"],
                        })
                    
                    processed_indices.append(idx)
                    
                    if len(processed_indices) % self.checkpoint_interval == 0:
                        self._save_checkpoint(
                            processed_indices, all_results, aggregated_metrics, sample_size
                        )
                        
                except Exception as e:
                    import traceback
                    logger.error(f"处理查询失败 (idx={idx}): {e}")
                    logger.error(traceback.format_exc())
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
            "exact_match": np.mean(aggregated_metrics["exact_match"]) if aggregated_metrics["exact_match"] else 0,
            "f1_score": np.mean(aggregated_metrics["f1_score"]) if aggregated_metrics["f1_score"] else 0,
            "semantic_similarity": np.nanmean(aggregated_metrics["semantic_similarity"]) if aggregated_metrics["semantic_similarity"] else 0.0,
        }
        
        experiment_result = {
            "experiment_name": "fine_grained_vector_retrieval",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "vector_store_path": self.vector_store_path,
                "test_data_path": self.test_data_path,
                "top_k_values": self.top_k_values,
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
        
        metrics_df = pd.DataFrame([{
            "Metric": "Recall@K",
            **{f"K={k}": v for k, v in results["metrics"]["recall_at_k"].items()},
        }, {
            "Metric": "Precision@K",
            **{f"K={k}": v for k, v in results["metrics"]["precision_at_k"].items()},
        }, {
            "Metric": "MRR",
            "Value": results["metrics"]["mrr"],
        }, {
            "Metric": "NDCG",
            "Value": results["metrics"]["ndcg"],
        }, {
            "Metric": "MAP",
            "Value": results["metrics"]["map_score"],
        }, {
            "Metric": "Hit Rate",
            "Value": results["metrics"]["hit_rate"],
        }, {
            "Metric": "Title Recall",
            "Value": results["metrics"]["title_recall"],
        }, {
            "Metric": "Title Precision",
            "Value": results["metrics"]["title_precision"],
        }, {
            "Metric": "Exact Match",
            "Value": results["metrics"]["exact_match"],
        }, {
            "Metric": "F1 Score",
            "Value": results["metrics"]["f1_score"],
        }, {
            "Metric": "Semantic Similarity",
            "Value": results["metrics"]["semantic_similarity"],
        }])
        
        metrics_path = self.output_dir / f"metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"指标表格已保存: {metrics_path}")
        
        self._print_results(results)
    
    def _print_results(self, results: Dict[str, Any]) -> None:
        """打印实验结果"""
        print("\n" + "=" * 60)
        print("实验结果 - 细粒度向量检索（句子级别）")
        print("=" * 60)
        
        metrics = results["metrics"]
        
        print("\n【Recall@K】")
        for k, v in metrics["recall_at_k"].items():
            print(f"  Recall@{k}: {v:.4f}")
        
        print("\n【Precision@K】")
        for k, v in metrics["precision_at_k"].items():
            print(f"  Precision@{k}: {v:.4f}")
        
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
        
        print("\n" + "=" * 60)


def main():
    """主函数"""
    vector_store_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/vector_stores/single_sentence"
    test_data_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/validation-00000-of-00001.parquet"
    output_dir = "e:/Code_Personal/Subject/test02/experiments/exp2_fine_grained_vector_retrieval"
    
    experiment = FineGrainedVectorRetrievalExperiment(
        vector_store_path=vector_store_path,
        test_data_path=test_data_path,
        output_dir=output_dir,
        top_k_values=[1, 3, 5, 7, 10],
        checkpoint_interval=10,
    )
    
    results = experiment.run_experiment(
        sample_size=None,
        save_details=True,
        resume=True,
    )


if __name__ == "__main__":
    main()
