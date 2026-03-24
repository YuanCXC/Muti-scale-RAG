# -*- coding: utf-8 -*-
"""RAG 评估器模块

提供 RAG 系统的完整评估功能。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import json
import time
from pathlib import Path
from tqdm import tqdm

from src.evaluation.metrics import RetrievalMetrics, GenerationMetrics, SystemMetrics
from src.llms import create_embedding_client
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowType(str, Enum):
    """工作流类型"""
    NAIVE = "naive"
    HYBRID = "hybrid"
    GRAPH = "graph"
    FULL = "full"


@dataclass
class EvaluationSample:
    """评估样本
    
    Attributes:
        query: 用户查询
        ground_truth: 标准答案
        relevant_docs: 相关文档 ID 列表
        metadata: 额外元数据
    """
    query: str
    ground_truth: str
    relevant_docs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """评估报告
    
    Attributes:
        total_samples: 总样本数
        successful_samples: 成功样本数
        failed_samples: 失败样本数
        avg_latency: 平均延迟
        avg_retrieval_metrics: 平均检索指标
        avg_generation_metrics: 平均生成指标
        avg_system_metrics: 平均系统指标
        workflow_type: 工作流类型
        timestamp: 时间戳
    """
    total_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    avg_latency: float = 0.0
    avg_retrieval_metrics: Dict[str, Any] = field(default_factory=dict)
    avg_generation_metrics: Dict[str, Any] = field(default_factory=dict)
    avg_system_metrics: Dict[str, Any] = field(default_factory=dict)
    workflow_type: str = "unknown"
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_samples": self.total_samples,
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
            "avg_latency": self.avg_latency,
            "avg_retrieval_metrics": self.avg_retrieval_metrics,
            "avg_generation_metrics": self.avg_generation_metrics,
            "avg_system_metrics": self.avg_system_metrics,
            "workflow_type": self.workflow_type,
            "timestamp": self.timestamp,
        }


class RAGEvaluator:
    """RAG 评估器
    
    用于评估 RAG 系统的性能。
    """
    
    def __init__(
        self,
        workflow: Any,
        workflow_type: WorkflowType,
        embedding_client: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        max_workers: int = None,
        k_values: List[int] = None,
    ):
        """初始化评估器
        
        Args:
            workflow: RAG 工作流实例
            workflow_type: 工作流类型
            embedding_client: Embedding 客户端（用于语义相似度计算）
            llm_client: LLM 客户端（用于 LLM-based 评估）
            checkpoint_path: 检查点路径
            max_workers: 最大并发数
            k_values: 评估的 K 值列表
        """
        config = get_config()
        
        self.workflow = workflow
        self.workflow_type = workflow_type
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.checkpoint_path = checkpoint_path
        self.max_workers = max_workers or 1
        self.k_values = k_values or [1, 3, 5, 10]
        
        self.retrieval_metrics = RetrievalMetrics(k_values=self.k_values)
        self.generation_metrics = GenerationMetrics()
        self.system_metrics = SystemMetrics()
        
        logger.info(
            f"初始化 RAGEvaluator: workflow_type={workflow_type.value}, "
            f"k_values={self.k_values}"
        )
    
    def evaluate(
        self,
        samples: List[EvaluationSample],
        resume: bool = False,
        save_checkpoint_every: int = 10,
    ) -> EvaluationReport:
        """评估样本
        
        Args:
            samples: 评估样本列表
            resume: 是否从检查点恢复
            save_checkpoint_every: 每多少个样本保存检查点
            
        Returns:
            EvaluationReport 实例
        """
        logger.info(f"开始评估 {len(samples)} 个样本")
        
        all_retrieval_metrics = []
        all_generation_metrics = []
        total_latency = 0.0
        success_count = 0
        failed_count = 0
        
        self.system_metrics.reset()
        
        start_idx = 0
        if resume and self.checkpoint_path:
            start_idx = self._load_checkpoint()
            logger.info(f"从检查点恢复: start_idx={start_idx}")
        
        for i, sample in enumerate(tqdm(samples[start_idx:], desc="评估中"), start=start_idx):
            try:
                chain_result = self.workflow.run(sample.query)
                
                if chain_result.success:
                    success_count += 1
                    total_latency += chain_result.latency
                    
                    retrieved_ids = [s.get("doc_id") for s in chain_result.sources if s.get("doc_id")]
                    
                    if sample.relevant_docs:
                        ret_metrics = self.retrieval_metrics.compute(
                            retrieved_ids, sample.relevant_docs
                        )
                        all_retrieval_metrics.append(ret_metrics.to_dict())
                    
                    gen_metrics = self.generation_metrics.compute(
                        chain_result.answer,
                        sample.ground_truth,
                        compute_semantic=self.embedding_client is not None,
                        embedding_client=self.embedding_client,
                    )
                    all_generation_metrics.append(gen_metrics.to_dict())
                    
                    context = getattr(chain_result, 'context', '') or ''
                    context_length = self.system_metrics.calculate_context_length(context)
                    token_cost = self.system_metrics.calculate_token_count(
                        context + chain_result.answer
                    )
                    evidence_count = self.system_metrics.calculate_evidence_count(
                        chain_result.sources
                    )
                    parent_recall_count = getattr(chain_result, 'parent_recall_count', 0) or 0
                    graph_expansion_count = getattr(chain_result, 'graph_expansion_count', 0) or 0
                    
                    self.system_metrics.add_sample(
                        context_length=context_length,
                        token_cost=token_cost,
                        evidence_count=evidence_count,
                        parent_recall_count=parent_recall_count,
                        graph_expansion_count=graph_expansion_count,
                    )
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"样本 {i} 评估失败: {e}")
                failed_count += 1
            
            if (i + 1) % save_checkpoint_every == 0 and self.checkpoint_path:
                self._save_checkpoint(i + 1)
        
        avg_latency = total_latency / success_count if success_count > 0 else 0
        
        avg_retrieval = self._aggregate_metrics(all_retrieval_metrics)
        avg_generation = self._aggregate_metrics(all_generation_metrics)
        avg_system = self.system_metrics.compute().to_dict()
        
        report = EvaluationReport(
            total_samples=len(samples),
            successful_samples=success_count,
            failed_samples=failed_count,
            avg_latency=avg_latency,
            avg_retrieval_metrics=avg_retrieval,
            avg_generation_metrics=avg_generation,
            avg_system_metrics=avg_system,
            workflow_type=self.workflow_type.value,
        )
        
        logger.info(
            f"评估完成: total={len(samples)}, success={success_count}, "
            f"failed={failed_count}, avg_latency={avg_latency:.2f}s"
        )
        
        return report
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合指标"""
        if not metrics_list:
            return {}
        
        aggregated = {}
        keys = set()
        for m in metrics_list:
            keys.update(m.keys())
        
        for key in keys:
            values = [m.get(key, 0) for m in metrics_list if isinstance(m.get(key), (int, float))]
            if values:
                aggregated[key] = sum(values) / len(values)
        
        return aggregated
    
    def _save_checkpoint(self, idx: int) -> None:
        """保存检查点"""
        if self.checkpoint_path:
            Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_path, 'w') as f:
                json.dump({"last_idx": idx}, f)
    
    def _load_checkpoint(self) -> int:
        """加载检查点"""
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)
                return data.get("last_idx", 0)
        return 0
    
    def save_report(self, report: EvaluationReport, output_path: str) -> None:
        """保存报告
        
        Args:
            report: 评估报告
            output_path: 输出路径
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(f"{output_path}_report.json", 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"报告已保存到: {output_path}_report.json")


def create_evaluator(
    workflow: Any,
    workflow_type: WorkflowType,
    embedding_client: Optional[Any] = None,
    llm_client: Optional[Any] = None,
    checkpoint_path: Optional[str] = None,
    **kwargs,
) -> RAGEvaluator:
    """创建 RAG 评估器
    
    Args:
        workflow: RAG 工作流实例
        workflow_type: 工作流类型
        embedding_client: Embedding 客户端
        llm_client: LLM 客户端
        checkpoint_path: 检查点路径
        **kwargs: 其他参数
        
    Returns:
        RAGEvaluator 实例
    """
    config = get_config()
    
    if embedding_client is None:
        embedding_client = create_embedding_client()
    
    if checkpoint_path is None:
        checkpoint_path = str(Path(config.faiss_index_path).parent / "checkpoints" / "eval.json")
    
    return RAGEvaluator(
        workflow=workflow,
        workflow_type=workflow_type,
        embedding_client=embedding_client,
        llm_client=llm_client,
        checkpoint_path=checkpoint_path,
        **kwargs,
    )


if __name__ == "__main__":
    print("=" * 50)
    print("测试 RAG 评估器")
    print("=" * 50)
    
    config = get_config()
    print(f"✓ 配置加载成功: llm_model={config.llm_model}")
    
    sample = EvaluationSample(
        query="什么是人工智能？",
        ground_truth="人工智能是计算机科学的一个分支。",
        relevant_docs=["doc1", "doc2"],
    )
    print(f"✓ EvaluationSample 创建: query={sample.query[:20]}...")
    
    report = EvaluationReport(
        total_samples=10,
        successful_samples=8,
        failed_samples=2,
        avg_latency=1.5,
    )
    print(f"✓ EvaluationReport 创建: total={report.total_samples}")
    
    print("\n所有测试通过!")
