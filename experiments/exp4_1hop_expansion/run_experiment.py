# -*- coding: utf-8 -*-
"""实验四：固定1-hop扩展

使用 Neo4j 中的结构图谱 + 语义图谱进行固定1-hop扩展检索实验。
评估指标：Recall, Precision, MRR, NDCG, MAP, Recall 提升率, 图谱覆盖率
支持断点续跑
"""

import json
import os
import sys
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
from src.llms.base_client import Message
from src.storage.graph_store.neo4j_store import Neo4jGraphStore
from src.storage.graph_store.local_graph import LocalGraphStore
from src.storage.graph_store.base_graph import Node, NodeType, Edge, EdgeType
from src.storage.vector_store.faiss_store import FAISSVectorStore
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OneHopExpansionExperiment:
    """固定1-hop扩展实验类
    
    使用 Neo4j 图谱进行实体链接和1-hop邻居扩展检索。
    """
    
    CHECKPOINT_FILE = "checkpoint.json"
    
    def __init__(
        self,
        test_data_path: str,
        output_dir: str,
        documents_path: Optional[str] = None,
        local_graph_path: Optional[str] = None,
        use_neo4j: bool = True,
        top_k_values: Optional[List[int]] = None,
        checkpoint_interval: int = 10,
    ):
        """初始化实验
        
        Args:
            test_data_path: 测试数据路径
            output_dir: 输出目录
            documents_path: 文档数据路径（用于构建标题索引）
            local_graph_path: 本地图谱路径（备用）
            use_neo4j: 是否使用 Neo4j
            top_k_values: 评估的 K 值列表
            checkpoint_interval: 检查点保存间隔
        """
        self.test_data_path = test_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.documents_path = documents_path
        self.local_graph_path = local_graph_path
        self.use_neo4j = use_neo4j
        
        self.top_k_values = top_k_values or [1, 3, 5, 10, 20, 50]
        self.metrics_calculator = RetrievalMetrics(k_values=self.top_k_values)
        self.checkpoint_interval = checkpoint_interval
        
        self.config = get_config()
        self.graph_store: Optional[Any] = None
        self.embedding_client: Optional[EmbeddingClient] = None
        self.llm_client: Optional[DeepSeekClient] = None
        self.generation_metrics = GenerationMetrics()
        self.test_data: Optional[pd.DataFrame] = None
        self.documents: List[Dict] = []
        self.title_to_doc_ids: Dict[str, Set[str]] = {}
        self.title_to_content: Dict[str, str] = {}
        self.entity_to_titles: Dict[str, Set[str]] = {}
        self.all_titles: Set[str] = set()
        self.total_graph_nodes: int = 0
        self.total_graph_edges: int = 0
        
    def load_graph_store(self) -> None:
        """加载图存储"""
        neo4j_connected = False
        
        if self.use_neo4j:
            try:
                logger.info("尝试连接 Neo4j...")
                self.graph_store = Neo4jGraphStore()
                self.total_graph_nodes = self.graph_store.count_nodes()
                self.total_graph_edges = self.graph_store.count_edges()
                
                if self.total_graph_nodes > 0:
                    logger.info(f"Neo4j 连接成功: nodes={self.total_graph_nodes}, edges={self.total_graph_edges}")
                    neo4j_connected = True
                    return
                else:
                    logger.warning("Neo4j 图谱为空，切换到本地图谱...")
                    self.graph_store.close()
            except Exception as e:
                logger.warning(f"Neo4j 连接失败: {e}")
                logger.info("切换到本地图谱...")
        
        if not neo4j_connected and self.local_graph_path:
            logger.info(f"加载本地图谱: {self.local_graph_path}")
            self.graph_store = LocalGraphStore()
            self._load_graph_from_json(self.local_graph_path)
            
            self.total_graph_nodes = self.graph_store.count_nodes()
            self.total_graph_edges = self.graph_store.count_edges()
            logger.info(f"本地图谱加载成功: nodes={self.total_graph_nodes}, edges={self.total_graph_edges}")
        elif not neo4j_connected:
            raise ValueError("需要提供 local_graph_path 或确保 Neo4j 中有数据")
    
    def _load_graph_from_json(self, json_path: str) -> None:
        """从 JSON 文件加载图谱
        
        支持三元组格式：[{"Subject": "...", "Predicate": "...", "Object": "..."}, ...]
        """
        json_file = Path(json_path)
        if json_file.is_dir():
            json_file = json_file / "graph.json"
        
        if not json_file.exists():
            json_file = Path(json_path).parent / "local_graph.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"图谱文件不存在: {json_path}")
        
        logger.info(f"从 JSON 加载图谱: {json_file}")
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            triples = data
            node_map: Dict[str, Node] = {}
            
            for triple in tqdm(triples, desc="构建图谱"):
                subject = triple.get("Subject", "").strip('"')
                predicate = triple.get("Predicate", "")
                obj = triple.get("Object", "").strip('"')
                
                if not subject or not obj:
                    continue
                
                if subject not in node_map:
                    node_map[subject] = Node(
                        id=f"entity_{len(node_map)}",
                        type=NodeType.ENTITY,
                        name=subject,
                        properties={"original_name": subject},
                    )
                    self.graph_store.add_node(node_map[subject])
                
                if obj not in node_map:
                    node_map[obj] = Node(
                        id=f"entity_{len(node_map)}",
                        type=NodeType.ENTITY,
                        name=obj,
                        properties={"original_name": obj},
                    )
                    self.graph_store.add_node(node_map[obj])
                
                try:
                    edge_type = EdgeType(predicate.lower().replace(" ", "_"))
                except ValueError:
                    edge_type = EdgeType.OTHER
                
                edge = Edge(
                    id=f"edge_{len(self.graph_store._edges)}",
                    source_id=node_map[subject].id,
                    target_id=node_map[obj].id,
                    type=edge_type,
                    properties={"predicate": predicate},
                    weight=1.0,
                )
                self.graph_store.add_edge(edge)
            
            logger.info(f"图谱构建完成: nodes={self.graph_store.count_nodes()}, edges={self.graph_store.count_edges()}")
        
        elif isinstance(data, dict):
            if "nodes" in data and "edges" in data:
                self.graph_store.load_from_real_kg(data)
            else:
                raise ValueError(f"不支持的 JSON 格式")
    
    def load_documents(self) -> None:
        """加载文档数据"""
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
                self.title_to_doc_ids[title] = {title}
                self.title_to_content[title] = doc.get("sentence_total", doc.get("content", ""))
        
        logger.info(f"标题索引构建完成: {len(self.all_titles)} 个唯一标题")
    
    def load_embedding_client(self) -> None:
        """加载嵌入客户端"""
        logger.info("加载嵌入客户端...")
        self.embedding_client = EmbeddingClient()
        logger.info(f"嵌入客户端加载完成: dimension={self.embedding_client.dimension}")
    
    def load_llm_client(self) -> None:
        """加载 LLM 客户端"""
        logger.info("加载 LLM 客户端...")
        self.llm_client = DeepSeekClient()
        logger.info("LLM 客户端加载完成")
    
    def load_test_data(self) -> None:
        """加载测试数据"""
        logger.info(f"加载测试数据: {self.test_data_path}")
        
        self.test_data = pd.read_parquet(self.test_data_path)
        logger.info(f"测试数据加载完成: {len(self.test_data)} 条记录")
    
    def entity_linking(self, query: str, titles: List[str]) -> List[str]:
        """实体链接：将查询中的实体链接到图谱节点
        
        Args:
            query: 查询文本
            titles: 相关标题列表
            
        Returns:
            链接到的实体节点ID列表
        """
        linked_entities = []
        
        for title in titles:
            # 使用 Cypher 查询 semantic 标签的节点
            cypher = """
            MATCH (n:semantic)
            WHERE toLower(n.name) CONTAINS toLower($title)
            RETURN elementId(n) as id, n.name as name
            LIMIT 10
            """
            result = self.graph_store.query(cypher, {"title": title})
            for node_data in result:
                if node_data.get("id"):
                    linked_entities.append(node_data["id"])
        
        return list(set(linked_entities))
    
    def one_hop_expansion(self, entity_ids: List[str]) -> Tuple[Set[str], Set[str]]:
        """执行1-hop扩展
        
        Args:
            entity_ids: 起始实体ID列表
            
        Returns:
            (扩展后的实体ID集合, 扩展后的边ID集合)
        """
        expanded_entities = set(entity_ids)
        expanded_edges = set()
        
        for entity_id in entity_ids:
            # 使用 Cypher 查询获取邻居节点
            cypher = """
            MATCH (n)-[r]-(neighbor)
            WHERE elementId(n) = $entity_id
            RETURN elementId(neighbor) as neighbor_id, elementId(r) as edge_id, type(r) as edge_type
            LIMIT 100
            """
            result = self.graph_store.query(cypher, {"entity_id": entity_id})
            for record in result:
                if record.get("neighbor_id"):
                    expanded_entities.add(record["neighbor_id"])
                if record.get("edge_id"):
                    expanded_edges.add(record["edge_id"])
        
        return expanded_entities, expanded_edges
    
    def get_entity_titles(self, entity_ids: Set[str]) -> Set[str]:
        """获取实体对应的标题
        
        Args:
            entity_ids: 实体ID集合
            
        Returns:
            标题集合
        """
        titles = set()
        
        for entity_id in entity_ids:
            # 使用 Cypher 查询获取节点名称
            cypher = """
            MATCH (n)
            WHERE elementId(n) = $entity_id
            RETURN n.name as name
            """
            result = self.graph_store.query(cypher, {"entity_id": entity_id})
            if result:
                node_name = result[0].get("name", "")
                if node_name and node_name in self.all_titles:
                    titles.add(node_name)
        
        return titles
    
    def get_relevant_titles(self, row: pd.Series) -> Set[str]:
        """获取相关文档标题"""
        supporting_facts = row.get("supporting_facts", {})
        titles = supporting_facts.get("title", [])
        
        if isinstance(titles, np.ndarray):
            titles = titles.tolist()
        
        return set(titles)
    
    def generate_answer(self, query: str, retrieved_titles: List[str]) -> str:
        """基于检索结果生成答案
        
        Args:
            query: 查询文本
            retrieved_titles: 检索到的标题列表
            
        Returns:
            生成的答案
        """
        if not self.llm_client:
            logger.warning("LLM 客户端未初始化，无法生成答案")
            return ""
        
        # 构建上下文
        contexts = []
        for title in retrieved_titles[:10]:  # 限制上下文长度
            content = self.title_to_content.get(title, "")
            if content:
                contexts.append(f"【{title}】\n{content}")
        
        if not contexts:
            return ""
        
        context_text = "\n\n".join(contexts)
        
        # 构建提示
        prompt = f"""请根据以下参考资料回答问题。如果参考资料中没有相关信息，请说明无法回答。

参考资料：
{context_text}

问题：{query}

请直接给出答案，不要解释："""
        
        try:
            messages = [Message(role="user", content=prompt)]
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
        context_titles: List[str],
        max_k: int,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """评估单个查询
        
        Args:
            query: 查询文本
            relevant_titles: 相关标题集合
            context_titles: 上下文中的标题列表
            max_k: 最大K值
            ground_truth: 标准答案（用于生成指标评估）
            
        Returns:
            评估结果字典
        """
        linked_entity_ids = self.entity_linking(query, context_titles)
        
        if not linked_entity_ids:
            for title in relevant_titles:
                nodes = self.graph_store.get_nodes_by_property("original_name", title)
                if nodes:
                    linked_entity_ids.extend([n.id for n in nodes])
            linked_entity_ids = list(set(linked_entity_ids))
        
        expanded_entities, expanded_edges = self.one_hop_expansion(linked_entity_ids)
        
        retrieved_titles = self.get_entity_titles(expanded_entities)
        
        retrieved_list = list(retrieved_titles)[:max_k]
        
        result = self.metrics_calculator.compute(retrieved_list, list(relevant_titles))
        
        title_recall = len(retrieved_titles & relevant_titles) / len(relevant_titles) if relevant_titles else 0
        title_precision = len(retrieved_titles & relevant_titles) / len(retrieved_titles) if retrieved_titles else 0
        
        graph_coverage = len(expanded_entities) / self.total_graph_nodes if self.total_graph_nodes > 0 else 0
        
        # 生成答案并计算生成指标
        generated_answer = ""
        gen_metrics = {"exact_match": 0.0, "f1_score": 0.0, "semantic_similarity": 0.0}
        
        if ground_truth and self.llm_client:
            generated_answer = self.generate_answer(query, retrieved_list)
            
            if generated_answer:
                gen_result = self.generation_metrics.compute(
                    predicted=generated_answer,
                    ground_truth=ground_truth,
                    compute_semantic=True,
                    embedding_client=self.embedding_client,
                )
                gen_metrics = {
                    "exact_match": gen_result.exact_match,
                    "f1_score": gen_result.f1_score,
                    "semantic_similarity": gen_result.semantic_similarity,
                }
        
        return {
            "linked_entities": linked_entity_ids[:10],
            "expanded_entity_count": len(expanded_entities),
            "expanded_edge_count": len(expanded_edges),
            "retrieved_titles": list(retrieved_titles),
            "relevant_titles": list(relevant_titles),
            "metrics": result.to_dict(),
            "title_recall": title_recall,
            "title_precision": title_precision,
            "graph_coverage": graph_coverage,
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "generation_metrics": gen_metrics,
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
        """运行实验"""
        logger.info("开始运行实验...")
        
        self.load_graph_store()
        self.load_documents()
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
            "graph_coverage": [],
            "expanded_entity_count": [],
            "expanded_edge_count": [],
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
                
                context = row.get("context", {})
                context_titles = context.get("title", [])
                if isinstance(context_titles, np.ndarray):
                    context_titles = context_titles.tolist()
                
                try:
                    result = self.evaluate_single_query(
                        query, relevant_titles, context_titles, max_k, ground_truth
                    )
                    
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
                    aggregated_metrics["graph_coverage"].append(result["graph_coverage"])
                    aggregated_metrics["expanded_entity_count"].append(result["expanded_entity_count"])
                    aggregated_metrics["expanded_edge_count"].append(result["expanded_edge_count"])
                    
                    # 收集生成指标
                    gen_metrics = result["generation_metrics"]
                    aggregated_metrics["exact_match"].append(gen_metrics["exact_match"])
                    aggregated_metrics["f1_score"].append(gen_metrics["f1_score"])
                    aggregated_metrics["semantic_similarity"].append(gen_metrics["semantic_similarity"])
                    
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
                            "graph_coverage": result["graph_coverage"],
                            "expanded_entity_count": result["expanded_entity_count"],
                            "expanded_edge_count": result["expanded_edge_count"],
                            "generated_answer": result["generated_answer"],
                            "generation_metrics": gen_metrics,
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
            "graph_coverage": np.mean(aggregated_metrics["graph_coverage"]),
            "avg_expanded_entities": np.mean(aggregated_metrics["expanded_entity_count"]),
            "avg_expanded_edges": np.mean(aggregated_metrics["expanded_edge_count"]),
            "exact_match": np.mean(aggregated_metrics["exact_match"]) if aggregated_metrics["exact_match"] else 0.0,
            "f1_score": np.mean(aggregated_metrics["f1_score"]) if aggregated_metrics["f1_score"] else 0.0,
            "semantic_similarity": np.nanmean(aggregated_metrics["semantic_similarity"]) if aggregated_metrics["semantic_similarity"] else 0.0,
        }
        
        baseline_recall = 0.1
        recall_improvement = {
            k: (v - baseline_recall) / baseline_recall * 100 if baseline_recall > 0 else 0
            for k, v in final_metrics["recall_at_k"].items()
        }
        final_metrics["recall_improvement"] = recall_improvement
        
        experiment_result = {
            "experiment_name": "one_hop_expansion",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "test_data_path": self.test_data_path,
                "documents_path": self.documents_path,
                "use_neo4j": self.use_neo4j,
                "top_k_values": self.top_k_values,
                "sample_size": sample_size or len(test_data),
                "total_test_samples": len(test_data),
                "graph_stats": {
                    "total_nodes": self.total_graph_nodes,
                    "total_edges": self.total_graph_edges,
                },
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
                "Metric": "图谱覆盖率",
                "Value": metrics["graph_coverage"],
            },
            {
                "Metric": "平均扩展实体数",
                "Value": metrics["avg_expanded_entities"],
            },
            {
                "Metric": "平均扩展边数",
                "Value": metrics["avg_expanded_edges"],
            },
            {
                "Metric": "Exact Match (EM)",
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
        ]
        
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = self.output_dir / f"metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"指标表格已保存: {metrics_path}")
        
        self._print_results(results)
    
    def _print_results(self, results: Dict[str, Any]) -> None:
        """打印实验结果"""
        print("\n" + "=" * 60)
        print("实验结果 - 固定1-hop扩展")
        print("=" * 60)
        
        config = results["config"]
        metrics = results["metrics"]
        
        print(f"\n【图谱统计】")
        print(f"  总节点数: {config['graph_stats']['total_nodes']}")
        print(f"  总边数: {config['graph_stats']['total_edges']}")
        
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
        
        print("\n【图谱指标】")
        print(f"  图谱覆盖率: {metrics['graph_coverage']:.4f}")
        print(f"  平均扩展实体数: {metrics['avg_expanded_entities']:.2f}")
        print(f"  平均扩展边数: {metrics['avg_expanded_edges']:.2f}")
        
        print("\n【生成指标】")
        print(f"  Exact Match (EM): {metrics['exact_match']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Semantic Similarity: {metrics['semantic_similarity']:.4f}")
        
        print("\n" + "=" * 60)


def main():
    """主函数"""
    test_data_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/validation-00000-of-00001.parquet"
    documents_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/valid_title_sentence.json"
    local_graph_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/local_graph.json"
    output_dir = "e:/Code_Personal/Subject/test02/experiments/exp4_1hop_expansion"
    
    experiment = OneHopExpansionExperiment(
        test_data_path=test_data_path,
        output_dir=output_dir,
        documents_path=documents_path,
        local_graph_path=local_graph_path,
        use_neo4j=True,
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
