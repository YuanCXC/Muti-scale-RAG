# -*- coding: utf-8 -*-
"""实验九：参数敏感性分析

目的：证明方法鲁棒性，不是调参凑出来的

方法：正交实验设计（L16正交表）
- k1（向量检索数）: 5, 10, 15, 20 (4水平)
- k2（关键词检索数）: 10, 20, 30 (3水平)
- k4（最终候选数）: 3, 5, 7, 10 (4水平)
- 父切片阈值: 0.2, 0.3, 0.4, 0.5 (4水平，默认值0.3)

正交设计组合数: 约16种（从192种全组合中精选代表性组合）
评估指标：Recall, Precision, MRR, NDCG, MAP, Exact Match, F1, Semantic Similarity
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
from src.storage.graph_store.neo4j_store import Neo4jGraphStore
from src.storage.graph_store.local_graph import LocalGraphStore
from src.storage.graph_store.base_graph import Node, NodeType, Edge, EdgeType
from src.storage.vector_store.faiss_store import FAISSVectorStore
from src.storage.vector_store.base_store import VectorMetadata
from src.retrievers.reranker import Reranker, create_reranker
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ParameterSensitivityExperiment:
    """参数敏感性实验类
    
    排列组合测试所有参数组合
    """
    
    CHECKPOINT_FILE = "checkpoint.json"
    
    DEFAULT_K1_VALUES = [5, 10, 15, 20]
    DEFAULT_K2_VALUES = [10, 20, 30]
    DEFAULT_K4_VALUES = [3, 5, 7, 10]
    DEFAULT_THRESHOLD_VALUES = [0.2, 0.3, 0.4, 0.5]
    
    def __init__(
        self,
        test_data_path: str,
        output_dir: str,
        documents_path: Optional[str] = None,
        vector_store_paths: Optional[Dict[str, str]] = None,
        local_graph_path: Optional[str] = None,
        use_neo4j: bool = True,
        sample_size: Optional[int] = None,
        k1_values: Optional[List[int]] = None,
        k2_values: Optional[List[int]] = None,
        k4_values: Optional[List[int]] = None,
        threshold_values: Optional[List[float]] = None,
        checkpoint_interval: int = 5,
    ):
        """初始化实验"""
        self.test_data_path = test_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.documents_path = documents_path
        self.vector_store_paths = vector_store_paths or {
            "paragraph": str(Path(test_data_path).parent / "vector_stores" / "valid_title_sentence"),
            "sentence": str(Path(test_data_path).parent / "vector_stores" / "single_sentence"),
        }
        self.local_graph_path = local_graph_path
        self.use_neo4j = use_neo4j
        
        self.sample_size = sample_size
        self.k1_values = k1_values or self.DEFAULT_K1_VALUES
        self.k2_values = k2_values or self.DEFAULT_K2_VALUES
        self.k4_values = k4_values or self.DEFAULT_K4_VALUES
        self.threshold_values = threshold_values or self.DEFAULT_THRESHOLD_VALUES
        self.checkpoint_interval = checkpoint_interval
        
        self.current_k1 = 10
        self.current_k2 = 20
        self.current_k4 = 7
        self.metrics_calculator = RetrievalMetrics(k_values=[self.current_k4])
        self.generation_metrics = GenerationMetrics()
        
        self.config = get_config()
        
        self.graph_store: Optional[Any] = None
        self.vector_stores: Dict[str, Optional[FAISSVectorStore]] = {}
        self.embedding_client: Optional[EmbeddingClient] = None
        self.llm_client: Optional[DeepSeekClient] = None
        self.reranker: Optional[Any] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.sampled_data: Optional[pd.DataFrame] = None
        self.documents: List[Dict] = []
        self.title_to_doc_ids: Dict[str, Set[str]] = {}
        self.title_to_content: Dict[str, str] = {}
        self.all_titles: Set[str] = set()
        self.total_graph_nodes: int = 0
        self.total_graph_edges: int = 0
        
        self.current_threshold = 0.8
        self.theta1 = 10
        
    def _save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """保存检查点"""
        checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            logger.info(f"检查点已保存")
        except Exception as e:
            logger.warning(f"保存检查点失败: {e}")
    
    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            logger.info(f"从检查点恢复: 已完成 {checkpoint.get('completed_combinations', 0)} 组合")
            return checkpoint
        except Exception as e:
            logger.warning(f"加载检查点失败: {e}")
            return None
    
    def _clear_checkpoint(self) -> None:
        """清除检查点"""
        checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.info("检查点已清除")
            except Exception as e:
                logger.warning(f"清除检查点失败: {e}")
        
    def load_all_resources(self) -> None:
        """加载所有资源"""
        self.load_graph_store()
        self.load_documents()
        self.load_vector_stores()
        self.load_embedding_client()
        self.load_llm_client()
        self.load_reranker()
        self.load_test_data()
        
        if self.sample_size:
            self.sampled_data = self.test_data.sample(n=self.sample_size, random_state=42)
            logger.info(f"采样 {self.sample_size} 条数据进行测试")
        else:
            self.sampled_data = self.test_data
            
    def load_graph_store(self) -> None:
        """加载图存储"""
        neo4j_connected = False
        
        if self.use_neo4j:
            try:
                logger.info("尝试连接 Neo4j...")
                self.graph_store = Neo4jGraphStore()
                
                results = self.graph_store.query('MATCH (n) RETURN DISTINCT labels(n) as labels, count(n) as count')
                total_nodes = sum(r['count'] for r in results)
                
                rel_results = self.graph_store.query('MATCH ()-[r]->() RETURN DISTINCT type(r) as type, count(r) as count')
                total_edges = sum(r['count'] for r in rel_results)
                
                self.total_graph_nodes = total_nodes
                self.total_graph_edges = total_edges
                
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
        """从 JSON 文件加载图谱"""
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
    
    def load_vector_stores(self) -> None:
        """加载向量存储"""
        for store_type, store_path in self.vector_store_paths.items():
            logger.info(f"加载 {store_type} 向量存储: {store_path}")
            try:
                store = FAISSVectorStore()
                store.load(store_path)
                self.vector_stores[store_type] = store
                logger.info(f"{store_type} 向量存储加载成功: vectors={store.count()}")
            except FileNotFoundError:
                logger.warning(f"{store_type} 向量存储文件不存在: {store_path}")
                self.vector_stores[store_type] = None
    
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
                if title not in self.title_to_doc_ids:
                    self.title_to_doc_ids[title] = set()
                self.title_to_doc_ids[title].add(doc.get("id", title))
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
        logger.info(f"LLM 客户端加载完成: model={self.llm_client.model}")
    
    def load_reranker(self) -> None:
        """加载重排序器"""
        logger.info("加载重排序器...")
        try:
            self.reranker = create_reranker(mode="api")
            logger.info("重排序器加载完成 (API模式)")
        except Exception as e:
            logger.warning(f"重排序器加载失败: {e}")
            self.reranker = None
    
    def load_test_data(self) -> None:
        """加载测试数据"""
        logger.info(f"加载测试数据: {self.test_data_path}")
        self.test_data = pd.read_parquet(self.test_data_path)
        logger.info(f"测试数据加载完成: {len(self.test_data)} 条记录")
    
    def step1_vector_retrieval(self, query: str) -> List[Dict]:
        """Step 1: 向量检索 - 完美复现原项目
        
        原项目逻辑：
        1. 使用句子级向量库（single_sentence_store）
        2. 检索 k1 个结果
        3. 返回 title 和 sentence
        
        Args:
            query: 查询文本
            
        Returns:
            检索结果列表
        """
        all_results = []
        
        sentence_store = self.vector_stores.get("sentence")
        if sentence_store:
            query_vector = self.embedding_client.embed(query)
            results = sentence_store.search(query_vector, top_k=self.current_k1)
            
            for result in results:
                title = result.metadata.extra.get("title", result.metadata.doc_id)
                content = result.metadata.content
                
                sentence_id = result.metadata.extra.get("sentence_id", "")
                
                all_results.append({
                    "id": result.id,
                    "title": title,
                    "content": content,
                    "score": result.score,
                    "source": "vector_sentence",
                    "sentence_id": sentence_id,
                    "is_sentence_level": True,
                })
        else:
            logger.warning("句子级向量库未加载，尝试使用其他向量库")
            for store_type, store in self.vector_stores.items():
                if not store:
                    continue
                
                query_vector = self.embedding_client.embed(query)
                results = store.search(query_vector, top_k=self.current_k1)
                
                for result in results:
                    title = result.metadata.extra.get("title", result.metadata.doc_id)
                    content = result.metadata.content
                    
                    sentence_id = result.metadata.extra.get("sentence_id", "")
                    is_sentence_level = bool(sentence_id)
                    
                    all_results.append({
                        "id": result.id,
                        "title": title,
                        "content": content,
                        "score": result.score,
                        "source": f"vector_{store_type}",
                        "sentence_id": sentence_id,
                        "is_sentence_level": is_sentence_level,
                    })
        
        return all_results
    
    def step2_extract_keywords(self, query: str) -> List[str]:
        """Step 2: 关键词提取"""
        prompt = f"""请从以下问题中提取关键实体和关键词，用于知识图谱检索。
只返回关键词列表，用逗号分隔，不要其他解释。

问题: {query}

关键词:"""
        
        try:
            response = self.llm_client.generate([Message(role="user", content=prompt)])
            if response.success:
                keywords = [k.strip() for k in response.content.split(",") if k.strip()]
                return keywords[:10]
        except Exception as e:
            logger.warning(f"关键词提取失败: {e}")
        
        return []
    
    def step3_keyword_retrieval(self, keywords: List[str]) -> List[Dict]:
        """Step 3: 关键词检索 (图谱) - 完美复现原项目
        
        原项目逻辑：
        1. 先精确匹配 Section 节点的 title 属性
        2. 如果结果不足 k2，再模糊匹配（CONTAINS）
        3. 返回 Section 节点的 title 和 sentence_total
        
        Args:
            keywords: 关键词列表
            
        Returns:
            检索结果列表
        """
        if not keywords:
            return []
        
        results = []
        seen_titles = set()
        
        def cypher_str_value(val):
            if not isinstance(val, str):
                return '""'
            out = val.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{out}"'
        
        strict_conditions = []
        fuzzy_conditions = []
        
        for k in keywords:
            val = cypher_str_value(k)
            strict_conditions.append(f'n.title = {val}')
            fuzzy_conditions.append(f'n.title CONTAINS {val}')
        
        strict_matches = " OR ".join(strict_conditions)
        fuzzy_matches = " OR ".join(fuzzy_conditions)
        
        cypher_strict = f"MATCH (n:Section) WHERE {strict_matches} RETURN n LIMIT {self.current_k2}"
        cypher_fuzzy = f"MATCH (n:Section) WHERE {fuzzy_matches} RETURN n LIMIT {self.current_k2}"
        
        try:
            strict_results = self.graph_store.query(cypher_strict)
            
            for item in strict_results:
                node = item.get("n", item)
                title = node.get("title", item.get("title", "")).strip('"') if isinstance(node, dict) else str(node)
                
                if title and title not in seen_titles:
                    content = node.get("sentence_total", self.title_to_content.get(title, ""))
                    results.append({
                        "id": title,
                        "title": title,
                        "content": content,
                        "score": 1.0,
                        "source": "graph",
                    })
                    seen_titles.add(title)
            
            if len(results) < 10:
                fuzzy_results = self.graph_store.query(cypher_fuzzy)
                
                for item in fuzzy_results:
                    node = item.get("n", item)
                    title = node.get("title", item.get("title", "")).strip('"') if isinstance(node, dict) else str(node)
                    
                    if title and title not in seen_titles:
                        content = node.get("sentence_total", self.title_to_content.get(title, ""))
                        results.append({
                            "id": title,
                            "title": title,
                            "content": content,
                            "score": 1.0,
                            "source": "graph",
                        })
                        seen_titles.add(title)
                    
                    if len(results) >= self.current_k2:
                        break
        except Exception as e:
            logger.warning(f"关键词检索失败: {e}")
        
        return results[:self.current_k2]
    
    def step4_graph_expansion(self, titles: List[str]) -> List[Dict]:
        """Step 4: 知识图谱扩展 (语义关联) - 完美复现原项目
        
        原项目逻辑：
        1. 从 Section 节点出发（通过 title 精确匹配）
        2. 通过 SEMANTIC_LINKS 关系找到关联的 semantic 节点
        3. 从这个 semantic 节点出发，通过 0-3 跳的关系路径扩展
        4. 过滤条件：排除 SEPARATES 关系，节点度数 ≤ 10
        5. 找到路径终点的 semantic 节点
        6. 再通过 SEMANTIC_LINKS 关系找到关联的 Section 节点
        
        Args:
            titles: 标题列表
            
        Returns:
            扩展后的结果列表
        """
        results = []
        seen_titles = set(titles)
        
        for title in titles:
            safe_title = title.replace('"', '\\"')
            
            cypher_query = (
                'MATCH (start:Section {title: "' + safe_title + '"})\n'
                'MATCH (start)-[:SEMANTIC_LINKS]-(first)\n'
                'MATCH p = (first)-[r*0..3]-(last)\n'
                'MATCH (last)-[:SEMANTIC_LINKS]-(n:Section)\n'
                'WHERE n <> start\n'
                "  AND ALL(rel IN r WHERE type(rel) <> 'SEPARATES')\n"
                "  AND ALL(x IN nodes(p) WHERE COUNT { (x)--() } <= 10)\n"
                "RETURN DISTINCT \n"
                "    n.title AS section_title,\n"
                "    n.sentence_total AS sentence_total\n"
            )
            
            try:
                expanded = self.graph_store.query(cypher_query)
                
                for item in expanded:
                    section_title = item.get("section_title", "").strip('"')
                    sentence_total = item.get("sentence_total", "")
                    
                    if not section_title:
                        continue
                    
                    if section_title not in seen_titles:
                        content = sentence_total if sentence_total else self.title_to_content.get(section_title, section_title)
                        results.append({
                            "id": section_title,
                            "title": section_title,
                            "content": content,
                            "score": 1.0,
                            "source": "expansion",
                        })
                        seen_titles.add(section_title)
            except Exception as e:
                logger.warning(f"图谱扩展查询失败 (title={title}): {e}")
                continue
        
        return results
    
    def step5_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Step 5: 重排序"""
        if not self.reranker or not candidates:
            return candidates[:self.current_k4]
        
        from src.retrievers.base_retriever import SearchResult
        
        search_results = [
            SearchResult(
                doc_id=c["id"],
                content=c.get("content", c.get("title", "")),
                score=c.get("score", 0.0),
                metadata=c,
            )
            for c in candidates
        ]
        
        reranked = self.reranker.rerank(query, search_results, top_k=self.current_k4)
        
        return [
            {
                "id": r.doc_id,
                "title": r.metadata.get("title", r.doc_id),
                "content": r.content,
                "score": r.score,
                "source": r.metadata.get("source", "reranked"),
            }
            for r in reranked
        ]
    
    def step6_parent_chunk_mapping(self, candidates: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Step 6: 父切片映射"""
        merged_dict = {}
        
        for item in candidates:
            title = item.get("title", "")
            content = item.get("content", "")
            is_sentence_level = item.get("is_sentence_level", False)
            
            if not title:
                continue
            
            if title not in merged_dict:
                if is_sentence_level and title in self.title_to_content:
                    full_content = self.title_to_content[title]
                else:
                    full_content = content
                
                merged_dict[title] = {
                    "title": title,
                    "content": [full_content],
                    "source": item.get("source", "merged"),
                    "is_sentence_level": is_sentence_level,
                }
            else:
                if is_sentence_level and title in self.title_to_content:
                    full_content = self.title_to_content[title]
                    if full_content not in merged_dict[title]["content"]:
                        merged_dict[title]["content"].append(full_content)
                else:
                    if content not in merged_dict[title]["content"]:
                        merged_dict[title]["content"].append(content)
        
        all_titles = []
        final_results = []
        for v in merged_dict.values():
            unique_content = list(dict.fromkeys(v["content"]))
            v["content"] = "\n".join([c for c in unique_content if c])
            final_results.append(v)
            all_titles.append(v["title"])
        
        return all_titles, final_results
    
    def step7_llm_scoring(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Step 7: LLM 证据强度打分 - 完美复现原项目
        
        使用原项目的 PARENT_CHUNK_MAPPING_SCORING_PROMPT：
        - 对每个上下文块进行证据强度评估
        - 返回 JSON 格式的分数列表
        - 支持解析失败时的正则回退
        
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
- 0.0–0.3: Little or no valid evidence
- 0.4–0.5: Contains some relevant information, but the evidence is weak, limited, or mainly local/background content
- 0.6–0.7: Provides relatively clear and useful evidence and offers strong support for the query, but it is still not sufficiently key or sufficient
- 0.8–1.0: Provides direct, key, and high-value evidential support; even if the complete answer may still require additional context, this chunk itself already has high support value

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
    
    def step8_update_low_score_docs(self, query: str, candidates: List[Dict], threshold: float = 0.8) -> List[Dict]:
        """Step 8: 低分文档更新 - 完美复现原项目
        
        原项目逻辑：
        1. 对证据强度低于阈值（默认0.8）的文档
        2. 从 Neo4j 的 Section 节点查询完整的 sentence_total
        3. 覆盖写入候选文档的 content 字段
        
        Args:
            query: 查询文本
            candidates: 候选文档列表
            threshold: 低分阈值（原项目默认0.8）
            
        Returns:
            更新后的结果列表
        """
        low_score_titles = []
        for candidate in candidates:
            evidence_score = candidate.get("evidence_score", 0)
            if evidence_score < threshold:
                title = candidate.get("title", "")
                if title:
                    low_score_titles.append(title)
        
        if not low_score_titles:
            for candidate in candidates:
                candidate["updated"] = False
                candidate["update_reason"] = "score_above_threshold"
            return candidates
        
        def cypher_str_value(val):
            if not isinstance(val, str):
                return ""
            out = val.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{out}"'
        
        titles_cypher = [cypher_str_value(title) for title in low_score_titles]
        titles_str = ", ".join(titles_cypher)
        cypher = (
            f"MATCH (n:Section) WHERE n.title IN [{titles_str}] "
            f"RETURN n.title AS title, n.sentence_total AS sentence_total"
        )
        
        try:
            results = self.graph_store.query(cypher)
            title_to_sentence = {}
            for item in results:
                t = item.get("title", "")
                s = item.get("sentence_total", "")
                if t:
                    title_to_sentence[t] = s
            
            for candidate in candidates:
                evidence_score = candidate.get("evidence_score", 0)
                if evidence_score < threshold:
                    title = candidate.get("title", "")
                    if title in title_to_sentence:
                        full_content = title_to_sentence[title]
                        candidate["content"] = full_content
                        candidate["updated"] = True
                        candidate["update_reason"] = "low_score_enhanced_from_graph"
                    elif title in self.title_to_content:
                        full_content = self.title_to_content[title]
                        candidate["content"] = full_content
                        candidate["updated"] = True
                        candidate["update_reason"] = "low_score_enhanced_from_local"
                    else:
                        candidate["updated"] = False
                        candidate["update_reason"] = "no_full_content_available"
                else:
                    candidate["updated"] = False
                    candidate["update_reason"] = "score_above_threshold"
        except Exception as e:
            logger.warning(f"低分文档更新失败: {e}")
            for candidate in candidates:
                candidate["updated"] = False
                candidate["update_reason"] = "query_failed"
        
        return candidates
    
    def step9_generate_answer(self, query: str, candidates: List[Dict]) -> str:
        """Step 9: LLM 生成最终答案"""
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
    
    def _add_to_dict_with_dedup(
        self,
        items: List[Dict],
        target_dict: List[Dict],
        dedup_type: str = "vector"
    ) -> None:
        """智能去重添加"""
        for item in items:
            title = item.get("title", "")
            content = item.get("content", "")
            
            if not title or not content:
                continue
            
            duplicate = False
            for entry in target_dict:
                if dedup_type == "vector":
                    if entry.get("title") == title and entry.get("content") == content:
                        duplicate = True
                        break
                elif dedup_type == "keywords":
                    if entry.get("title") == title:
                        duplicate = True
                        break
            
            if not duplicate:
                target_dict.append({
                    "id": item.get("id", title),
                    "title": title,
                    "content": content,
                    "source": item.get("source", "unknown"),
                    "sentence_id": item.get("sentence_id", ""),
                    "is_sentence_level": item.get("is_sentence_level", False),
                })
    
    def run_adaptive_retrieval(self, query: str) -> Tuple[List[str], Dict]:
        """运行完整的自适应检索流程"""
        stats = {
            "vector_results": 0,
            "keywords": [],
            "graph_results": 0,
            "expanded_results": 0,
            "reranked_results": 0,
            "final_results": 0,
            "latency": {},
        }
        
        title_sentence_dict = []
        
        start_time = time.time()
        vector_results = self.step1_vector_retrieval(query)
        stats["latency"]["vector"] = time.time() - start_time
        stats["vector_results"] = len(vector_results)
        
        self._add_to_dict_with_dedup(vector_results, title_sentence_dict, dedup_type="vector")
        
        start_time = time.time()
        keywords = self.step2_extract_keywords(query)
        stats["latency"]["keywords"] = time.time() - start_time
        stats["keywords"] = keywords
        
        start_time = time.time()
        graph_results = self.step3_keyword_retrieval(keywords)
        stats["latency"]["graph"] = time.time() - start_time
        stats["graph_results"] = len(graph_results)
        
        self._add_to_dict_with_dedup(graph_results, title_sentence_dict, dedup_type="keywords")
        
        title_list = list(set([item["title"] for item in title_sentence_dict if "title" in item]))
        
        start_time = time.time()
        expanded_results = self.step4_graph_expansion(title_list)
        stats["latency"]["expansion"] = time.time() - start_time
        stats["expanded_results"] = len(expanded_results)
        
        self._add_to_dict_with_dedup(expanded_results, title_sentence_dict, dedup_type="keywords")
        
        start_time = time.time()
        reranked = self.step5_rerank(query, title_sentence_dict)
        stats["latency"]["rerank"] = time.time() - start_time
        stats["reranked_results"] = len(reranked)
        
        start_time = time.time()
        all_titles, mapped = self.step6_parent_chunk_mapping(reranked)
        stats["latency"]["mapping"] = time.time() - start_time
        
        start_time = time.time()
        scored = self.step7_llm_scoring(query, mapped)
        stats["latency"]["scoring"] = time.time() - start_time
        
        start_time = time.time()
        updated = self.step8_update_low_score_docs(query, scored)
        stats["latency"]["update"] = time.time() - start_time
        
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
        """评估单个查询"""
        retrieved_titles, stats = self.run_adaptive_retrieval(query)
        
        retrieved_list = retrieved_titles[:max_k]
        
        result = self.metrics_calculator.compute(retrieved_list, list(relevant_titles))
        
        retrieved_set = set(retrieved_titles)
        title_recall = len(retrieved_set & relevant_titles) / len(relevant_titles) if relevant_titles else 0
        title_precision = len(retrieved_set & relevant_titles) / len(retrieved_set) if retrieved_set else 0
        
        generated_answer = self.step9_generate_answer(query, [{"title": t, "content": self.title_to_content.get(t, t)} for t in retrieved_titles[:5]])
        
        gen_result = self.generation_metrics.compute(
            predicted=generated_answer,
            ground_truth=ground_truth,
            compute_semantic=True,
            embedding_client=self.embedding_client,
        )
        
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
        }
    
    def run_experiment_with_params(
        self,
        k1: int = 10,
        k2: int = 20,
        k4: int = 7,
        threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """使用指定参数运行实验"""
        self.current_k1 = k1
        self.current_k2 = k2
        self.current_k4 = k4
        self.current_threshold = threshold
        self.metrics_calculator = RetrievalMetrics(k_values=[self.current_k4])
        
        test_data = self.sampled_data
        max_k = self.current_k4
        
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
        }
        
        for idx in test_data.index:
            row = test_data.loc[idx]
            query = row["question"]
            ground_truth = row.get("answer", "")
            relevant_titles = self.get_relevant_titles(row)
            
            try:
                result = self.evaluate_single_query(query, ground_truth, relevant_titles, max_k)
                
                metrics = result["metrics"]
                aggregated_metrics["recall_at_k"].append(
                    metrics["recall_at_k"].get(self.current_k4, 0)
                )
                aggregated_metrics["precision_at_k"].append(
                    metrics["precision_at_k"].get(self.current_k4, 0)
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
                
            except Exception as e:
                logger.error(f"处理查询失败 (idx={idx}): {e}")
                continue
        
        final_metrics = {
            "recall_at_k": np.mean(aggregated_metrics["recall_at_k"]) if aggregated_metrics["recall_at_k"] else 0.0,
            "precision_at_k": np.mean(aggregated_metrics["precision_at_k"]) if aggregated_metrics["precision_at_k"] else 0.0,
            "mrr": np.mean(aggregated_metrics["mrr"]) if aggregated_metrics["mrr"] else 0.0,
            "ndcg": np.mean(aggregated_metrics["ndcg"]) if aggregated_metrics["ndcg"] else 0.0,
            "map_score": np.mean(aggregated_metrics["map_score"]) if aggregated_metrics["map_score"] else 0.0,
            "hit_rate": np.mean(aggregated_metrics["hit_rate"]) if aggregated_metrics["hit_rate"] else 0.0,
            "title_recall": np.mean(aggregated_metrics["title_recall"]) if aggregated_metrics["title_recall"] else 0.0,
            "title_precision": np.mean(aggregated_metrics["title_precision"]) if aggregated_metrics["title_precision"] else 0.0,
            "exact_match": np.mean(aggregated_metrics["exact_match"]) if aggregated_metrics["exact_match"] else 0.0,
            "f1_score": np.mean(aggregated_metrics["f1_score"]) if aggregated_metrics["f1_score"] else 0.0,
            "semantic_similarity": np.nanmean(aggregated_metrics["semantic_similarity"]) if aggregated_metrics["semantic_similarity"] else 0.0,
        }
        
        return {
            "params": {
                "k1": k1,
                "k2": k2,
                "k4": k4,
                "threshold": threshold,
            },
            "metrics": final_metrics,
        }
    
    def _generate_orthogonal_array(self) -> List[Tuple]:
        """生成正交实验设计数组
        
        使用 L16(4^4) 正交表变体，适配混合水平 (4×3×4×4)
        保证每个因素的每个水平与其他因素的所有水平均匀搭配
        """
        n_k1 = len(self.k1_values)
        n_k2 = len(self.k2_values)
        n_k4 = len(self.k4_values)
        n_th = len(self.threshold_values)
        
        combinations = []
        
        base_table = [
            [0, 0, 0, 0],
            [0, 1, 1, 2],
            [0, 2, 2, 3],
            [0, 3, 3, 1],
            [1, 0, 1, 3],
            [1, 1, 0, 1],
            [1, 2, 3, 0],
            [1, 3, 2, 2],
            [2, 0, 2, 1],
            [2, 1, 3, 3],
            [2, 2, 0, 2],
            [2, 3, 1, 0],
            [3, 0, 3, 2],
            [3, 1, 2, 0],
            [3, 2, 1, 3],
            [3, 3, 0, 1],
        ]
        
        for row in base_table:
            k1_idx = row[0] % n_k1
            k2_idx = row[1] % n_k2
            k4_idx = row[2] % n_k4
            th_idx = row[3] % n_th
            
            combinations.append((
                self.k1_values[k1_idx],
                self.k2_values[k2_idx],
                self.k4_values[k4_idx],
                self.threshold_values[th_idx],
            ))
        
        unique_combinations = list(dict.fromkeys(combinations))
        
        return unique_combinations
    
    def run_sensitivity_analysis(self, resume: bool = True) -> Dict[str, Any]:
        """运行正交实验设计参数敏感性分析"""
        logger.info("=" * 60)
        logger.info("开始参数敏感性分析（正交实验设计）")
        logger.info("=" * 60)
        
        all_combinations = self._generate_orthogonal_array()
        
        total_combinations = len(all_combinations)
        logger.info(f"正交设计组合数: {total_combinations}")
        logger.info(f"k1: {self.k1_values}")
        logger.info(f"k2: {self.k2_values}")
        logger.info(f"k4: {self.k4_values}")
        logger.info(f"threshold: {self.threshold_values}")
        logger.info(f"原始全组合数: {len(self.k1_values) * len(self.k2_values) * len(self.k4_values) * len(self.threshold_values)}")
        logger.info(f"节省实验次数: {len(self.k1_values) * len(self.k2_values) * len(self.k4_values) * len(self.threshold_values) - total_combinations}")
        
        results = []
        completed_combinations = 0
        
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                results = checkpoint.get("results", [])
                completed_combinations = checkpoint.get("completed_combinations", 0)
                logger.info(f"从检查点恢复: 已完成 {completed_combinations} 组合")
        
        remaining_combinations = all_combinations[completed_combinations:]
        
        if not remaining_combinations:
            logger.info("所有组合已测试完成")
        else:
            pbar = tqdm(remaining_combinations, desc="参数组合测试", initial=completed_combinations, total=total_combinations)
            
            for k1, k2, k4, threshold in remaining_combinations:
                logger.info(f"\n测试组合: k1={k1}, k2={k2}, k4={k4}, threshold={threshold}")
                
                try:
                    result = self.run_experiment_with_params(
                        k1=k1,
                        k2=k2,
                        k4=k4,
                        threshold=threshold,
                    )
                    results.append(result)
                    completed_combinations += 1
                    
                    if completed_combinations % self.checkpoint_interval == 0:
                        self._save_checkpoint({
                            "results": results,
                            "completed_combinations": completed_combinations,
                            "total_combinations": total_combinations,
                            "timestamp": datetime.now().isoformat(),
                        })
                        
                except Exception as e:
                    logger.error(f"测试组合失败: {e}")
                    results.append({
                        "params": {"k1": k1, "k2": k2, "k4": k4, "threshold": threshold},
                        "metrics": {},
                        "error": str(e),
                    })
                    completed_combinations += 1
                
                pbar.update(1)
            
            pbar.close()
        
        analysis = self._analyze_combination_results(results)
        
        self._save_results(results, analysis)
        
        self._clear_checkpoint()
        
        return {"results": results, "analysis": analysis}
    
    def _analyze_combination_results(self, results: List[Dict]) -> Dict[str, Any]:
        """分析排列组合结果"""
        valid_results = [r for r in results if "error" not in r and r.get("metrics")]
        
        if not valid_results:
            return {"error": "无有效结果"}
        
        all_recall = [r["metrics"].get("recall_at_k", 0) for r in valid_results]
        all_precision = [r["metrics"].get("precision_at_k", 0) for r in valid_results]
        all_f1 = [r["metrics"].get("f1_score", 0) for r in valid_results]
        all_em = [r["metrics"].get("exact_match", 0) for r in valid_results]
        
        best_recall_idx = np.argmax(all_recall)
        best_f1_idx = np.argmax(all_f1)
        best_em_idx = np.argmax(all_em)
        
        best_recall_result = valid_results[best_recall_idx]
        best_f1_result = valid_results[best_f1_idx]
        best_em_result = valid_results[best_em_idx]
        
        k1_impact = self._analyze_single_param_impact(valid_results, "k1", self.k1_values)
        k2_impact = self._analyze_single_param_impact(valid_results, "k2", self.k2_values)
        k4_impact = self._analyze_single_param_impact(valid_results, "k4", self.k4_values)
        threshold_impact = self._analyze_single_param_impact(valid_results, "threshold", self.threshold_values)
        
        overall_std = np.mean([
            k1_impact.get("recall_std", 0),
            k2_impact.get("precision_std", 0),
            k4_impact.get("f1_std", 0),
            threshold_impact.get("em_std", 0),
        ])
        
        return {
            "total_combinations": len(results),
            "valid_combinations": len(valid_results),
            "best_recall": {
                "params": best_recall_result["params"],
                "recall@10": all_recall[best_recall_idx],
            },
            "best_f1": {
                "params": best_f1_result["params"],
                "f1": all_f1[best_f1_idx],
            },
            "best_exact_match": {
                "params": best_em_result["params"],
                "em": all_em[best_em_idx],
            },
            "overall_metrics": {
                "recall_mean": float(np.mean(all_recall)),
                "recall_std": float(np.std(all_recall)),
                "recall_range": float(max(all_recall) - min(all_recall)),
                "f1_mean": float(np.mean(all_f1)),
                "f1_std": float(np.std(all_f1)),
                "f1_range": float(max(all_f1) - min(all_f1)),
                "em_mean": float(np.mean(all_em)),
                "em_std": float(np.std(all_em)),
                "em_range": float(max(all_em) - min(all_em)),
            },
            "param_impact": {
                "k1": k1_impact,
                "k2": k2_impact,
                "k4": k4_impact,
                "threshold": threshold_impact,
            },
            "robustness": {
                "overall_std": float(overall_std),
                "level": "强" if overall_std < 0.02 else ("中等" if overall_std < 0.05 else "弱"),
                "conclusion": "方法具有良好的鲁棒性，性能对参数变化不敏感" if overall_std < 0.03 else "方法鲁棒性一般",
            },
        }
    
    def _analyze_single_param_impact(
        self,
        results: List[Dict],
        param_name: str,
        param_values: List,
    ) -> Dict[str, Any]:
        """分析单个参数的影响"""
        param_groups = {v: [] for v in param_values}
        
        for r in results:
            param_val = r["params"].get(param_name)
            if param_val in param_groups:
                param_groups[param_val].append(r)
        
        recall_means = []
        precision_means = []
        f1_means = []
        em_means = []
        
        for val in param_values:
            group = param_groups[val]
            if group:
                recalls = [r["metrics"].get("recall_at_k", 0) for r in group]
                precisions = [r["metrics"].get("precision_at_k", 0) for r in group]
                f1s = [r["metrics"].get("f1_score", 0) for r in group]
                ems = [r["metrics"].get("exact_match", 0) for r in group]
                
                recall_means.append(np.mean(recalls))
                precision_means.append(np.mean(precisions))
                f1_means.append(np.mean(f1s))
                em_means.append(np.mean(ems))
            else:
                recall_means.append(0)
                precision_means.append(0)
                f1_means.append(0)
                em_means.append(0)
        
        return {
            "param_values": list(param_values),
            "recall_means": [float(x) for x in recall_means],
            "recall_std": float(np.std(recall_means)) if len(recall_means) > 1 else 0.0,
            "precision_means": [float(x) for x in precision_means],
            "precision_std": float(np.std(precision_means)) if len(precision_means) > 1 else 0.0,
            "f1_means": [float(x) for x in f1_means],
            "f1_std": float(np.std(f1_means)) if len(f1_means) > 1 else 0.0,
            "em_means": [float(x) for x in em_means],
            "em_std": float(np.std(em_means)) if len(em_means) > 1 else 0.0,
        }
    
    def _save_results(self, results: List[Dict], analysis: Dict) -> None:
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = {
            "experiment_name": "parameter_sensitivity_combination",
            "timestamp": timestamp,
            "config": {
                "test_data_path": self.test_data_path,
                "sample_size": self.sample_size,
                "k1_values": self.k1_values,
                "k2_values": self.k2_values,
                "k4_values": self.k4_values,
                "threshold_values": self.threshold_values,
                "total_combinations": len(results),
            },
            "analysis": analysis,
        }
        
        summary_path = self.output_dir / f"combination_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"组合分析摘要已保存: {summary_path}")
        
        rows = []
        for r in results:
            if "error" not in r and r.get("metrics"):
                rows.append({
                    "k1": r["params"]["k1"],
                    "k2": r["params"]["k2"],
                    "k4": r["params"]["k4"],
                    "threshold": r["params"]["threshold"],
                    f"Recall@{r['params']['k4']}": r["metrics"].get("recall_at_k", 0),
                    f"Precision@{r['params']['k4']}": r["metrics"].get("precision_at_k", 0),
                    "MRR": r["metrics"].get("mrr", 0),
                    "NDCG": r["metrics"].get("ndcg", 0),
                    "MAP": r["metrics"].get("map_score", 0),
                    "Exact Match": r["metrics"].get("exact_match", 0),
                    "F1": r["metrics"].get("f1_score", 0),
                    "Semantic Similarity": r["metrics"].get("semantic_similarity", 0),
                })
        
        if rows:
            df = pd.DataFrame(rows)
            df_path = self.output_dir / f"all_combinations_{timestamp}.csv"
            df.to_csv(df_path, index=False)
            logger.info(f"所有组合结果已保存: {df_path}")
        
        self._print_results(analysis)
    
    def _print_results(self, analysis: Dict) -> None:
        """打印实验结果"""
        print("\n" + "=" * 80)
        print("参数敏感性分析结果（排列组合）")
        print("=" * 80)
        
        print(f"\n【测试统计】")
        print(f"  总组合数: {analysis.get('total_combinations', 0)}")
        print(f"  有效组合数: {analysis.get('valid_combinations', 0)}")
        
        print(f"\n【最优参数组合】")
        best_recall = analysis.get("best_recall", {})
        print(f"  最高 Recall@10: {best_recall.get('recall@10', 0):.4f}")
        print(f"    参数: k1={best_recall.get('params', {}).get('k1')}, k2={best_recall.get('params', {}).get('k2')}, k4={best_recall.get('params', {}).get('k4')}, threshold={best_recall.get('params', {}).get('threshold')}")
        
        best_f1 = analysis.get("best_f1", {})
        print(f"  最高 F1: {best_f1.get('f1', 0):.4f}")
        print(f"    参数: k1={best_f1.get('params', {}).get('k1')}, k2={best_f1.get('params', {}).get('k2')}, k4={best_f1.get('params', {}).get('k4')}, threshold={best_f1.get('params', {}).get('threshold')}")
        
        best_em = analysis.get("best_exact_match", {})
        print(f"  最高 Exact Match: {best_em.get('em', 0):.4f}")
        print(f"    参数: k1={best_em.get('params', {}).get('k1')}, k2={best_em.get('params', {}).get('k2')}, k4={best_em.get('params', {}).get('k4')}, threshold={best_em.get('params', {}).get('threshold')}")
        
        print(f"\n【整体指标分布】")
        overall = analysis.get("overall_metrics", {})
        print(f"  Recall@10: mean={overall.get('recall_mean', 0):.4f}, std={overall.get('recall_std', 0):.4f}, range={overall.get('recall_range', 0):.4f}")
        print(f"  F1: mean={overall.get('f1_mean', 0):.4f}, std={overall.get('f1_std', 0):.4f}, range={overall.get('f1_range', 0):.4f}")
        print(f"  Exact Match: mean={overall.get('em_mean', 0):.4f}, std={overall.get('em_std', 0):.4f}, range={overall.get('em_range', 0):.4f}")
        
        print(f"\n【参数影响分析】")
        param_impact = analysis.get("param_impact", {})
        for param_name in ["k1", "k2", "k4", "threshold"]:
            impact = param_impact.get(param_name, {})
            print(f"  {param_name}:")
            print(f"    Recall std: {impact.get('recall_std', 0):.4f}")
            print(f"    F1 std: {impact.get('f1_std', 0):.4f}")
        
        print(f"\n【鲁棒性评估】")
        robustness = analysis.get("robustness", {})
        print(f"  整体标准差: {robustness.get('overall_std', 0):.4f}")
        print(f"  鲁棒性等级: {robustness.get('level', '未知')}")
        print(f"  结论: {robustness.get('conclusion', '')}")
        
        print("\n" + "=" * 80)


def main():
    """主函数"""
    test_data_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/validation-00000-of-00001.parquet"
    documents_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/valid_title_sentence.json"
    vector_store_paths = {
        "paragraph": "e:/Code_Personal/Subject/test02/data/hotpotqa/vector_stores/valid_title_sentence",
        "sentence": "e:/Code_Personal/Subject/test02/data/hotpotqa/vector_stores/single_sentence",
    }
    local_graph_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/local_graph.json"
    output_dir = "e:/Code_Personal/Subject/test02/experiments/exp9_parameter_sensitivity"
    
    experiment = ParameterSensitivityExperiment(
        test_data_path=test_data_path,
        output_dir=output_dir,
        documents_path=documents_path,
        vector_store_paths=vector_store_paths,
        local_graph_path=local_graph_path,
        use_neo4j=True,
        sample_size=500,
        k1_values=[5, 10, 15, 20],
        k2_values=[10, 20, 30],
        k4_values=[3, 5, 7, 10],
        threshold_values=[0.2, 0.3, 0.4, 0.5],
        checkpoint_interval=5,
    )
    
    experiment.load_all_resources()
    results = experiment.run_sensitivity_analysis()


if __name__ == "__main__":
    main()
