# -*- coding: utf-8 -*-
"""实验七：按问题复杂度分层实验

基于实验六的自适应策略，按问题类型（comparison/bridge）分层进行实验。
评估指标：Recall, Precision, MRR, NDCG, MAP, Recall 提升率, 图谱覆盖率, Exact Match, F1, Semantic Similarity

流程：
1. 加载数据并按 type 字段分层（comparison/bridge）
2. 对每种类型分别运行实验六的自适应策略
3. 对比两种类型的实验结果

自适应策略流程：
1. 向量检索 (FAISS, k1=10) - 同时使用段落和句子级向量库
2. 关键词提取 (LLM 提取实体)
3. 关键词检索 (Neo4j/本地图谱, k2=20)
4. 知识图谱扩展 (语义关联)
5. 重排序 (本地/API 模型, 选 top k4=10)
6. 父切片映射 (上下文整合)
7. LLM 证据强度打分 (0-1 分)
8. 低分文档更新 (查询完整内容)
9. LLM 生成最终答案
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


class ComplexityStratifiedExperiment:
    """按问题复杂度分层实验类
    
    按问题类型（comparison/bridge）分层，分别运行自适应策略实验。
    完全借鉴实验六的实现。
    """
    
    CHECKPOINT_FILE = "checkpoint.json"
    
    def __init__(
        self,
        test_data_path: str,
        output_dir: str,
        documents_path: Optional[str] = None,
        vector_store_paths: Optional[Dict[str, str]] = None,
        local_graph_path: Optional[str] = None,
        use_neo4j: bool = True,
        k1: int = 20,
        k2: int = 30,
        k4: int = 10,
        theta1: int = 10,
        checkpoint_interval: int = 10,
    ):
        """初始化实验
        
        Args:
            test_data_path: 测试数据路径
            output_dir: 输出目录
            documents_path: 文档数据路径
            vector_store_paths: 向量存储路径字典 {"paragraph": path, "sentence": path}
            local_graph_path: 本地图谱路径
            use_neo4j: 是否使用 Neo4j
            k1: 向量检索返回数量
            k2: 图谱检索返回数量
            k4: 重排序后保留数量（评估使用的K值）
            theta1: 精确匹配结果阈值，不足时启用模糊匹配
            checkpoint_interval: 检查点保存间隔
        """
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
        
        self.k1 = k1
        self.k2 = k2
        self.k4 = k4
        self.theta1 = theta1
        self.metrics_calculator = RetrievalMetrics(k_values=[self.k4])
        self.generation_metrics = GenerationMetrics()
        self.checkpoint_interval = checkpoint_interval
        
        self.config = get_config()
        
        self.graph_store: Optional[Any] = None
        self.vector_stores: Dict[str, Optional[FAISSVectorStore]] = {}
        self.embedding_client: Optional[EmbeddingClient] = None
        self.llm_client: Optional[DeepSeekClient] = None
        self.reranker: Optional[Any] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.documents: List[Dict] = []
        self.title_to_doc_ids: Dict[str, Set[str]] = {}
        self.title_to_content: Dict[str, str] = {}
        self.all_titles: Set[str] = set()
        self.total_graph_nodes: int = 0
        self.total_graph_edges: int = 0
        
    def load_graph_store(self) -> None:
        """加载图存储 - 完美复现原项目
        
        原项目图谱结构：
        - Section 节点（包含 title, sentence_total）
        - semantic 节点（语义实体）
        - SEMANTIC_LINKS 关系（Section -> semantic）
        - semantic 节点之间的各种关系
        """
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
                    logger.info(f"节点标签: {[r['labels'] for r in results]}")
                    logger.info(f"关系类型: {[r['type'] for r in rel_results]}")
                    
                    has_section = any('Section' in r['labels'] for r in results if isinstance(r['labels'], list))
                    has_semantic_links = any(r['type'] == 'SEMANTIC_LINKS' for r in rel_results)
                    
                    if has_section and has_semantic_links:
                        logger.info("✅ 图谱结构验证通过：包含 Section 节点和 SEMANTIC_LINKS 关系")
                        neo4j_connected = True
                        return
                    else:
                        logger.warning("⚠️ 图谱结构不完整：缺少 Section 节点或 SEMANTIC_LINKS 关系")
                        logger.warning("请确保 Neo4j 中已构建原项目的图谱结构（运行 Structural_Graph.py 和 Semantic_Graph.py）")
                else:
                    logger.warning("Neo4j 图谱为空，切换到本地图谱...")
                    self.graph_store.close()
            except Exception as e:
                logger.warning(f"Neo4j 连接失败: {e}")
                logger.info("切换到本地图谱...")
        
        if not neo4j_connected and self.local_graph_path:
            logger.info(f"加载本地图谱: {self.local_graph_path}")
            logger.warning("⚠️ 本地图谱不支持原项目的多跳扩展逻辑，建议使用 Neo4j 图谱")
            self.graph_store = LocalGraphStore()
            self._load_graph_from_json(self.local_graph_path)
            
            self.total_graph_nodes = self.graph_store.count_nodes()
            self.total_graph_edges = self.graph_store.count_edges()
            logger.info(f"本地图谱加载成功: nodes={self.total_graph_nodes}, edges={self.total_graph_edges}")
        elif not neo4j_connected:
            raise ValueError("需要提供 local_graph_path 或确保 Neo4j 中有原项目的图谱结构")
    
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
        
        type_counts = self.test_data['type'].value_counts()
        logger.info(f"问题类型分布: {type_counts.to_dict()}")
    
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
            results = sentence_store.search(query_vector, top_k=self.k1)
            
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
                results = store.search(query_vector, top_k=self.k1)
                
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
        """Step 2: 关键词提取 (LLM 提取实体)
        
        Args:
            query: 查询文本
            
        Returns:
            关键词列表
        """
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
        2. 如果结果不足 theta1，再模糊匹配（CONTAINS）
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
        
        cypher_strict = f"MATCH (n:Section) WHERE {strict_matches} RETURN n LIMIT {self.k2}"
        cypher_fuzzy = f"MATCH (n:Section) WHERE {fuzzy_matches} RETURN n LIMIT {self.k2}"
        
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
            
            if len(results) < self.theta1:
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
                    
                    if len(results) >= self.k2:
                        break
        except Exception as e:
            logger.warning(f"关键词检索失败: {e}")
        
        return results[:self.k2]
    
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
        """Step 5: 重排序
        
        Args:
            query: 查询文本
            candidates: 候选文档列表
            
        Returns:
            重排序后的结果列表
        """
        if not self.reranker or not candidates:
            return candidates[:self.k4]
        
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
        
        reranked = self.reranker.rerank(query, search_results, top_k=self.k4)
        
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
        """Step 6: 父切片映射（句子到段落映射）
        
        将句子级检索结果映射回完整段落：
        1. 识别句子级检索结果
        2. 从 title_to_content 获取完整段落
        3. 合并同一标题下的多个句子
        
        Args:
            candidates: 候选文档列表
            
        Returns:
            (所有标题列表, 合并后的结果列表)
        """
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
- 0.0-0.3: Little or no valid evidence
- 0.4-0.5: Contains some relevant information, but the evidence is weak, limited, or mainly local/background content
- 0.6-0.7: Provides relatively clear and useful evidence and offers strong support for the query, but it is still not sufficiently key or sufficient
- 0.8-1.0: Provides direct, key, and high-value evidential support; even if the complete answer may still require supplementation from other context chunks, this chunk itself already has high support value

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
        """Step 9: LLM 生成最终答案
        
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
    
    def _add_to_dict_with_dedup(
        self,
        items: List[Dict],
        target_dict: List[Dict],
        dedup_type: str = "vector"
    ) -> None:
        """智能去重添加到目标字典
        
        参考原项目的 add_title_sentence_to_dict：
        - dedup_type='vector': 只有 title 和 content 都一样时才算重复
        - dedup_type='keywords': 只要 title 一样就算重复
        
        Args:
            items: 待添加的项列表
            target_dict: 目标字典列表
            dedup_type: 去重类型
        """
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
    
    def run_adaptive_retrieval(self, query: str) -> Tuple[List[str], Dict, List[Dict]]:
        """运行完整的自适应检索流程
        
        Args:
            query: 查询文本
            
        Returns:
            (检索到的标题列表, 流程统计信息, 候选文档列表)
        """
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
        initial_vector_search_title = []
        
        start_time = time.time()
        vector_results = self.step1_vector_retrieval(query)
        stats["latency"]["vector"] = time.time() - start_time
        stats["vector_results"] = len(vector_results)
        
        self._add_to_dict_with_dedup(vector_results, title_sentence_dict, dedup_type="vector")
        
        initial_vector_search_title = list(set([item["title"] for item in title_sentence_dict if "title" in item]))
        
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
        
        return retrieved_titles, stats, updated
    
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
        retrieved_titles, stats, candidates = self.run_adaptive_retrieval(query)
        
        retrieved_list = retrieved_titles[:max_k]
        
        result = self.metrics_calculator.compute(retrieved_list, list(relevant_titles))
        
        retrieved_set = set(retrieved_titles)
        title_recall = len(retrieved_set & relevant_titles) / len(relevant_titles) if relevant_titles else 0
        title_precision = len(retrieved_set & relevant_titles) / len(retrieved_set) if retrieved_set else 0
        
        generated_answer = self.step9_generate_answer(query, candidates)
        
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
    
    def _save_checkpoint(
        self,
        query_type: str,
        processed_indices: List[int],
        all_results: List[Dict],
        aggregated_metrics: Dict,
        sample_size: Optional[int],
    ) -> None:
        """保存检查点"""
        checkpoint = {
            "query_type": query_type,
            "processed_indices": processed_indices,
            "all_results": all_results,
            "aggregated_metrics": aggregated_metrics,
            "sample_size": sample_size,
            "timestamp": datetime.now().isoformat(),
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_{query_type}.json"
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False)
        
        logger.info(f"检查点已保存 ({query_type}): 已处理 {len(processed_indices)} 条")
    
    def _load_checkpoint(self, query_type: str) -> Optional[Dict]:
        """加载检查点"""
        checkpoint_path = self.output_dir / f"checkpoint_{query_type}.json"
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            
            if "processed_indices" in checkpoint:
                checkpoint["processed_indices"] = [int(idx) for idx in checkpoint["processed_indices"]]
            
            logger.info(f"从检查点恢复 ({query_type}): 已处理 {len(checkpoint['processed_indices'])} 条")
            return checkpoint
        except Exception as e:
            logger.warning(f"加载检查点失败: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            return None
    
    def _clear_checkpoint(self, query_type: str) -> None:
        """清除检查点"""
        checkpoint_path = self.output_dir / f"checkpoint_{query_type}.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"检查点已清除 ({query_type})")
    
    def run_experiment_for_type(
        self,
        query_type: str,
        test_data: pd.DataFrame,
        sample_size: Optional[int] = None,
        save_details: bool = True,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """运行特定类型问题的实验
        
        Args:
            query_type: 问题类型 (comparison/bridge)
            test_data: 测试数据
            sample_size: 采样数量
            save_details: 是否保存详细结果
            resume: 是否从检查点恢复
            
        Returns:
            实验结果
        """
        logger.info(f"开始运行 {query_type} 类型问题实验...")
        
        if sample_size:
            test_data = test_data.sample(n=sample_size, random_state=42)
            logger.info(f"采样 {sample_size} 条数据进行测试")
        
        max_k = self.k4
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
            "vector_latency": [],
            "graph_latency": [],
            "total_latency": [],
        }
        
        if resume:
            checkpoint = self._load_checkpoint(query_type)
            if checkpoint:
                if checkpoint.get("sample_size") == sample_size:
                    processed_indices = checkpoint["processed_indices"]
                    all_results = checkpoint["all_results"]
                    aggregated_metrics = checkpoint["aggregated_metrics"]
                    logger.info(f"从检查点恢复成功，跳过 {len(processed_indices)} 条已处理数据")
                else:
                    logger.info("采样数量不匹配，从头开始")
        
        processed_set = set(processed_indices)
        remaining_indices = [idx for idx in test_indices if idx not in processed_set]
        
        if not remaining_indices:
            logger.info(f"所有 {query_type} 类型数据已处理完成")
        else:
            pbar = tqdm(remaining_indices, desc=f"评估进度 ({query_type})", initial=len(processed_indices), total=len(test_indices))
            
            for idx in remaining_indices:
                row = test_data.loc[idx]
                query = row["question"]
                ground_truth = row.get("answer", "")
                relevant_titles = self.get_relevant_titles(row)
                
                try:
                    result = self.evaluate_single_query(query, ground_truth, relevant_titles, max_k)
                    
                    metrics = result["metrics"]
                    aggregated_metrics["recall_at_k"].append(
                        metrics["recall_at_k"].get(self.k4, 0)
                    )
                    aggregated_metrics["precision_at_k"].append(
                        metrics["precision_at_k"].get(self.k4, 0)
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
                    
                    stats = result.get("stats", {})
                    aggregated_metrics["vector_latency"].append(
                        stats.get("latency", {}).get("vector", 0)
                    )
                    aggregated_metrics["graph_latency"].append(
                        stats.get("latency", {}).get("graph", 0)
                    )
                    aggregated_metrics["total_latency"].append(
                        sum(stats.get("latency", {}).values())
                    )
                    
                    if save_details:
                        all_results.append({
                            "id": row.get("id", idx),
                            "question": query,
                            "answer": ground_truth,
                            "relevant_titles": list(relevant_titles),
                            "retrieved_titles": result["retrieved_titles"],
                            "generated_answer": result["generated_answer"],
                            "metrics": metrics,
                            "generation_metrics": gen_metrics,
                            "title_recall": result["title_recall"],
                            "title_precision": result["title_precision"],
                            "stats": stats,
                        })
                    
                    processed_indices.append(idx)
                    
                    if len(processed_indices) % self.checkpoint_interval == 0:
                        self._save_checkpoint(
                            query_type, processed_indices, all_results, aggregated_metrics, sample_size
                        )
                        
                except Exception as e:
                    logger.error(f"处理查询失败 (idx={idx}, type={query_type}): {e}")
                    continue
                
                pbar.update(1)
            
            pbar.close()
        
        final_metrics = {
            "recall_at_k": {
                self.k4: np.mean(aggregated_metrics["recall_at_k"]) if aggregated_metrics["recall_at_k"] else 0
            },
            "precision_at_k": {
                self.k4: np.mean(aggregated_metrics["precision_at_k"]) if aggregated_metrics["precision_at_k"] else 0
            },
            "mrr": np.mean(aggregated_metrics["mrr"]) if aggregated_metrics["mrr"] else 0,
            "ndcg": np.mean(aggregated_metrics["ndcg"]) if aggregated_metrics["ndcg"] else 0,
            "map_score": np.mean(aggregated_metrics["map_score"]) if aggregated_metrics["map_score"] else 0,
            "hit_rate": np.mean(aggregated_metrics["hit_rate"]) if aggregated_metrics["hit_rate"] else 0,
            "title_recall": np.mean(aggregated_metrics["title_recall"]) if aggregated_metrics["title_recall"] else 0,
            "title_precision": np.mean(aggregated_metrics["title_precision"]) if aggregated_metrics["title_precision"] else 0,
            "exact_match": np.mean(aggregated_metrics["exact_match"]) if aggregated_metrics["exact_match"] else 0,
            "f1_score": np.mean(aggregated_metrics["f1_score"]) if aggregated_metrics["f1_score"] else 0,
            "semantic_similarity": np.nanmean(aggregated_metrics["semantic_similarity"]) if aggregated_metrics["semantic_similarity"] else 0,
            "avg_vector_latency": np.mean(aggregated_metrics["vector_latency"]) if aggregated_metrics["vector_latency"] else 0,
            "avg_graph_latency": np.mean(aggregated_metrics["graph_latency"]) if aggregated_metrics["graph_latency"] else 0,
            "avg_total_latency": np.mean(aggregated_metrics["total_latency"]) if aggregated_metrics["total_latency"] else 0,
        }
        
        baseline_recall = 0.1
        recall_improvement = {
            k: (v - baseline_recall) / baseline_recall * 100 if baseline_recall > 0 else 0
            for k, v in final_metrics["recall_at_k"].items()
        }
        final_metrics["recall_improvement"] = recall_improvement
        
        graph_coverage = self.total_graph_nodes if self.total_graph_nodes > 0 else 0
        final_metrics["graph_coverage"] = graph_coverage
        
        experiment_result = {
            "experiment_name": f"complexity_stratified_{query_type}",
            "query_type": query_type,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "test_data_path": self.test_data_path,
                "documents_path": self.documents_path,
                "vector_store_paths": self.vector_store_paths,
                "use_neo4j": self.use_neo4j,
                "k1": self.k1,
                "k2": self.k2,
                "k4": self.k4,
                "theta1": self.theta1,
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
        
        self._clear_checkpoint(query_type)
        
        return experiment_result
    
    def run_experiment(
        self,
        sample_size_per_type: Optional[int] = None,
        save_details: bool = True,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """运行完整的分层实验
        
        Args:
            sample_size_per_type: 每种类型的采样数量
            save_details: 是否保存详细结果
            resume: 是否从检查点恢复
            
        Returns:
            完整的实验结果
        """
        logger.info("开始运行按问题复杂度分层实验...")
        
        self.load_graph_store()
        self.load_documents()
        self.load_vector_stores()
        self.load_embedding_client()
        self.load_llm_client()
        self.load_reranker()
        self.load_test_data()
        
        comparison_data = self.test_data[self.test_data['type'] == 'comparison']
        bridge_data = self.test_data[self.test_data['type'] == 'bridge']
        
        logger.info(f"Comparison 类型问题数量: {len(comparison_data)}")
        logger.info(f"Bridge 类型问题数量: {len(bridge_data)}")
        
        comparison_results = self.run_experiment_for_type(
            query_type="comparison",
            test_data=comparison_data,
            sample_size=sample_size_per_type,
            save_details=save_details,
            resume=resume,
        )
        
        bridge_results = self.run_experiment_for_type(
            query_type="bridge",
            test_data=bridge_data,
            sample_size=sample_size_per_type,
            save_details=save_details,
            resume=resume,
        )
        
        comparison_summary = self._create_summary(comparison_results)
        bridge_summary = self._create_summary(bridge_results)
        
        comparison_df = self._create_metrics_dataframe(comparison_results, "comparison")
        bridge_df = self._create_metrics_dataframe(bridge_results, "bridge")
        
        comparison_df.to_csv(self.output_dir / "metrics_comparison.csv", index=False)
        bridge_df.to_csv(self.output_dir / "metrics_bridge.csv", index=False)
        
        if save_details:
            if comparison_results.get("details"):
                with open(self.output_dir / "details_comparison.json", "w", encoding="utf-8") as f:
                    json.dump(comparison_results["details"], f, ensure_ascii=False, indent=2)
            if bridge_results.get("details"):
                with open(self.output_dir / "details_bridge.json", "w", encoding="utf-8") as f:
                    json.dump(bridge_results["details"], f, ensure_ascii=False, indent=2)
        
        comparison_report = {k: v for k, v in comparison_results.items() if k != "details"}
        bridge_report = {k: v for k, v in bridge_results.items() if k != "details"}
        
        with open(self.output_dir / "experiment_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison_report, f, ensure_ascii=False, indent=2)
        with open(self.output_dir / "experiment_bridge.json", "w", encoding="utf-8") as f:
            json.dump(bridge_report, f, ensure_ascii=False, indent=2)
        
        comparison_metrics = comparison_results["metrics"]
        bridge_metrics = bridge_results["metrics"]
        
        comparison_report = {
            "query_type": "comparison",
            "sample_size": comparison_results["config"]["total_test_samples"],
            "metrics": comparison_metrics,
        }
        
        bridge_report = {
            "query_type": "bridge",
            "sample_size": bridge_results["config"]["total_test_samples"],
            "metrics": bridge_metrics,
        }
        
        self._print_comparison_results(comparison_report, bridge_report)
        
        return {
            "comparison": comparison_results,
            "bridge": bridge_results,
        }
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """创建结果摘要"""
        return {
            "query_type": results["query_type"],
            "sample_size": results["config"]["total_test_samples"],
            "metrics": results["metrics"],
        }
    
    def _create_metrics_dataframe(self, results: Dict[str, Any], query_type: str) -> pd.DataFrame:
        """创建指标数据框"""
        metrics = results["metrics"]
        
        rows = [
            {
                "Query_Type": query_type,
                "Metric": "Recall@K",
                **{f"K={k}": v for k, v in metrics["recall_at_k"].items()},
            },
            {
                "Query_Type": query_type,
                "Metric": "Precision@K",
                **{f"K={k}": v for k, v in metrics["precision_at_k"].items()},
            },
            {
                "Query_Type": query_type,
                "Metric": "Recall提升率(%)",
                **{f"K={k}": v for k, v in metrics["recall_improvement"].items()},
            },
            {
                "Query_Type": query_type,
                "Metric": "MRR",
                "Value": metrics["mrr"],
            },
            {
                "Query_Type": query_type,
                "Metric": "NDCG",
                "Value": metrics["ndcg"],
            },
            {
                "Query_Type": query_type,
                "Metric": "MAP",
                "Value": metrics["map_score"],
            },
            {
                "Query_Type": query_type,
                "Metric": "Hit Rate",
                "Value": metrics["hit_rate"],
            },
            {
                "Query_Type": query_type,
                "Metric": "Title Recall",
                "Value": metrics["title_recall"],
            },
            {
                "Query_Type": query_type,
                "Metric": "Title Precision",
                "Value": metrics["title_precision"],
            },
            {
                "Query_Type": query_type,
                "Metric": "Exact Match",
                "Value": metrics["exact_match"],
            },
            {
                "Query_Type": query_type,
                "Metric": "F1 Score",
                "Value": metrics["f1_score"],
            },
            {
                "Query_Type": query_type,
                "Metric": "Semantic Similarity",
                "Value": metrics["semantic_similarity"],
            },
            {
                "Query_Type": query_type,
                "Metric": "图谱覆盖率",
                "Value": metrics["graph_coverage"],
            },
            {
                "Query_Type": query_type,
                "Metric": "平均向量检索延迟(s)",
                "Value": metrics["avg_vector_latency"],
            },
            {
                "Query_Type": query_type,
                "Metric": "平均图谱检索延迟(s)",
                "Value": metrics["avg_graph_latency"],
            },
            {
                "Query_Type": query_type,
                "Metric": "平均总延迟(s)",
                "Value": metrics["avg_total_latency"],
            },
        ]
        
        return pd.DataFrame(rows)
    
    def _print_comparison_results(
        self,
        comparison_report: Dict[str, Any],
        bridge_report: Dict[str, Any],
    ) -> None:
        """打印对比结果"""
        print("\n" + "=" * 80)
        print("实验七：按问题复杂度分层实验结果对比")
        print("=" * 80)
        
        print(f"\n【数据统计】")
        print(f"  Comparison 类型: {comparison_report['sample_size']} 条")
        print(f"  Bridge 类型: {bridge_report['sample_size']} 条")
        
        print("\n" + "-" * 80)
        print("【检索指标对比】")
        print("-" * 80)
        
        comp_metrics = comparison_report["metrics"]
        bridge_metrics = bridge_report["metrics"]
        
        print(f"\n{'指标':<20} {'Comparison':<15} {'Bridge':<15} {'差异':<15}")
        print("-" * 65)
        
        print(f"{'MRR':<20} {comp_metrics['mrr']:<15.4f} {bridge_metrics['mrr']:<15.4f} {comp_metrics['mrr'] - bridge_metrics['mrr']:<15.4f}")
        print(f"{'NDCG':<20} {comp_metrics['ndcg']:<15.4f} {bridge_metrics['ndcg']:<15.4f} {comp_metrics['ndcg'] - bridge_metrics['ndcg']:<15.4f}")
        print(f"{'MAP':<20} {comp_metrics['map_score']:<15.4f} {bridge_metrics['map_score']:<15.4f} {comp_metrics['map_score'] - bridge_metrics['map_score']:<15.4f}")
        print(f"{'Hit Rate':<20} {comp_metrics['hit_rate']:<15.4f} {bridge_metrics['hit_rate']:<15.4f} {comp_metrics['hit_rate'] - bridge_metrics['hit_rate']:<15.4f}")
        print(f"{'Title Recall':<20} {comp_metrics['title_recall']:<15.4f} {bridge_metrics['title_recall']:<15.4f} {comp_metrics['title_recall'] - bridge_metrics['title_recall']:<15.4f}")
        print(f"{'Title Precision':<20} {comp_metrics['title_precision']:<15.4f} {bridge_metrics['title_precision']:<15.4f} {comp_metrics['title_precision'] - bridge_metrics['title_precision']:<15.4f}")
        
        print("\n【Recall@K 和 Precision@K 对比】")
        print(f"{'K值':<10} {'Comparison Recall':<20} {'Bridge Recall':<20} {'差异':<15}")
        print("-" * 65)
        comp_recall = comp_metrics['recall_at_k']
        bridge_recall = bridge_metrics['recall_at_k']
        diff = comp_recall - bridge_recall
        print(f"K={self.k4:<8} {comp_recall:<20.4f} {bridge_recall:<20.4f} {diff:<15.4f}")
        
        print(f"\n{'K值':<10} {'Comparison Precision':<20} {'Bridge Precision':<20} {'差异':<15}")
        print("-" * 65)
        comp_prec = comp_metrics['precision_at_k']
        bridge_prec = bridge_metrics['precision_at_k']
        diff = comp_prec - bridge_prec
        print(f"K={self.k4:<8} {comp_prec:<20.4f} {bridge_prec:<20.4f} {diff:<15.4f}")
        
        print("\n" + "-" * 80)
        print("【生成指标对比】")
        print("-" * 80)
        
        print(f"\n{'指标':<20} {'Comparison':<15} {'Bridge':<15} {'差异':<15}")
        print("-" * 65)
        print(f"{'Exact Match':<20} {comp_metrics['exact_match']:<15.4f} {bridge_metrics['exact_match']:<15.4f} {comp_metrics['exact_match'] - bridge_metrics['exact_match']:<15.4f}")
        print(f"{'F1 Score':<20} {comp_metrics['f1_score']:<15.4f} {bridge_metrics['f1_score']:<15.4f} {comp_metrics['f1_score'] - bridge_metrics['f1_score']:<15.4f}")
        print(f"{'Semantic Similarity':<20} {comp_metrics['semantic_similarity']:<15.4f} {bridge_metrics['semantic_similarity']:<15.4f} {comp_metrics['semantic_similarity'] - bridge_metrics['semantic_similarity']:<15.4f}")
        
        print("\n" + "-" * 80)
        print("【延迟指标对比】")
        print("-" * 80)
        
        print(f"\n{'指标':<20} {'Comparison':<15} {'Bridge':<15} {'差异':<15}")
        print("-" * 65)
        print(f"{'向量检索延迟(s)':<20} {comp_metrics['avg_vector_latency']:<15.4f} {bridge_metrics['avg_vector_latency']:<15.4f} {comp_metrics['avg_vector_latency'] - bridge_metrics['avg_vector_latency']:<15.4f}")
        print(f"{'图谱检索延迟(s)':<20} {comp_metrics['avg_graph_latency']:<15.4f} {bridge_metrics['avg_graph_latency']:<15.4f} {comp_metrics['avg_graph_latency'] - bridge_metrics['avg_graph_latency']:<15.4f}")
        print(f"{'总延迟(s)':<20} {comp_metrics['avg_total_latency']:<15.4f} {bridge_metrics['avg_total_latency']:<15.4f} {comp_metrics['avg_total_latency'] - bridge_metrics['avg_total_latency']:<15.4f}")
        
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
    output_dir = "e:/Code_Personal/Subject/test02/experiments/exp7_complexity_stratified"
    
    experiment = ComplexityStratifiedExperiment(
        test_data_path=test_data_path,
        output_dir=output_dir,
        documents_path=documents_path,
        vector_store_paths=vector_store_paths,
        local_graph_path=local_graph_path,
        use_neo4j=True,
        k1=10,
        k2=20,
        k4=7,
        theta1=10,
        checkpoint_interval=10,
    )
    
    results = experiment.run_experiment(
        sample_size_per_type=None,
        save_details=True,
        resume=True,
    )


if __name__ == "__main__":
    main()
