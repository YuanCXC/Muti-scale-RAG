# -*- coding: utf-8 -*-
"""实验六改进版：LLM驱动的自适应多尺度检索

核心逻辑：LLM判断驱动的顺序路由
初始检索 → 路径①(细粒度) → LLM判断是否足够?
    ├─ 足够 → 返回结果
    └─ 不足够 → 路径②(局部回升) → LLM判断是否足够?
                    ├─ 足够 → 返回结果
                    └─ 不足够 → 路径③(图扩展) → 返回结果

路径说明：
- 路径① 细粒度保持：重排序选top-k，适合简单问题
- 路径② 局部尺度回升：回溯父节点补全上下文，适合碎片化证据
- 路径③ 图扩展+多尺度组装：多跳扩展获取桥接证据，适合复杂问题
"""

import json
import os
import re
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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


class RoutingPath(Enum):
    FINE_GRAINED = "path_1_fine_grained"
    LOCAL_ESCALATION = "path_2_local_escalation"
    GRAPH_EXPANSION = "path_3_graph_expansion"


@dataclass
class EvidenceUnit:
    id: str
    title: str
    content: str
    granularity: str = "sentence"
    source: str = "unknown"
    score: float = 0.0
    is_sentence_level: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalVariant:
    """实验/消融变体配置。"""

    name: str
    description: str
    enable_graph_expansion: bool = True
    enable_scale_rebound: bool = True
    enable_summary_evidence: bool = True


class LLMAdaptiveRetrieval:
    """LLM驱动的自适应多尺度检索实验"""

    CHECKPOINT_FILE = "checkpoint.json"

    def __init__(
        self,
        test_data_path: str,
        output_dir: str,
        documents_path: Optional[str] = None,
        vector_store_paths: Optional[Dict[str, str]] = None,
        local_graph_path: Optional[str] = None,
        use_neo4j: bool = True,
        k1: int = 10,
        k2: int = 20,
        k4: int = 7,
        max_hops: int = 3,
        max_degree: int = 10,
        token_budget: int = 3600,
        checkpoint_interval: int = 10,
        max_workers: int = 4,
    ):
        self.test_data_path = test_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.documents_path = documents_path
        self.vector_store_paths = vector_store_paths or {}
        self.local_graph_path = local_graph_path
        self.use_neo4j = use_neo4j

        self.k1 = k1
        self.k2 = k2
        self.k4 = k4
        self.max_hops = max_hops
        self.max_degree = max_degree
        self.token_budget = token_budget
        self.checkpoint_interval = checkpoint_interval
        self.max_workers = max_workers

        self.config = get_config()
        self.metrics_calculator = RetrievalMetrics(k_values=[self.k4])
        self.generation_metrics = GenerationMetrics()
        
        self.score_weights = {
            "semantic": 0.35,
            "complement": 0.30,
            "redundancy": 0.20,
            "cost": 0.15,
        }

        self.graph_store: Optional[Any] = None
        self.vector_stores: Dict[str, Optional[FAISSVectorStore]] = {}
        self.embedding_client: Optional[EmbeddingClient] = None
        self.llm_client: Optional[DeepSeekClient] = None
        self.reranker: Optional[Any] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.documents: List[Dict] = []
        self.title_to_content: Dict[str, str] = {}
        self.all_titles: Set[str] = set()
        self.total_graph_nodes: int = 0
        self.total_graph_edges: int = 0

    def load_all(self) -> None:
        """加载所有资源"""
        self._load_graph_store()
        self._load_documents()
        self._load_vector_stores()
        self._load_embedding_client()
        self._load_llm_client()
        self._load_reranker()
        self._load_test_data()

    def _load_graph_store(self) -> None:
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
                    has_section = any('Section' in r['labels'] for r in results if isinstance(r['labels'], list))
                    has_semantic_links = any(r['type'] == 'SEMANTIC_LINKS' for r in rel_results)

                    if has_section and has_semantic_links:
                        neo4j_connected = True
                        return
            except Exception as e:
                logger.warning(f"Neo4j 连接失败: {e}")

        if not neo4j_connected and self.local_graph_path:
            logger.info(f"加载本地图谱: {self.local_graph_path}")
            self.graph_store = LocalGraphStore()
            self._load_graph_from_json(self.local_graph_path)

            self.total_graph_nodes = self.graph_store.count_nodes()
            self.total_graph_edges = self.graph_store.count_edges()
            logger.info(f"本地图谱加载成功: nodes={self.total_graph_nodes}, edges={self.total_graph_edges}")

    def _load_graph_from_json(self, json_path: str) -> None:
        json_file = Path(json_path)
        if json_file.is_dir():
            json_file = json_file / "graph.json"
        if not json_file.exists():
            json_file = Path(json_path).parent / "local_graph.json"

        logger.info(f"从 JSON 加载图谱: {json_file}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            node_map: Dict[str, Node] = {}
            for triple in tqdm(data, desc="构建图谱"):
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

    def _load_documents(self) -> None:
        if not self.documents_path:
            return
        logger.info(f"加载文档数据: {self.documents_path}")
        with open(self.documents_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        logger.info(f"文档加载完成: {len(self.documents)} 个")

        for doc in self.documents:
            title = doc.get("title", "")
            if title:
                self.all_titles.add(title)
                self.title_to_content[title] = doc.get("sentence_total", doc.get("content", ""))
        logger.info(f"标题索引构建完成: {len(self.all_titles)} 个")

    def _load_vector_stores(self) -> None:
        for store_type, store_path in self.vector_store_paths.items():
            logger.info(f"加载 {store_type} 向量存储: {store_path}")
            try:
                store = FAISSVectorStore()
                store.load(store_path)
                self.vector_stores[store_type] = store
                logger.info(f"{store_type} 向量存储加载成功: vectors={store.count()}")
            except FileNotFoundError:
                logger.warning(f"{store_type} 向量存储文件不存在")
                self.vector_stores[store_type] = None

    def _load_embedding_client(self) -> None:
        logger.info("加载嵌入客户端...")
        self.embedding_client = EmbeddingClient()
        logger.info(f"嵌入客户端加载完成: dimension={self.embedding_client.dimension}")

    def _load_llm_client(self) -> None:
        logger.info("加载 LLM 客户端...")
        self.llm_client = DeepSeekClient()
        logger.info(f"LLM 客户端加载完成: model={self.llm_client.model}")

    def _load_reranker(self) -> None:
        logger.info("加载重排序器...")
        try:
            self.reranker = create_reranker(mode="api")
            logger.info("重排序器加载完成 (API模式)")
        except Exception as e:
            logger.warning(f"重排序器加载失败: {e}")
            self.reranker = None

    def _load_test_data(self) -> None:
        logger.info(f"加载测试数据: {self.test_data_path}")
        self.test_data = pd.read_parquet(self.test_data_path)
        logger.info(f"测试数据加载完成: {len(self.test_data)} 条")

    def initial_retrieval(self, query: str) -> List[EvidenceUnit]:
        """初始检索：向量检索 + 关键词检索"""
        all_units: List[EvidenceUnit] = []
        seen_ids: Set[str] = set()

        vector_units = self._vector_retrieval(query)
        for unit in vector_units:
            if unit.id not in seen_ids:
                all_units.append(unit)
                seen_ids.add(unit.id)

        keywords = self._extract_keywords(query)
        keyword_units = self._keyword_retrieval(keywords, query=query)
        for unit in keyword_units:
            if unit.id not in seen_ids:
                all_units.append(unit)
                seen_ids.add(unit.id)

        return all_units

    def _vector_retrieval(self, query: str) -> List[EvidenceUnit]:
        results = []
        sentence_store = self.vector_stores.get("sentence")
        if sentence_store:
            query_vector = self.embedding_client.embed(query)
            search_results = sentence_store.search(query_vector, top_k=self.k1)
            for r in search_results:
                title = r.metadata.extra.get("title", r.metadata.doc_id)
                content = r.metadata.content
                sentence_id = r.metadata.extra.get("sentence_id", "")
                results.append(EvidenceUnit(
                    id=r.id,
                    title=title,
                    content=content,
                    granularity="sentence" if sentence_id else "paragraph",
                    source="vector",
                    score=r.score,
                    is_sentence_level=bool(sentence_id),
                ))
        return results

    def _extract_keywords(self, query: str) -> List[str]:
        prompt = f"""从以下问题中提取关键实体和关键词，用于知识图谱检索。
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

    def _keyword_retrieval(self, keywords: List[str], query: str = "") -> List[EvidenceUnit]:
        if not keywords:
            return []

        results = []
        seen_titles = set()
        query_entities = self._extract_query_entities(query) if query else set()

        def cypher_str_value(val):
            if not isinstance(val, str):
                return '""'
            out = val.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{out}"'

        def extract_node_property(node, prop_name, default=""):
            if node is None:
                return default
            if isinstance(node, dict):
                return node.get(prop_name, default)
            try:
                val = node.get(prop_name)
                return val if val is not None else default
            except (AttributeError, TypeError):
                return default

        conditions = [f'n.title CONTAINS {cypher_str_value(k)}' for k in keywords]
        matches = " OR ".join(conditions)
        cypher = f"MATCH (n:Section) WHERE {matches} RETURN n LIMIT {self.k2}"

        try:
            query_results = self.graph_store.query(cypher)
            for item in query_results:
                node = item.get("n", item)
                title = extract_node_property(node, "title", "").strip('"')
                if title and title not in seen_titles:
                    content = extract_node_property(node, "sentence_total", "") or self.title_to_content.get(title, "")
                    
                    unit = EvidenceUnit(
                        id=title,
                        title=title,
                        content=content,
                        granularity="paragraph",
                        source="graph",
                        score=0.5,
                    )
                    
                    if query:
                        selected_contents = [u.content for u in results if u.content]
                        dynamic_score = self._compute_evidence_score(
                            unit, query, selected_contents, query_entities
                        )
                        unit.score = dynamic_score
                    else:
                        unit.score = 0.8
                    
                    results.append(unit)
                    seen_titles.add(title)
        except Exception as e:
            logger.warning(f"关键词检索失败: {e}")

        return results[:self.k2]

    def path1_fine_grained(self, reranked: List[EvidenceUnit], query: str) -> List[EvidenceUnit]:
        """路径①：细粒度保持 - 直接返回重排序结果"""
        logger.info("执行路径①：细粒度保持")
        return reranked[:self.k4]

    def path2_local_escalation(self, reranked: List[EvidenceUnit], query: str) -> List[EvidenceUnit]:
        """路径②：局部尺度回升 - 通过图谱PARENT_OF关系回溯父节点"""
        logger.info("执行路径②：局部尺度回升")

        escalated = []
        seen_titles = set()
        for unit in reranked[:self.k4]:
            if unit.title not in seen_titles:
                if unit.is_sentence_level and len(unit.content) < 50:
                    parent_content = self._get_parent_content(unit.title)
                    if parent_content:
                        escalated.append(EvidenceUnit(
                            id=f"parent_{unit.title}",
                            title=unit.title,
                            content=parent_content,
                            granularity="paragraph",
                            source="graph_escalation",
                            score=unit.score,
                            is_sentence_level=False,
                            metadata={"escalation_type": "parent_of", "original_granularity": "sentence"},
                        ))
                        seen_titles.add(unit.title)
                        continue
                
                escalated.append(unit)
                seen_titles.add(unit.title)

        return escalated[:self.k4]

    def _get_parent_content(self, title: str) -> Optional[str]:
        """通过图谱PARENT_OF关系获取父节点内容"""
        safe_title = title.replace('"', '\\"')
        
        cypher = f'''
        MATCH (child:Section {{title: "{safe_title}"}})
        MATCH (parent:Section)-[:PARENT_OF]->(child)
        RETURN parent.title AS parent_title, parent.sentence_total AS sentence_total
        LIMIT 1
        '''
        
        try:
            results = self.graph_store.query(cypher)
            if results:
                parent = results[0]
                parent_content = parent.get("sentence_total", "")
                if parent_content:
                    return parent_content
                parent_title = parent.get("parent_title", "").strip('"')
                if parent_title and parent_title in self.title_to_content:
                    return self.title_to_content[parent_title]
        except Exception as e:
            logger.debug(f"PARENT_OF查询失败(title={title}): {e}")
        
        cypher_contains = f'''
        MATCH (child:Section {{title: "{safe_title}"}})
        MATCH (parent:Section)-[:CONTAINS]->(child)
        RETURN parent.title AS parent_title, parent.sentence_total AS sentence_total
        LIMIT 1
        '''
        
        try:
            results = self.graph_store.query(cypher_contains)
            if results:
                parent = results[0]
                parent_content = parent.get("sentence_total", "")
                if parent_content:
                    return parent_content
                parent_title = parent.get("parent_title", "").strip('"')
                if parent_title and parent_title in self.title_to_content:
                    return self.title_to_content[parent_title]
        except Exception as e:
            logger.debug(f"CONTAINS查询失败(title={title}): {e}")
        
        if title in self.title_to_content:
            return self.title_to_content[title]
        
        return None

    def path3_graph_expansion(
        self,
        reranked: List[EvidenceUnit],
        query: str,
        enable_summary_evidence: bool = True,
    ) -> List[EvidenceUnit]:
        """路径③：图扩展+多尺度组装，使用动态打分"""
        logger.info(f"执行路径③：图扩展+多尺度组装 (max_hops={self.max_hops})")

        title_list = list(set(u.title for u in reranked if u.title))
        
        selected_contents = [u.content for u in reranked[:self.k4] if u.content]
        query_entities = self._extract_query_entities(query)
        
        expanded = self._graph_expand(
            title_list, 
            query=query,
            selected_contents=selected_contents,
            query_entities=query_entities,
            enable_summary_evidence=enable_summary_evidence,
        )

        all_units = list(reranked)
        seen_titles = set(u.title for u in all_units)
        for unit in expanded:
            if unit.title not in seen_titles:
                all_units.append(unit)
                seen_titles.add(unit.title)

        all_reranked = self._rerank(query, all_units)
        
        return all_reranked[:self.k4]

    def _rerank(self, query: str, candidates: List[EvidenceUnit]) -> List[EvidenceUnit]:
        if not self.reranker or not candidates:
            return candidates[:self.k4]

        from src.retrievers.base_retriever import SearchResult

        search_results = [
            SearchResult(
                doc_id=u.id,
                content=u.content or u.title,
                score=u.score,
                metadata={"title": u.title, "source": u.source, "granularity": u.granularity},
            )
            for u in candidates
        ]

        max_retries = 1000
        retry_delay = 2.0
        reranked = None
        
        for attempt in range(max_retries):
            try:
                reranked = self.reranker.rerank(query, search_results, top_k=min(len(candidates), self.k4))
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"重排序失败(尝试 {attempt+1}/{max_retries}): {e}，等待重试...")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"重排序失败(尝试 {max_retries}次): {e}，使用原始排序")
                    return sorted(candidates, key=lambda u: u.score, reverse=True)[:self.k4]

        result = []
        for r in reranked:
            matching = [u for u in candidates if u.id == r.doc_id]
            if matching:
                unit = matching[0]
                unit.score = r.score
                result.append(unit)
            else:
                result.append(EvidenceUnit(
                    id=r.doc_id,
                    title=r.metadata.get("title", r.doc_id),
                    content=r.content,
                    granularity=r.metadata.get("granularity", "paragraph"),
                    source=r.metadata.get("source", "reranked"),
                    score=r.score,
                ))

        return result

    def _graph_expand(
        self, 
        titles: List[str],
        query: str = "",
        selected_contents: List[str] = None,
        query_entities: Set[str] = None,
        enable_summary_evidence: bool = True,
    ) -> List[EvidenceUnit]:
        """图谱扩展，使用动态打分和摘要证据"""
        results = []
        seen_titles = set(titles)
        if selected_contents is None:
            selected_contents = []
        if query_entities is None:
            query_entities = set()

        for title in titles:
            safe_title = title.replace('"', '\\"')
            cypher_query = (
                'MATCH (start:Section {title: "' + safe_title + '"})\n'
                'MATCH (start)-[:SEMANTIC_LINKS]-(first)\n'
                f'MATCH p = (first)-[r*0..{self.max_hops}]-(last)\n'
                'MATCH (last)-[:SEMANTIC_LINKS]-(n:Section)\n'
                'WHERE n <> start\n'
                "  AND ALL(rel IN r WHERE type(rel) <> 'SEPARATES')\n"
                f"  AND ALL(x IN nodes(p) WHERE COUNT {{ (x)--() }} <= {self.max_degree})\n"
                "RETURN DISTINCT \n"
                "    n.title AS section_title,\n"
                "    n.sentence_total AS sentence_total\n"
            )

            try:
                expanded = self.graph_store.query(cypher_query)
                for item in expanded:
                    section_title = item.get("section_title", "").strip('"')
                    if not section_title or section_title in seen_titles:
                        continue
                    
                    content = item.get("sentence_total", "")
                    if not content:
                        content = self.title_to_content.get(section_title, "")

                    summary_content = None
                    if enable_summary_evidence:
                        summary_content = self._get_summary_evidence(section_title, max_hops=min(self.max_hops, 2))
                        if summary_content:
                            content = f"{content}\n\n[摘要证据]\n{summary_content}" if content else summary_content
                    
                    granularity = "paragraph"
                    
                    unit = EvidenceUnit(
                        id=section_title,
                        title=section_title,
                        content=content,
                        granularity=granularity,
                        source="expansion",
                        score=0.5,
                        metadata={"has_summary_evidence": bool(summary_content)},
                    )
                    
                    if query and selected_contents is not None:
                        dynamic_score = self._compute_evidence_score(
                            unit, query, selected_contents, query_entities
                        )
                        unit.score = dynamic_score
                    else:
                        unit.score = 0.7

                    results.append(unit)
                    seen_titles.add(section_title)
                    
            except Exception as e:
                logger.warning(f"图谱扩展查询失败 (title={title}): {e}")
                continue

        return results

    def llm_check_sufficiency(
        self,
        query: str,
        candidates: List[EvidenceUnit],
        path_name: str,
    ) -> Tuple[bool, str]:
        """LLM判断当前检索结果是否足够回答问题"""
        if not candidates:
            return False, "候选集合为空"

        selected, used_tokens = self._select_within_budget(candidates, query, reserve_tokens=300)
        
        context_parts = []
        for i, c in enumerate(selected, 1):
            content = c.content[:600] if c.content else ""
            granularity_label = {"sentence": "句子级", "paragraph": "段落级"}.get(c.granularity, c.granularity)
            context_parts.append(f"[文档{i}({granularity_label})] {content}")

        context = "\n".join(context_parts)

        prompt = f"""你需要判断：当前检索到的文档是否包含足够的信息来回答问题。

问题: {query}

当前检索路径: {path_name}

检索到的文档内容:
{context}

判断规则：
1. 如果文档缺失问题中的关键实体或概念 → 回答"不足够"
2. 如果文档只提供部分信息，无法完整回答 → 回答"不足够"  
3. 如果问题需要多跳推理，但文档缺少中间证据 → 回答"不足够"
4. 只有文档包含完整证据能回答问题时 → 回答"足够"

请严格按以下格式回复（不要添加其他内容）：
判断结果: 不足够
理由: [简述缺少什么信息]

或

判断结果: 足够
理由: [简述文档包含哪些关键信息]"""

        try:
            response = self.llm_client.generate([Message(role="user", content=prompt)])
            if response.success:
                result_text = response.content.strip()
                logger.debug(f"LLM原始回复: {result_text[:300]}")

                is_sufficient = False
                if "判断结果" in result_text:
                    for line in result_text.split("\n"):
                        line_lower = line.lower()
                        if "判断结果" in line_lower or "判断结果" in line:
                            if "不足够" in line:
                                is_sufficient = False
                            elif "足够" in line:
                                is_sufficient = True
                            break
                else:
                    if "不足够" in result_text:
                        is_sufficient = False
                    elif "足够" in result_text:
                        is_sufficient = True

                reason = ""
                if "理由" in result_text:
                    for line in result_text.split("\n"):
                        if "理由" in line:
                            reason = line.split("理由")[-1].strip(": ").strip()
                            break
                if not reason:
                    reason = result_text[:200]

                insufficient_keywords = ["未", "无", "缺少", "缺失", "不足", "没有", "无法", "不能", "不存在"]
                if is_sufficient:
                    for kw in insufficient_keywords:
                        if kw in reason:
                            is_sufficient = False
                            logger.debug(f"理由包含'{kw}'，修正判断为不足够")
                            break

                return is_sufficient, reason
        except Exception as e:
            logger.warning(f"LLM充分性判断失败: {e}")

        return False, "LLM判断失败"

    def run_adaptive_retrieval(
        self,
        query: str,
        variant: Optional[RetrievalVariant] = None,
    ) -> Tuple[List[str], Dict]:
        """运行LLM驱动的自适应检索"""
        variant = variant or RetrievalVariant(name="full", description="完整模型")
        stats = {
            "variant": variant.name,
            "variant_config": {
                "enable_graph_expansion": variant.enable_graph_expansion,
                "enable_scale_rebound": variant.enable_scale_rebound,
                "enable_summary_evidence": variant.enable_summary_evidence,
            },
            "initial_results": 0,
            "routing_path": "",
            "llm_judgments": [],
            "final_results": 0,
            "latency": {},
        }

        t0 = time.time()
        candidates = self.initial_retrieval(query)
        stats["initial_results"] = len(candidates)
        stats["latency"]["initial"] = time.time() - t0

        t0 = time.time()
        reranked = self._rerank(query, candidates)
        stats["latency"]["rerank"] = time.time() - t0

        final_candidates = None
        final_path = None

        t0 = time.time()
        path1_candidates = self.path1_fine_grained(reranked, query)
        stats["latency"]["path1"] = time.time() - t0

        t0 = time.time()
        is_sufficient, reason = self.llm_check_sufficiency(query, path1_candidates, "路径①：细粒度保持")
        stats["latency"]["llm_check1"] = time.time() - t0
        stats["llm_judgments"].append({
            "path": RoutingPath.FINE_GRAINED.value,
            "is_sufficient": is_sufficient,
            "reason": reason,
            "count": len(path1_candidates),
        })
        logger.info(f"路径① LLM判断: {'足够' if is_sufficient else '不足够'} - {reason[:80]}")

        if is_sufficient:
            final_candidates = path1_candidates
            final_path = RoutingPath.FINE_GRAINED
            stats["routing_path"] = final_path.value
        else:
            path2_candidates = path1_candidates
            is_sufficient = False

            if variant.enable_scale_rebound:
                t0 = time.time()
                path2_candidates = self.path2_local_escalation(reranked, query)
                stats["latency"]["path2"] = time.time() - t0

                t0 = time.time()
                is_sufficient, reason = self.llm_check_sufficiency(query, path2_candidates, "路径②：局部尺度回升")
                stats["latency"]["llm_check2"] = time.time() - t0
                stats["llm_judgments"].append({
                    "path": RoutingPath.LOCAL_ESCALATION.value,
                    "is_sufficient": is_sufficient,
                    "reason": reason,
                    "count": len(path2_candidates),
                })
                logger.info(f"路径② LLM判断: {'足够' if is_sufficient else '不足够'} - {reason[:80]}")
            else:
                stats["llm_judgments"].append({
                    "path": RoutingPath.LOCAL_ESCALATION.value,
                    "is_sufficient": False,
                    "reason": "消融实验禁用尺度回升",
                    "count": len(path2_candidates),
                    "skipped": True,
                })

            if is_sufficient:
                final_candidates = path2_candidates
                final_path = RoutingPath.LOCAL_ESCALATION
                stats["routing_path"] = final_path.value
            elif variant.enable_graph_expansion:
                t0 = time.time()
                path3_candidates = self.path3_graph_expansion(
                    reranked,
                    query,
                    enable_summary_evidence=variant.enable_summary_evidence,
                )
                stats["latency"]["path3"] = time.time() - t0

                t0 = time.time()
                is_sufficient, reason = self.llm_check_sufficiency(query, path3_candidates, "路径③：图扩展+多尺度组装")
                stats["latency"]["llm_check3"] = time.time() - t0
                stats["llm_judgments"].append({
                    "path": RoutingPath.GRAPH_EXPANSION.value,
                    "is_sufficient": is_sufficient,
                    "reason": reason,
                    "count": len(path3_candidates),
                })
                logger.info(f"路径③ LLM判断: {'足够' if is_sufficient else '不足够'} - {reason[:80]}")

                final_candidates = path3_candidates
                final_path = RoutingPath.GRAPH_EXPANSION
                stats["routing_path"] = final_path.value
            else:
                final_candidates = path2_candidates
                final_path = RoutingPath.LOCAL_ESCALATION if variant.enable_scale_rebound else RoutingPath.FINE_GRAINED
                stats["routing_path"] = f"{final_path.value}_no_graph_expansion"
                stats["llm_judgments"].append({
                    "path": RoutingPath.GRAPH_EXPANSION.value,
                    "is_sufficient": False,
                    "reason": "消融实验禁用图扩展",
                    "count": len(final_candidates),
                    "skipped": True,
                })

        stats["final_results"] = len(final_candidates)

        t0 = time.time()
        answer = self._generate_answer(query, final_candidates)
        stats["latency"]["generation"] = time.time() - t0
        stats["generated_answer"] = answer

        seen = set()
        retrieved_titles = []
        for c in final_candidates:
            if c.title and c.title not in seen:
                retrieved_titles.append(c.title)
                seen.add(c.title)
        return retrieved_titles, stats

    def _estimate_tokens(self, text: str) -> int:
        """估算文本的token数量（中英文混合）"""
        if not text:
            return 0
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.25)

    def _compute_evidence_score(
        self,
        unit: EvidenceUnit,
        query: str,
        selected_contents: List[str],
        query_entities: Set[str],
    ) -> float:
        """计算证据单元的综合得分
        
        Score(u) = α·SemanticRelevance + β·EvidenceComplement - γ·Redundancy - δ·Cost
        
        Args:
            unit: 证据单元
            query: 查询文本
            selected_contents: 已选中的内容列表（用于计算冗余度）
            query_entities: 查询中的实体集合
            
        Returns:
            综合得分
        """
        semantic_relevance = self._compute_semantic_relevance(unit, query)
        evidence_complement = self._compute_evidence_complement(unit, selected_contents, query_entities)
        redundancy = self._compute_redundancy(unit, selected_contents)
        cost = self._compute_cost(unit)
        
        score = (
            self.score_weights["semantic"] * semantic_relevance
            + self.score_weights["complement"] * evidence_complement
            - self.score_weights["redundancy"] * redundancy
            - self.score_weights["cost"] * cost
        )
        
        return round(score, 4)

    def _compute_semantic_relevance(self, unit: EvidenceUnit, query: str) -> float:
        """计算语义相关性"""
        base_score = unit.score if unit.score > 0 else 0.5
        
        query_words = set(query.lower().split())
        content_words = set((unit.content or "").lower().split())
        overlap = len(query_words & content_words)
        word_boost = min(overlap / max(len(query_words), 1), 1.0) * 0.2
        
        return min(base_score + word_boost, 1.0)

    def _compute_evidence_complement(
        self,
        unit: EvidenceUnit,
        selected_contents: List[str],
        query_entities: Set[str],
    ) -> float:
        """计算证据互补性"""
        if not selected_contents:
            return 1.0
        
        unit_words = set((unit.content or "").lower().split())
        new_words = unit_words.copy()
        for content in selected_contents:
            existing_words = set(content.lower().split())
            new_words = new_words - existing_words
        
        complement = len(new_words) / max(len(unit_words), 1)
        
        if query_entities:
            unit_entities = query_entities & unit_words
            covered_entities = set()
            for content in selected_contents:
                covered_entities.update(query_entities & set(content.lower().split()))
            new_entities = unit_entities - covered_entities
            entity_complement = len(new_entities) / max(len(query_entities), 1)
            complement = 0.6 * complement + 0.4 * entity_complement
        
        return round(min(complement, 1.0), 4)

    def _compute_redundancy(self, unit: EvidenceUnit, selected_contents: List[str]) -> float:
        """计算冗余度"""
        if not selected_contents:
            return 0.0
        
        unit_words = set((unit.content or "").lower().split())
        if not unit_words:
            return 0.0
        
        max_overlap = 0.0
        for content in selected_contents:
            existing_words = set(content.lower().split())
            if not existing_words:
                continue
            overlap = len(unit_words & existing_words) / len(unit_words)
            max_overlap = max(max_overlap, overlap)
        
        return round(max_overlap, 4)

    def _compute_cost(self, unit: EvidenceUnit) -> float:
        """计算代价（归一化的token消耗）"""
        token_count = self._estimate_tokens(unit.content or "")
        cost = token_count / self.token_budget
        return round(min(cost, 1.0), 4)

    def _extract_query_entities(self, query: str) -> Set[str]:
        """提取查询中的实体"""
        entities = set()
        
        for match in re.finditer(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', query):
            entities.add(match.group().lower())
        
        for match in re.finditer(r'"([^"]+)"', query):
            entities.add(match.group(1).lower())
        
        for match in re.finditer(r"'([^']+)'", query):
            entities.add(match.group(1).lower())
        
        return entities

    def _get_summary_evidence(self, title: str, max_hops: int = 2) -> Optional[str]:
        """动态生成摘要型大尺度证据
        
        通过整合语义图谱中一定跳数范围内的实体与关系信息，
        将结构化知识压缩为语义集中的文本表示。
        
        Args:
            title: 文档标题
            max_hops: 整合范围（跳数）
            
        Returns:
            压缩后的摘要文本
        """
        safe_title = title.replace('"', '\\"')
        
        cypher = f'''
        MATCH (start:Section {{title: "{safe_title}"}})
        OPTIONAL MATCH (start)-[:SEMANTIC_LINKS*1..{max_hops}]-(related:Entity)
        WITH start, collect(DISTINCT related.name) AS entities
        OPTIONAL MATCH (start)-[r:SEMANTIC_LINKS*1..{max_hops}]-(related2:Section)
        WITH start, entities, 
             collect(DISTINCT related2.title) AS related_sections
        RETURN entities, related_sections
        LIMIT 1
        '''
        
        try:
            results = self.graph_store.query(cypher)
            if results:
                r = results[0]
                
                entities = r.get("entities", [])
                related_sections = r.get("related_sections", [])
                
                parts = []
                
                if entities:
                    entity_list = [e for e in entities if e][:5]
                    if entity_list:
                        parts.append(f"关键实体: {', '.join(entity_list)}")
                
                if related_sections:
                    sec_list = [s for s in related_sections if s and s != title][:3]
                    if sec_list:
                        parts.append(f"关联章节: {', '.join(sec_list)}")
                
                if parts:
                    return "\n".join(parts)
                    
        except Exception as e:
            logger.debug(f"摘要生成失败(title={title}): {e}")
        
        return None

    def _select_within_budget(
        self, 
        candidates: List[EvidenceUnit], 
        query: str,
        reserve_tokens: int = 500
    ) -> Tuple[List[EvidenceUnit], int]:
        """在token预算范围内选择候选文档
        
        Args:
            candidates: 候选文档列表
            query: 查询文本
            reserve_tokens: 为prompt模板和回答预留的token数
            
        Returns:
            (选中的候选列表, 实际使用的token数)
        """
        query_tokens = self._estimate_tokens(query)
        available_budget = self.token_budget - query_tokens - reserve_tokens
        available_budget = max(available_budget, 500)
        
        selected = []
        total_tokens = 0
        
        for unit in candidates:
            content = unit.content or ""
            unit_tokens = self._estimate_tokens(content)
            
            if total_tokens + unit_tokens <= available_budget:
                selected.append(unit)
                total_tokens += unit_tokens
            elif not selected:
                truncated_content = content[:int(available_budget * 0.7)]
                unit.content = truncated_content
                selected.append(unit)
                total_tokens = self._estimate_tokens(truncated_content)
                break
            else:
                break
        
        return selected, total_tokens

    def _generate_answer(self, query: str, candidates: List[EvidenceUnit]) -> str:
        selected, used_tokens = self._select_within_budget(candidates, query)
        
        context_parts = []
        for i, c in enumerate(selected, 1):
            content = c.content[:800] if c.content else ""
            granularity_label = {"sentence": "句子级", "paragraph": "段落级"}.get(c.granularity, c.granularity)
            context_parts.append(f"[文档{i}({granularity_label})] {content}")

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

    def get_relevant_titles(self, row: pd.Series) -> Set[str]:
        supporting_facts = row.get("supporting_facts", {})
        titles = supporting_facts.get("title", [])
        if isinstance(titles, np.ndarray):
            titles = titles.tolist()
        return set(titles)
    
    def get_relevant_titles_row(self, row_data: Dict) -> Set[str]:
        supporting_facts = row_data.get("supporting_facts", {})
        titles = supporting_facts.get("title", [])
        if isinstance(titles, np.ndarray):
            titles = titles.tolist()
        return set(titles)

    def retrieve(
        self,
        query: str,
        variant: Optional[RetrievalVariant] = None,
    ) -> Tuple[List[str], str, Dict]:
        """纯检索流程 - 只接收query，不接触Ground Truth
        
        这是系统的核心检索接口，严格隔离于评测数据。
        
        Args:
            query: 用户查询
            
        Returns:
            (retrieved_titles, generated_answer, stats)
        """
        retrieved_titles, stats = self.run_adaptive_retrieval(query, variant=variant)
        generated_answer = stats.get("generated_answer", "")
        return retrieved_titles, generated_answer, stats

    def evaluate_retrieval(
        self,
        retrieved_titles: List[str],
        generated_answer: str,
        ground_truth: str,
        relevant_titles: Set[str],
        max_k: int,
    ) -> Dict[str, Any]:
        """评测流程 - 接收检索结果和Ground Truth进行指标计算
        
        此函数在检索完成后调用，严格隔离于检索流程。
        
        Args:
            retrieved_titles: 检索结果
            generated_answer: 生成的答案
            ground_truth: 标准答案
            relevant_titles: 相关文档标题集合
            max_k: 最大K值
            
        Returns:
            评测指标字典
        """
        retrieved_list = retrieved_titles[:max_k]
        result = self.metrics_calculator.compute(retrieved_list, list(relevant_titles))

        retrieved_set = set(retrieved_titles)
        title_recall = len(retrieved_set & relevant_titles) / len(relevant_titles) if relevant_titles else 0
        title_precision = len(retrieved_set & relevant_titles) / len(retrieved_set) if retrieved_set else 0

        gen_result = self.generation_metrics.compute(
            predicted=generated_answer,
            ground_truth=ground_truth,
            compute_semantic=True,
            embedding_client=self.embedding_client,
        )

        return {
            "metrics": result.to_dict(),
            "generation_metrics": gen_result.to_dict(),
            "title_recall": title_recall,
            "title_precision": title_precision,
        }

    def evaluate_single_query(
        self,
        query: str,
        ground_truth: str,
        relevant_titles: Set[str],
        max_k: int,
        variant: Optional[RetrievalVariant] = None,
    ) -> Dict[str, Any]:
        """完整评测流程 - 先检索后评测，严格数据隔离
        
        流程：
        1. 调用 retrieve() 进行检索（不接触Ground Truth）
        2. 调用 evaluate_retrieval() 进行评测（使用Ground Truth）
        
        Args:
            query: 用户查询
            ground_truth: 标准答案
            relevant_titles: 相关文档标题集合
            max_k: 最大K值
            
        Returns:
            完整结果字典
        """
        # Step 1: 检索流程（严格隔离，不接触Ground Truth）
        retrieved_titles, generated_answer, stats = self.retrieve(query, variant=variant)
        
        # Step 2: 评测流程（在检索完成后引入Ground Truth）
        eval_results = self.evaluate_retrieval(
            retrieved_titles=retrieved_titles,
            generated_answer=generated_answer,
            ground_truth=ground_truth,
            relevant_titles=relevant_titles,
            max_k=max_k,
        )

        return {
            "retrieved_titles": retrieved_titles[:20],
            "relevant_titles": list(relevant_titles),
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "metrics": eval_results["metrics"],
            "generation_metrics": eval_results["generation_metrics"],
            "title_recall": eval_results["title_recall"],
            "title_precision": eval_results["title_precision"],
            "stats": stats,
        }

    def _save_checkpoint(
        self,
        processed_indices: List[int],
        all_results: List[Dict],
        aggregated_metrics: Dict,
        sample_size: Optional[int],
    ) -> None:
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
        checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        if not checkpoint_path.exists():
            return None
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            if "processed_indices" in checkpoint:
                checkpoint["processed_indices"] = [int(idx) for idx in checkpoint["processed_indices"]]
            logger.info(f"从检查点恢复: 已处理 {len(checkpoint['processed_indices'])} 条")
            return checkpoint
        except Exception as e:
            logger.warning(f"加载检查点失败: {e}")
            return None

    def _clear_checkpoint(self) -> None:
        checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("检查点已清除")

    def _get_experiment_variants(self) -> List[RetrievalVariant]:
        return [
            RetrievalVariant(
                name="full",
                description="主实验：图扩展 + 尺度回升 + 摘要证据",
            ),
            RetrievalVariant(
                name="no_graph_expansion",
                description="消融：去除图扩展",
                enable_graph_expansion=False,
            ),
            RetrievalVariant(
                name="no_scale_rebound",
                description="消融：去除尺度回升",
                enable_scale_rebound=False,
            ),
            RetrievalVariant(
                name="no_summary_evidence",
                description="消融：去除摘要证据",
                enable_summary_evidence=False,
            ),
        ]

    def _empty_aggregated_metrics(self) -> Dict[str, Any]:
        return {
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
            "total_latency": [],
            "path_distribution": {
                RoutingPath.FINE_GRAINED.value: 0,
                RoutingPath.LOCAL_ESCALATION.value: 0,
                RoutingPath.GRAPH_EXPANSION.value: 0,
            },
        }

    def _single_metrics_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        metrics = result["metrics"]
        gen_metrics = result["generation_metrics"]
        stats = result.get("stats", {})
        return {
            "recall_at_k": metrics["recall_at_k"].get(self.k4, 0),
            "precision_at_k": metrics["precision_at_k"].get(self.k4, 0),
            "mrr": metrics["mrr"],
            "ndcg": metrics["ndcg"],
            "map_score": metrics["map_score"],
            "hit_rate": metrics["hit_rate"],
            "title_recall": result["title_recall"],
            "title_precision": result["title_precision"],
            "exact_match": gen_metrics["exact_match"],
            "f1_score": gen_metrics["f1_score"],
            "semantic_similarity": gen_metrics["semantic_similarity"],
            "total_latency": sum(stats.get("latency", {}).values()),
            "routing_path": stats.get("routing_path", ""),
        }

    def _append_metrics(self, aggregated_metrics: Dict[str, Any], single_metrics: Dict[str, Any]) -> None:
        for key in [
            "recall_at_k",
            "precision_at_k",
            "mrr",
            "ndcg",
            "map_score",
            "hit_rate",
            "title_recall",
            "title_precision",
            "exact_match",
            "f1_score",
            "semantic_similarity",
            "total_latency",
        ]:
            aggregated_metrics[key].append(single_metrics[key])

        routing_path = single_metrics["routing_path"]
        if routing_path:
            aggregated_metrics["path_distribution"].setdefault(routing_path, 0)
            aggregated_metrics["path_distribution"][routing_path] += 1

    def _finalize_metrics(self, aggregated_metrics: Dict[str, Any]) -> Dict[str, Any]:
        total_samples = len(aggregated_metrics["recall_at_k"])
        return {
            "recall_at_k": np.mean(aggregated_metrics["recall_at_k"]) if total_samples > 0 else 0,
            "precision_at_k": np.mean(aggregated_metrics["precision_at_k"]) if total_samples > 0 else 0,
            "mrr": np.mean(aggregated_metrics["mrr"]) if total_samples > 0 else 0,
            "ndcg": np.mean(aggregated_metrics["ndcg"]) if total_samples > 0 else 0,
            "map_score": np.mean(aggregated_metrics["map_score"]) if total_samples > 0 else 0,
            "hit_rate": np.mean(aggregated_metrics["hit_rate"]) if total_samples > 0 else 0,
            "title_recall": np.mean(aggregated_metrics["title_recall"]) if total_samples > 0 else 0,
            "title_precision": np.mean(aggregated_metrics["title_precision"]) if total_samples > 0 else 0,
            "exact_match": np.mean(aggregated_metrics["exact_match"]) if total_samples > 0 else 0,
            "f1_score": np.mean(aggregated_metrics["f1_score"]) if total_samples > 0 else 0,
            "semantic_similarity": float(np.nanmean(aggregated_metrics["semantic_similarity"])) if total_samples > 0 else 0,
            "avg_total_latency": np.mean(aggregated_metrics["total_latency"]) if total_samples > 0 else 0,
            "path_distribution": aggregated_metrics["path_distribution"],
        }

    def run_experiment(
        self,
        sample_size: Optional[int] = None,
        save_details: bool = True,
        resume: bool = True,
    ) -> Dict[str, Any]:
        logger.info("开始运行LLM驱动的自适应多尺度检索实验...")

        self.load_all()

        test_data = self.test_data
        if sample_size:
            test_data = test_data.sample(n=sample_size, random_state=42)
            logger.info(f"采样 {sample_size} 条数据进行测试")

        max_k = self.k4
        test_indices = test_data.index.tolist()
        variants = self._get_experiment_variants()
        variant_map = {variant.name: variant for variant in variants}

        processed_indices: List[int] = []
        all_results: Dict[str, List[Dict]] = {variant.name: [] for variant in variants}
        aggregated_metrics = {variant.name: self._empty_aggregated_metrics() for variant in variants}

        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                checkpoint_variants = set(checkpoint.get("aggregated_metrics", {}).keys())
                if checkpoint.get("sample_size") == sample_size and checkpoint_variants == set(variant_map.keys()):
                    processed_indices = checkpoint["processed_indices"]
                    all_results = checkpoint["all_results"]
                    aggregated_metrics = checkpoint["aggregated_metrics"]
                    logger.info(f"从检查点恢复成功，跳过 {len(processed_indices)} 条已处理数据")

        remaining_indices = [idx for idx in test_indices if idx not in processed_indices]

        if not remaining_indices:
            logger.info("所有数据已处理完成")
        else:
            results_lock = threading.Lock()
            checkpoint_lock = threading.Lock()
            max_retries = 3
            retry_delay = 2.0

            def process_single_query(idx: int) -> Optional[Tuple[int, Dict[str, Dict], Dict[str, Dict[str, Any]]]]:
                try:
                    row_data = test_data.loc[idx].to_dict()
                except Exception as e:
                    logger.error(f"读取数据失败 (idx={idx}): {e}")
                    return None
                
                query = row_data.get("question", "")
                ground_truth = row_data.get("answer", "")
                relevant_titles = self.get_relevant_titles_row(row_data)
                query_metrics: Dict[str, Dict] = {}
                query_details: Dict[str, Dict[str, Any]] = {}

                for variant in variants:
                    for attempt in range(max_retries):
                        try:
                            result = self.evaluate_single_query(
                                query,
                                ground_truth,
                                relevant_titles,
                                max_k,
                                variant=variant,
                            )

                            gen_metrics = result["generation_metrics"]
                            generated_answer = result.get("generated_answer", "")

                            api_failed = False
                            if gen_metrics.get("semantic_similarity") == 0.0 and ground_truth:
                                api_failed = True
                                logger.warning(f"查询 {idx}/{variant.name} Embedding失败(尝试 {attempt+1}/{max_retries})")
                            if not generated_answer and ground_truth:
                                api_failed = True
                                logger.warning(f"查询 {idx}/{variant.name} LLM生成失败(尝试 {attempt+1}/{max_retries})")

                            if api_failed and attempt < max_retries - 1:
                                time.sleep(retry_delay * (attempt + 1))
                                continue

                            query_metrics[variant.name] = self._single_metrics_from_result(result)

                            if save_details:
                                metrics = result["metrics"]
                                stats = result.get("stats", {})
                                query_details[variant.name] = {
                                    "variant": variant.name,
                                    "variant_description": variant.description,
                                    "id": row_data.get("id", idx),
                                    "question": query,
                                    "answer": ground_truth,
                                    "generated_answer": result["generated_answer"],
                                    "relevant_titles": list(relevant_titles),
                                    "retrieved_titles": result["retrieved_titles"],
                                    "metrics": metrics,
                                    "generation_metrics": gen_metrics,
                                    "title_recall": result["title_recall"],
                                    "title_precision": result["title_precision"],
                                    "stats": stats,
                                }

                            break

                        except Exception as e:
                            logger.error(f"处理查询失败 (idx={idx}, variant={variant.name}, 尝试 {attempt+1}/{max_retries}): {e}")
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay * (attempt + 1))
                                continue
                            return None

                return (idx, query_metrics, query_details)

            pbar = tqdm(remaining_indices, desc="评估进度", initial=len(processed_indices), total=len(test_indices))

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(process_single_query, idx): idx for idx in remaining_indices}

                for future in as_completed(futures):
                    result = future.result()
                    if result is None:
                        pbar.update(1)
                        continue

                    idx, query_metrics, query_details = result

                    with results_lock:
                        for variant_name, single_metrics in query_metrics.items():
                            self._append_metrics(aggregated_metrics[variant_name], single_metrics)

                        for variant_name, detail_result in query_details.items():
                            all_results[variant_name].append(detail_result)

                        processed_indices.append(idx)

                    with checkpoint_lock:
                        if len(processed_indices) % self.checkpoint_interval == 0:
                            self._save_checkpoint(processed_indices, all_results, aggregated_metrics, sample_size)

                    pbar.update(1)

            pbar.close()

        final_metrics = {
            variant.name: self._finalize_metrics(aggregated_metrics[variant.name])
            for variant in variants
        }

        experiment_result = {
            "experiment_name": "llm_adaptive_multiscale",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "test_data_path": self.test_data_path,
                "k1": self.k1,
                "k2": self.k2,
                "k4": self.k4,
                "max_hops": self.max_hops,
                "max_degree": self.max_degree,
                "token_budget": self.token_budget,
                "sample_size": sample_size or len(test_data),
                "total_test_samples": len(test_data),
                "max_workers": self.max_workers,
                "variants": {
                    variant.name: {
                        "description": variant.description,
                        "enable_graph_expansion": variant.enable_graph_expansion,
                        "enable_scale_rebound": variant.enable_scale_rebound,
                        "enable_summary_evidence": variant.enable_summary_evidence,
                    }
                    for variant in variants
                },
                "graph_stats": {
                    "total_nodes": self.total_graph_nodes,
                    "total_edges": self.total_graph_edges,
                },
            },
            "metrics": final_metrics,
            "details": all_results if save_details else None,
        }

        self._clear_checkpoint()
        self._save_and_print_result(experiment_result, save_details)

        return experiment_result

    def _save_and_print_result(self, results: Dict[str, Any], save_details: bool = True) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary_path = self.output_dir / f"experiment_{timestamp}.json"
        summary = {k: v for k, v in results.items() if k != "details"}
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"实验摘要已保存: {summary_path}")

        if save_details and results.get("details"):
            details_path = self.output_dir / f"details_{timestamp}.json"
            with open(details_path, "w", encoding="utf-8") as f:
                json.dump(results["details"], f, ensure_ascii=False, indent=2)
            logger.info(f"详细结果已保存: {details_path}")

        self._print_result(results)

    def _print_result(self, results: Dict[str, Any]) -> None:
        metrics = results["metrics"]
        config = results["config"]
        variants = config.get("variants", {})

        print("\n" + "=" * 80)
        print("实验六改进版：主实验 + 消融实验完成")
        print("=" * 80)

        print(f"\n【配置参数】")
        print(f"  向量检索 K1: {config['k1']}")
        print(f"  图谱检索 K2: {config['k2']}")
        print(f"  重排序 K4: {config['k4']}")
        print(f"  最大跳数: {config['max_hops']}")
        print(f"  并发线程数: {config.get('max_workers', 4)}")

        print(f"\n【主实验与消融指标】")
        header = (
            f"  {'Variant':<22} "
            f"Recall@{config['k4']:<3} "
            f"Precision@{config['k4']:<3} "
            f"MRR     NDCG    MAP     F1      Semantic  Latency(s)"
        )
        print(header)
        for variant_name, variant_metrics in metrics.items():
            print(
                f"  {variant_name:<22} "
                f"{variant_metrics['recall_at_k']:.4f}     "
                f"{variant_metrics['precision_at_k']:.4f}        "
                f"{variant_metrics['mrr']:.4f}  "
                f"{variant_metrics['ndcg']:.4f}  "
                f"{variant_metrics['map_score']:.4f}  "
                f"{variant_metrics['f1_score']:.4f}  "
                f"{variant_metrics['semantic_similarity']:.4f}    "
                f"{variant_metrics['avg_total_latency']:.4f}"
            )

        full_metrics = metrics.get("full")
        if full_metrics:
            print(f"\n【相对主实验变化】")
            for variant_name, variant_metrics in metrics.items():
                if variant_name == "full":
                    continue
                delta_recall = variant_metrics["recall_at_k"] - full_metrics["recall_at_k"]
                delta_f1 = variant_metrics["f1_score"] - full_metrics["f1_score"]
                description = variants.get(variant_name, {}).get("description", variant_name)
                print(f"  {variant_name}: ΔRecall={delta_recall:+.4f}, ΔF1={delta_f1:+.4f} ({description})")

        print(f"\n【路由路径分布】")
        for variant_name, variant_metrics in metrics.items():
            print(f"  {variant_name}:")
            path_dist = variant_metrics["path_distribution"]
            total_path = sum(path_dist.values()) or 1
            for path_name, count in path_dist.items():
                pct = count / total_path * 100
                print(f"    {path_name}: {count} ({pct:.1f}%)")

        print("\n" + "=" * 80)


def main():
    test_data_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/validation-00000-of-00001.parquet"
    documents_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/valid_title_sentence.json"
    vector_store_paths = {
        "paragraph": "e:/Code_Personal/Subject/test02/data/hotpotqa/vector_stores/valid_title_sentence",
        "sentence": "e:/Code_Personal/Subject/test02/data/hotpotqa/vector_stores/single_sentence",
    }
    local_graph_path = "e:/Code_Personal/Subject/test02/data/hotpotqa/local_graph.json"
    output_dir = "e:/Code_Personal/Subject/test02/experiments/exp6b_adaptive_multiscale"

    experiment = LLMAdaptiveRetrieval(
        test_data_path=test_data_path,
        output_dir=output_dir,
        documents_path=documents_path,
        vector_store_paths=vector_store_paths,
        local_graph_path=local_graph_path,
        use_neo4j=True,
        k1=10,
        k2=20,
        k4=7,
        max_hops=2,
        max_degree=10,
        token_budget=3600,
        checkpoint_interval=10,
        max_workers=50,
    )

    results = experiment.run_experiment(
        sample_size=50,
        save_details=True,
        resume=True,
    )


if __name__ == "__main__":
    main()
