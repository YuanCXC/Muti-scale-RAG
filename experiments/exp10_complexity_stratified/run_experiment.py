# -*- coding: utf-8 -*-
"""实验十：按问题复杂度分层分析实验

基于问题复杂度（低/中/高三个层级）执行自适应策略，验证在高复杂度问题上实施分层分析策略可获得更显著的性能收益。

实验设计：
1. 实现问题复杂度分类器（基于支持文档数量、问题长度、实体数量等特征）
2. 将问题分为低/中/高三个复杂度层级
3. 对每个层级分别运行实验六的自适应策略
4. 对比分析不同复杂度问题上的性能差异

评估指标：
- 精确匹配率(EM)
- F1分数
- 平均上下文长度(Avg Context Len)
- 路由触发比例
- Recall, Precision, MRR, NDCG, MAP

自适应策略流程：
1. 向量检索 (FAISS, k1=10)
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


class ComplexityClassifier:
    """问题复杂度分类器
    
    基于多个特征计算问题复杂度：
    1. 支持文档数量（supporting_facts中的title数量）
    2. 问题长度（字符数）
    3. 问题中的实体数量（使用LLM提取）
    4. 问题类型特征（是否包含比较词、推理词等）
    """
    
    def __init__(self, llm_client: Optional[DeepSeekClient] = None):
        self.llm_client = llm_client
        
        self.comparison_keywords = [
            "same", "different", "similar", "compare", "comparison", "versus", "vs",
            "better", "worse", "more", "less", "larger", "smaller", "older", "younger",
            "哪个", "哪个更", "比较", "相同", "不同", "相似", "区别"
        ]
        
        self.inference_keywords = [
            "why", "how", "because", "reason", "cause", "result", "effect",
            "为什么", "如何", "原因", "导致", "结果", "影响"
        ]
        
        self.multi_hop_keywords = [
            "who", "what", "where", "when", "which", "whose",
            "谁", "什么", "哪里", "何时", "哪个", "谁的"
        ]
    
    def extract_entities(self, query: str) -> List[str]:
        """使用LLM提取问题中的实体"""
        if not self.llm_client:
            return []
        
        prompt = f"""请从以下问题中提取所有关键实体和名词短语。
只返回实体列表，用逗号分隔，不要其他解释。

问题: {query}

实体:"""
        
        try:
            response = self.llm_client.generate([Message(role="user", content=prompt)])
            if response.success:
                entities = [e.strip() for e in response.content.split(",") if e.strip()]
                return entities[:15]
        except Exception as e:
            logger.warning(f"实体提取失败: {e}")
        
        return []
    
    def count_supporting_docs(self, supporting_facts: Dict) -> int:
        """统计支持文档数量"""
        titles = supporting_facts.get("title", [])
        if isinstance(titles, np.ndarray):
            titles = titles.tolist()
        return len(set(titles)) if titles else 0
    
    def has_comparison_pattern(self, query: str) -> bool:
        """检测是否包含比较模式"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.comparison_keywords)
    
    def has_inference_pattern(self, query: str) -> bool:
        """检测是否包含推理模式"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.inference_keywords)
    
    def calculate_complexity_score(
        self,
        query: str,
        supporting_facts: Dict,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """计算问题复杂度分数
        
        Args:
            query: 问题文本
            supporting_facts: 支持事实
            use_llm: 是否使用LLM提取实体
            
        Returns:
            复杂度分数和特征字典
        """
        features = {}
        
        features["supporting_docs"] = self.count_supporting_docs(supporting_facts)
        
        features["query_length"] = len(query)
        
        if use_llm and self.llm_client:
            entities = self.extract_entities(query)
            features["entity_count"] = len(entities)
            features["entities"] = entities
        else:
            words = query.split()
            features["entity_count"] = len([w for w in words if len(w) > 3])
            features["entities"] = []
        
        features["has_comparison"] = self.has_comparison_pattern(query)
        features["has_inference"] = self.has_inference_pattern(query)
        
        score = 0.0
        
        if features["supporting_docs"] == 1:
            score += 0.1
        elif features["supporting_docs"] == 2:
            score += 0.3
        elif features["supporting_docs"] >= 3:
            score += 0.5
        
        if features["query_length"] < 50:
            score += 0.1
        elif features["query_length"] < 100:
            score += 0.2
        else:
            score += 0.3
        
        if features["entity_count"] <= 2:
            score += 0.1
        elif features["entity_count"] <= 4:
            score += 0.2
        else:
            score += 0.3
        
        if features["has_comparison"]:
            score += 0.2
        if features["has_inference"]:
            score += 0.2
        
        features["complexity_score"] = min(score, 1.0)
        
        return features
    
    def classify_complexity(self, complexity_score: float) -> str:
        """根据复杂度分数分类
        
        Args:
            complexity_score: 复杂度分数 (0-1)
            
        Returns:
            复杂度层级 (low/medium/high)
        """
        if complexity_score < 0.4:
            return "low"
        elif complexity_score < 0.7:
            return "medium"
        else:
            return "high"


class ComplexityStratifiedExperiment:
    """按问题复杂度分层实验类
    
    按问题复杂度（低/中/高）分层，分别运行自适应策略实验。
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
        k1: int = 10,
        k2: int = 20,
        k4: int = 7,
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
        
        self.complexity_classifier: Optional[ComplexityClassifier] = None
    
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
        
        self.complexity_classifier = ComplexityClassifier(llm_client=self.llm_client)
    
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
        """Step 1: 向量检索"""
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
        """Step 2: 关键词提取 (LLM 提取实体)"""
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
        """Step 3: 关键词检索 (图谱)"""
        if not keywords:
            return []
        
        results = []
        seen_titles = set()
        
        def cypher_str_value(val):
            if not isinstance(val, str):
                return '""'
            out = val.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{out}"'
        
        def extract_node_property(node, prop_name, default=""):
            """安全提取节点属性"""
            if node is None:
                return default
            if isinstance(node, dict):
                return node.get(prop_name, default)
            try:
                val = node.get(prop_name)
                return val if val is not None else default
            except (AttributeError, TypeError):
                return default
        
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
                title = extract_node_property(node, "title", "").strip('"')
                
                if title and title not in seen_titles:
                    content = extract_node_property(node, "sentence_total", "") or self.title_to_content.get(title, "")
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
                    title = extract_node_property(node, "title", "").strip('"')
                    
                    if title and title not in seen_titles:
                        content = extract_node_property(node, "sentence_total", "") or self.title_to_content.get(title, "")
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
        """Step 4: 知识图谱扩展 (语义关联)"""
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
        """Step 6: 父切片映射（句子到段落映射）"""
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
        """Step 7: LLM 证据强度打分"""
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
- 0.8–1.0: Provides direct, key, and high-value evidential support; even if the complete answer may still require supplementation from other context chunks, this chunk itself already has high support value

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
        """Step 8: 低分文档更新"""
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
        """智能去重添加到目标字典"""
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
        """运行完整的自适应检索流程"""
        stats = {
            "vector_results": 0,
            "keywords": [],
            "graph_results": 0,
            "expanded_results": 0,
            "reranked_results": 0,
            "final_results": 0,
            "latency": {},
            "routing_triggered": {
                "vector": False,
                "keywords": False,
                "graph": False,
                "expansion": False,
                "rerank": False,
                "scoring": False,
                "update": False,
            },
        }
        
        title_sentence_dict = []
        
        start_time = time.time()
        vector_results = self.step1_vector_retrieval(query)
        stats["latency"]["vector"] = time.time() - start_time
        stats["vector_results"] = len(vector_results)
        stats["routing_triggered"]["vector"] = len(vector_results) > 0
        
        self._add_to_dict_with_dedup(vector_results, title_sentence_dict, dedup_type="vector")
        
        start_time = time.time()
        keywords = self.step2_extract_keywords(query)
        stats["latency"]["keywords"] = time.time() - start_time
        stats["keywords"] = keywords
        stats["routing_triggered"]["keywords"] = len(keywords) > 0
        
        start_time = time.time()
        graph_results = self.step3_keyword_retrieval(keywords)
        stats["latency"]["graph"] = time.time() - start_time
        stats["graph_results"] = len(graph_results)
        stats["routing_triggered"]["graph"] = len(graph_results) > 0
        
        self._add_to_dict_with_dedup(graph_results, title_sentence_dict, dedup_type="keywords")
        
        title_list = list(set([item["title"] for item in title_sentence_dict if "title" in item]))
        
        start_time = time.time()
        expanded_results = self.step4_graph_expansion(title_list)
        stats["latency"]["expansion"] = time.time() - start_time
        stats["expanded_results"] = len(expanded_results)
        stats["routing_triggered"]["expansion"] = len(expanded_results) > 0
        
        self._add_to_dict_with_dedup(expanded_results, title_sentence_dict, dedup_type="keywords")
        
        start_time = time.time()
        reranked = self.step5_rerank(query, title_sentence_dict)
        stats["latency"]["rerank"] = time.time() - start_time
        stats["reranked_results"] = len(reranked)
        stats["routing_triggered"]["rerank"] = len(reranked) > 0
        
        start_time = time.time()
        all_titles, mapped = self.step6_parent_chunk_mapping(reranked)
        stats["latency"]["mapping"] = time.time() - start_time
        
        start_time = time.time()
        scored = self.step7_llm_scoring(query, mapped)
        stats["latency"]["scoring"] = time.time() - start_time
        stats["routing_triggered"]["scoring"] = True
        
        start_time = time.time()
        updated = self.step8_update_low_score_docs(query, scored)
        stats["latency"]["update"] = time.time() - start_time
        stats["routing_triggered"]["update"] = any(c.get("updated", False) for c in updated)
        
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
    
    def calculate_context_length(self, candidates: List[Dict]) -> int:
        """计算上下文总长度"""
        total_length = 0
        for c in candidates:
            content = c.get("content", "")
            total_length += len(content)
        return total_length
    
    def evaluate_single_query(
        self,
        query: str,
        ground_truth: str,
        relevant_titles: Set[str],
        max_k: int,
    ) -> Dict[str, Any]:
        """评估单个查询"""
        retrieved_titles, stats, candidates = self.run_adaptive_retrieval(query)
        
        retrieved_list = retrieved_titles[:max_k]
        
        result = self.metrics_calculator.compute(retrieved_list, list(relevant_titles))
        
        retrieved_set = set(retrieved_titles)
        title_recall = len(retrieved_set & relevant_titles) / len(relevant_titles) if relevant_titles else 0
        title_precision = len(retrieved_set & relevant_titles) / len(retrieved_set) if retrieved_set else 0
        
        context_length = self.calculate_context_length(candidates)
        
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
            "context_length": context_length,
            "stats": stats,
        }
    
    def _save_checkpoint(
        self,
        complexity_level: str,
        processed_indices: List[int],
        all_results: List[Dict],
        aggregated_metrics: Dict,
        sample_size: Optional[int],
    ) -> None:
        """保存检查点"""
        checkpoint = {
            "complexity_level": complexity_level,
            "processed_indices": processed_indices,
            "all_results": all_results,
            "aggregated_metrics": aggregated_metrics,
            "sample_size": sample_size,
            "timestamp": datetime.now().isoformat(),
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_{complexity_level}.json"
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False)
        
        logger.info(f"检查点已保存 ({complexity_level}): 已处理 {len(processed_indices)} 条")
    
    def _load_checkpoint(self, complexity_level: str) -> Optional[Dict]:
        """加载检查点"""
        checkpoint_path = self.output_dir / f"checkpoint_{complexity_level}.json"
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            
            if "processed_indices" in checkpoint:
                checkpoint["processed_indices"] = [int(idx) for idx in checkpoint["processed_indices"]]
            
            logger.info(f"从检查点恢复 ({complexity_level}): 已处理 {len(checkpoint['processed_indices'])} 条")
            return checkpoint
        except Exception as e:
            logger.warning(f"加载检查点失败: {e}")
            return None
    
    def _clear_checkpoint(self, complexity_level: str) -> None:
        """清除检查点"""
        checkpoint_path = self.output_dir / f"checkpoint_{complexity_level}.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"检查点已清除 ({complexity_level})")
    
    def run_experiment_for_complexity(
        self,
        complexity_level: str,
        test_data: pd.DataFrame,
        sample_size: Optional[int] = None,
        save_details: bool = True,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """运行特定复杂度层级问题的实验"""
        logger.info(f"开始运行 {complexity_level} 复杂度问题实验...")
        
        if sample_size and len(test_data) > sample_size:
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
            "context_length": [],
            "vector_latency": [],
            "graph_latency": [],
            "total_latency": [],
            "routing_triggered": {
                "vector": [],
                "keywords": [],
                "graph": [],
                "expansion": [],
                "rerank": [],
                "scoring": [],
                "update": [],
            },
        }
        
        if resume:
            checkpoint = self._load_checkpoint(complexity_level)
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
            logger.info(f"所有 {complexity_level} 复杂度数据已处理完成")
        else:
            pbar = tqdm(remaining_indices, desc=f"评估进度 ({complexity_level})", initial=len(processed_indices), total=len(test_indices))
            
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
                    
                    aggregated_metrics["context_length"].append(result["context_length"])
                    
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
                    
                    routing = stats.get("routing_triggered", {})
                    for key in aggregated_metrics["routing_triggered"].keys():
                        aggregated_metrics["routing_triggered"][key].append(
                            routing.get(key, False)
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
                            "context_length": result["context_length"],
                            "stats": stats,
                        })
                    
                    processed_indices.append(idx)
                    
                    if len(processed_indices) % self.checkpoint_interval == 0:
                        self._save_checkpoint(
                            complexity_level, processed_indices, all_results, aggregated_metrics, sample_size
                        )
                        
                except Exception as e:
                    logger.error(f"处理查询失败 (idx={idx}, complexity={complexity_level}): {e}")
                    continue
                
                pbar.update(1)
            
            pbar.close()
        
        final_metrics = {
            "recall_at_k": np.mean(aggregated_metrics["recall_at_k"]),
            "precision_at_k": np.mean(aggregated_metrics["precision_at_k"]),
            "mrr": np.mean(aggregated_metrics["mrr"]),
            "ndcg": np.mean(aggregated_metrics["ndcg"]),
            "map_score": np.mean(aggregated_metrics["map_score"]),
            "hit_rate": np.mean(aggregated_metrics["hit_rate"]),
            "title_recall": np.mean(aggregated_metrics["title_recall"]),
            "title_precision": np.mean(aggregated_metrics["title_precision"]),
            "exact_match": np.mean(aggregated_metrics["exact_match"]),
            "f1_score": np.mean(aggregated_metrics["f1_score"]),
            "semantic_similarity": np.nanmean(aggregated_metrics["semantic_similarity"]),
            "avg_context_length": np.mean(aggregated_metrics["context_length"]),
            "avg_vector_latency": np.mean(aggregated_metrics["vector_latency"]),
            "avg_graph_latency": np.mean(aggregated_metrics["graph_latency"]),
            "avg_total_latency": np.mean(aggregated_metrics["total_latency"]),
            "routing_trigger_ratio": {
                key: np.mean(values) for key, values in aggregated_metrics["routing_triggered"].items()
            },
        }
        
        experiment_result = {
            "experiment_name": f"complexity_stratified_{complexity_level}",
            "complexity_level": complexity_level,
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
        
        self._clear_checkpoint(complexity_level)
        
        return experiment_result
    
    def run_experiment(
        self,
        sample_size: Optional[int] = None,
        save_details: bool = True,
        resume: bool = True,
        balance_samples: bool = False,
    ) -> Dict[str, Any]:
        """运行完整的分层实验
        
        Args:
            sample_size: 总采样数量
            save_details: 是否保存详细结果
            resume: 是否从检查点恢复
            balance_samples: 是否均衡各复杂度层级的样本数量
        """
        logger.info("开始运行按问题复杂度分层实验...")
        
        self.load_graph_store()
        self.load_documents()
        self.load_vector_stores()
        self.load_embedding_client()
        self.load_llm_client()
        self.load_reranker()
        self.load_test_data()
        
        test_data = self.test_data
        if sample_size:
            test_data = test_data.sample(n=sample_size, random_state=42)
            logger.info(f"采样 {sample_size} 条数据进行测试")
        
        logger.info("开始对问题进行复杂度分类...")
        complexity_features_list = []
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="复杂度分类"):
            query = row["question"]
            supporting_facts = row.get("supporting_facts", {})
            
            features = self.complexity_classifier.calculate_complexity_score(
                query, supporting_facts, use_llm=False
            )
            
            complexity_level = self.complexity_classifier.classify_complexity(
                features["complexity_score"]
            )
            
            features["complexity_level"] = complexity_level
            features["index"] = idx
            complexity_features_list.append(features)
        
        complexity_df = pd.DataFrame(complexity_features_list)
        complexity_df.to_csv(self.output_dir / "complexity_classification.csv", index=False)
        
        logger.info("\n复杂度分布统计 (均衡前):")
        for level in ["low", "medium", "high"]:
            count = len(complexity_df[complexity_df["complexity_level"] == level])
            logger.info(f"  {level}: {count} 条 ({count/len(test_data)*100:.1f}%)")
        
        if balance_samples:
            level_counts = {
                level: len(complexity_df[complexity_df["complexity_level"] == level])
                for level in ["low", "medium", "high"]
            }
            min_count = min(c for c in level_counts.values() if c > 0)
            
            logger.info(f"\n均衡采样: 每个层级采样 {min_count} 条")
            
            balanced_indices = []
            for level in ["low", "medium", "high"]:
                level_indices = complexity_df[complexity_df["complexity_level"] == level]["index"].tolist()
                if len(level_indices) > min_count:
                    np.random.seed(42)
                    level_indices = list(np.random.choice(level_indices, size=min_count, replace=False))
                balanced_indices.extend(level_indices)
            
            complexity_df = complexity_df[complexity_df["index"].isin(balanced_indices)]
            
            logger.info("\n复杂度分布统计 (均衡后):")
            for level in ["low", "medium", "high"]:
                count = len(complexity_df[complexity_df["complexity_level"] == level])
                logger.info(f"  {level}: {count} 条")
        
        results_by_complexity = {}
        
        for level in ["low", "medium", "high"]:
            level_indices = complexity_df[complexity_df["complexity_level"] == level]["index"].tolist()
            level_data = test_data.loc[level_indices]
            
            if len(level_data) == 0:
                logger.warning(f"{level} 复杂度层级没有数据，跳过")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"开始处理 {level} 复杂度层级: {len(level_data)} 条数据")
            logger.info(f"{'='*60}")
            
            level_results = self.run_experiment_for_complexity(
                complexity_level=level,
                test_data=level_data,
                sample_size=None,
                save_details=save_details,
                resume=resume,
            )
            
            results_by_complexity[level] = level_results
        
        self._save_comparison_results(results_by_complexity, save_details)
        
        return results_by_complexity
    
    def _save_comparison_results(
        self,
        results_by_complexity: Dict[str, Dict],
        save_details: bool,
    ) -> None:
        """保存对比结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary_data = []
        for level in ["low", "medium", "high"]:
            if level in results_by_complexity:
                result = results_by_complexity[level]
                metrics = result["metrics"]
                
                summary_data.append({
                    "Complexity_Level": level,
                    "Sample_Size": result["config"]["total_test_samples"],
                    "EM": metrics["exact_match"],
                    "F1": metrics["f1_score"],
                    "Avg_Context_Len": metrics["avg_context_length"],
                    "Recall@K": metrics["recall_at_k"],
                    "Precision@K": metrics["precision_at_k"],
                    "MRR": metrics["mrr"],
                    "NDCG": metrics["ndcg"],
                    "MAP": metrics["map_score"],
                    "Hit_Rate": metrics["hit_rate"],
                    "Title_Recall": metrics["title_recall"],
                    "Title_Precision": metrics["title_precision"],
                    "Semantic_Sim": metrics["semantic_similarity"],
                    "Vector_Latency": metrics["avg_vector_latency"],
                    "Graph_Latency": metrics["avg_graph_latency"],
                    "Total_Latency": metrics["avg_total_latency"],
                    "Vector_Routing": metrics["routing_trigger_ratio"]["vector"],
                    "Keywords_Routing": metrics["routing_trigger_ratio"]["keywords"],
                    "Graph_Routing": metrics["routing_trigger_ratio"]["graph"],
                    "Expansion_Routing": metrics["routing_trigger_ratio"]["expansion"],
                    "Rerank_Routing": metrics["routing_trigger_ratio"]["rerank"],
                    "Scoring_Routing": metrics["routing_trigger_ratio"]["scoring"],
                    "Update_Routing": metrics["routing_trigger_ratio"]["update"],
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / f"comparison_summary_{timestamp}.csv", index=False)
        
        for level, result in results_by_complexity.items():
            report = {k: v for k, v in result.items() if k != "details"}
            with open(self.output_dir / f"experiment_{level}_{timestamp}.json", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            if save_details and result.get("details"):
                with open(self.output_dir / f"details_{level}_{timestamp}.json", "w", encoding="utf-8") as f:
                    json.dump(result["details"], f, ensure_ascii=False, indent=2)
        
        self._print_comparison_results(results_by_complexity)
    
    def _print_comparison_results(self, results_by_complexity: Dict[str, Dict]) -> None:
        """打印对比结果"""
        print("\n" + "=" * 80)
        print("实验十：按问题复杂度分层分析实验结果对比")
        print("=" * 80)
        
        print(f"\n【数据统计】")
        for level in ["low", "medium", "high"]:
            if level in results_by_complexity:
                count = results_by_complexity[level]["config"]["total_test_samples"]
                print(f"  {level.upper()} 复杂度: {count} 条")
        
        print("\n" + "-" * 80)
        print("【核心评估指标对比】")
        print("-" * 80)
        
        print(f"\n{'指标':<20} {'LOW':<15} {'MEDIUM':<15} {'HIGH':<15}")
        print("-" * 65)
        
        metrics_keys = [
            ("精确匹配率(EM)", "exact_match"),
            ("F1分数", "f1_score"),
            ("平均上下文长度", "avg_context_length"),
        ]
        
        for label, key in metrics_keys:
            values = []
            for level in ["low", "medium", "high"]:
                if level in results_by_complexity:
                    values.append(f"{results_by_complexity[level]['metrics'][key]:.4f}")
                else:
                    values.append("N/A")
            print(f"{label:<20} {values[0]:<15} {values[1]:<15} {values[2]:<15}")
        
        print("\n" + "-" * 80)
        print("【检索指标对比】")
        print("-" * 80)
        
        retrieval_metrics = [
            ("Recall@K", "recall_at_k"),
            ("Precision@K", "precision_at_k"),
            ("MRR", "mrr"),
            ("NDCG", "ndcg"),
            ("MAP", "map_score"),
            ("Hit Rate", "hit_rate"),
            ("Title Recall", "title_recall"),
            ("Title Precision", "title_precision"),
        ]
        
        print(f"\n{'指标':<20} {'LOW':<15} {'MEDIUM':<15} {'HIGH':<15}")
        print("-" * 65)
        
        for label, key in retrieval_metrics:
            values = []
            for level in ["low", "medium", "high"]:
                if level in results_by_complexity:
                    values.append(f"{results_by_complexity[level]['metrics'][key]:.4f}")
                else:
                    values.append("N/A")
            print(f"{label:<20} {values[0]:<15} {values[1]:<15} {values[2]:<15}")
        
        print("\n" + "-" * 80)
        print("【路由触发比例对比】")
        print("-" * 80)
        
        routing_keys = [
            ("向量检索", "vector"),
            ("关键词提取", "keywords"),
            ("图谱检索", "graph"),
            ("图谱扩展", "expansion"),
            ("重排序", "rerank"),
            ("LLM打分", "scoring"),
            ("低分更新", "update"),
        ]
        
        print(f"\n{'路由步骤':<20} {'LOW':<15} {'MEDIUM':<15} {'HIGH':<15}")
        print("-" * 65)
        
        for label, key in routing_keys:
            values = []
            for level in ["low", "medium", "high"]:
                if level in results_by_complexity:
                    ratio = results_by_complexity[level]['metrics']['routing_trigger_ratio'][key]
                    values.append(f"{ratio:.2%}")
                else:
                    values.append("N/A")
            print(f"{label:<20} {values[0]:<15} {values[1]:<15} {values[2]:<15}")
        
        print("\n" + "-" * 80)
        print("【延迟指标对比】")
        print("-" * 80)
        
        latency_metrics = [
            ("向量检索延迟(s)", "avg_vector_latency"),
            ("图谱检索延迟(s)", "avg_graph_latency"),
            ("总延迟(s)", "avg_total_latency"),
        ]
        
        print(f"\n{'指标':<20} {'LOW':<15} {'MEDIUM':<15} {'HIGH':<15}")
        print("-" * 65)
        
        for label, key in latency_metrics:
            values = []
            for level in ["low", "medium", "high"]:
                if level in results_by_complexity:
                    values.append(f"{results_by_complexity[level]['metrics'][key]:.4f}")
                else:
                    values.append("N/A")
            print(f"{label:<20} {values[0]:<15} {values[1]:<15} {values[2]:<15}")
        
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
    output_dir = "e:/Code_Personal/Subject/test02/experiments/exp10_complexity_stratified"
    
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
        sample_size=None,
        save_details=True,
        resume=True,
        balance_samples=True,
    )


if __name__ == "__main__":
    main()
