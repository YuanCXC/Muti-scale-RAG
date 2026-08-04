# -*- coding: utf-8 -*-
"""自适应多尺度 RAG 主实验与消融实验

证据路由策略：细粒度保持—父级回升—图扩展—摘要补充。

支持完整方法、结构图谱消融、语义图谱消融、自适应路由消融和预算控制消融。

本文件为自包含实现，不依赖 new_experiments_v2/core.py。
仅依赖项目基础库 src.*（向量存储、图存储、LLM 客户端、重排序器、日志）。
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# 允许脚本放在项目根目录或 exp_v3 子目录中直接运行。
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "src").exists():
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "src").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.logger import get_logger


logger = get_logger(__name__)


DEFAULT_HOTPOT_DIR = PROJECT_ROOT / "data" / "hotpotqa"


@dataclass
class RetrievalConfig:
    """自适应多尺度 RAG 实验配置。"""

    test_data_path: Path = DEFAULT_HOTPOT_DIR / "validation-00000-of-00001.parquet"
    documents_path: Path = DEFAULT_HOTPOT_DIR / "valid_title_sentence.json"
    paragraph_store_path: Path = DEFAULT_HOTPOT_DIR / "vector_stores" / "valid_title_sentence"
    sentence_store_path: Path = DEFAULT_HOTPOT_DIR / "vector_stores" / "single_sentence"
    output_dir: Path = PROJECT_ROOT / "exp_v3" / "results"

    sample_size: int = 5
    random_seed: int = 42
    k1: int = 10
    k2: int = 20
    k3: int = 7
    hmax: int = 2
    context_budget: int = 3600
    complexity_threshold: float = 0.80
    parent_threshold: float = 0.60
    fragmentation_threshold: float = 0.65

    # 仅优化现有实现细节，不改变双图与三路由框架。
    # 图扩展只使用高置信种子，并限制每个种子的邻居数量，降低错误传播。
    max_graph_seeds: int = 2
    max_graph_neighbors: int = 3
    graph_max_degree: int = 300

    # 二跳扩展采用更严格的触发条件：高复杂度、多跳线索、候选互补不足且较分散。
    second_hop_complementarity_threshold: float = 0.45
    second_hop_fragmentation_threshold: float = 0.70

    # 局部父级回升：命中句前后各保留若干句；定位失败时再回退到完整父文本。
    parent_window_size: int = 2

    # 各路径共享统一 token 预算，但限制证据单元数量，避免简单查询上下文膨胀。
    fine_max_units: int = 4
    parent_max_units: int = 5
    graph_max_units: int = 7
    max_context_units: int = 20

    run_generation: bool = True
    run_judge: bool = True
    use_api_reranker: bool = True
    require_neo4j: bool = True

    max_workers: int = 4
    checkpoint_interval: int = 10
    judge_max_retries: int = 3

    retrieval_cache_dir: str = ""
    skip_retrieval: bool = False


@dataclass
class EvidenceUnit:
    """一个检索到的证据单元，可以是句子、段落或摘要粒度。"""

    id: str
    title: str
    content: str
    score: float = 0.0
    source: str = "unknown"
    granularity: str = "sentence"
    is_sentence_level: bool = False
    sentence_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return estimate_tokens(self.content)


@dataclass
class QueryComplexity:
    score: float
    entity_count: int
    relation_constraint_count: int
    multi_hop_indicator: bool
    text_length: int
    detail: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvidenceStatus:
    score: float
    concentration: float
    fragmentation: float
    complementarity: float
    short_sentence_ratio: float
    shared_parent_ratio: float
    detail: Dict[str, Any] = field(default_factory=dict)


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


@dataclass
class MethodResult:
    units: List[EvidenceUnit]
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankSearchResult:
    """现有重排序器实现接受的最小适配器。"""

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryComplexityScorer:
    """基于规则的实现。"""

    QUESTION_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "did",
        "do",
        "does",
        "for",
        "from",
        "in",
        "is",
        "of",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
    }

    MULTI_HOP_PATTERNS = [
        r"\b(same|different|both|either|neither)\b",
        r"\b(compare|comparison|versus|vs\.?|difference|similar)\b",
        r"\b(before|after|during|while|until|since)\b",
        r"\b(author|director|founder|producer|creator).*\b(born|birth|nationality|country|city)\b",
        r"\b(who|what|which|where|when)\b.*\b(who|what|which|where|when)\b",
    ]

    RELATION_PATTERNS = [
        r"\b(same|different)\s+(nationality|country|state|city|language|genre)\b",
        r"\b(older|younger|earlier|later|larger|smaller|more|less|fewer|greater)\b",
        r"\b(belong|owned|founded|created|established|directed|written|produced|born)\b",
        r"\b(nationality|country|birthplace|occupation|genre|release|location)\b",
    ]

    def compute(self, query: str) -> QueryComplexity:
        entities = self._extract_entities(query)
        relation_count = self._count_relation_constraints(query)
        multi_hop = self._detect_multi_hop(query)
        text_length = len(query)

        entity_score = min(len(entities) / 2.0, 1.0)
        relation_score = min(relation_count / 1.0, 1.0)
        multi_hop_score = 1.0 if multi_hop else 0.0
        length_score = min(text_length / 120.0, 1.0)

        score = (
            0.35 * entity_score
            + 0.25 * relation_score
            + 0.25 * multi_hop_score
            + 0.15 * length_score
        )

        return QueryComplexity(
            score=round(min(score, 1.0), 4),
            entity_count=len(entities),
            relation_constraint_count=relation_count,
            multi_hop_indicator=multi_hop,
            text_length=text_length,
            detail={
                "entity_score": round(entity_score, 4),
                "relation_score": round(relation_score, 4),
                "multi_hop_score": round(multi_hop_score, 4),
                "length_score": round(length_score, 4),
            },
        )

    def _extract_entities(self, query: str) -> Set[str]:
        entities: Set[str] = set()

        for quoted in re.findall(r'"([^"]+)"|\'([^\']+)\'', query):
            item = quoted[0] or quoted[1]
            if item:
                entities.add(item.strip().lower())

        for match in re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b", query):
            cleaned = match.strip()
            if cleaned.lower() not in self.QUESTION_WORDS and len(cleaned) > 1:
                entities.add(cleaned.lower())

        return entities

    def _count_relation_constraints(self, query: str) -> int:
        return sum(1 for pattern in self.RELATION_PATTERNS if re.search(pattern, query, re.IGNORECASE))

    def _detect_multi_hop(self, query: str) -> bool:
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in self.MULTI_HOP_PATTERNS)


def estimate_tokens(text: str) -> int:
    """用于上下文预算核算的确定性 token 估算。"""

    if not text:
        return 0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_chars = len(text) - chinese_chars
    return max(1, int(chinese_chars * 1.5 + other_chars / 4.0))


def compute_retrieval_metrics(
    retrieved_titles: Sequence[str],
    relevant_titles: Set[str],
    avg_context_len: float,
    latency_ms: float,
    expanded_nodes: float = 0.0,
) -> ExperimentMetrics:
    """计算标题级别的 Recall、Precision、MRR、NDCG 和 MAP。"""

    relevant = set(relevant_titles)
    retrieved = [title for title in retrieved_titles if title]
    if not relevant:
        return ExperimentMetrics(avg_len=avg_context_len, time_ms=latency_ms, expanded_nodes=expanded_nodes)

    hits = [1 if title in relevant else 0 for title in retrieved]
    hit_count = sum(hits)
    recall = hit_count / len(relevant)
    precision = hit_count / len(retrieved) if retrieved else 0.0

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


def ci95(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if len(nums) < 2:
        return None
    return 1.96 * statistics.stdev(nums) / math.sqrt(len(nums))


def round2(value: Optional[float]) -> Optional[float]:
    """保留 2 位小数"""
    if value is None or not isinstance(value, (int, float)) or math.isnan(float(value)):
        return None
    return round(float(value), 2)


def unique_by_title(units: Iterable[EvidenceUnit], keep_content_distinct: bool = False) -> List[EvidenceUnit]:
    seen: Set[Tuple[str, str]] = set()
    out: List[EvidenceUnit] = []
    for unit in units:
        key = (unit.title, unit.content if keep_content_distinct else "")
        if unit.title and key not in seen:
            seen.add(key)
            out.append(unit)
    return out


def unique_nonempty_titles(titles: Iterable[str]) -> List[str]:
    """按输入顺序去重标题，并过滤空标题。"""
    seen: Set[str] = set()
    result: List[str] = []
    for title in titles:
        cleaned = str(title or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def context_len(units: Sequence[EvidenceUnit]) -> int:
    return sum(unit.token_count for unit in units)


class AdaptiveMultiscaleRAGExperiment:
    """自适应多尺度 RAG 主实验与消融实验。

    证据路由：细粒度保持—父级回升—图扩展—摘要补充。
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.complexity_scorer = QueryComplexityScorer()

        self.embedding_client: Optional[Any] = None
        self.llm_client: Optional[Any] = None
        self.reranker: Optional[Any] = None
        self.graph_store: Optional[Any] = None
        self.sentence_store: Optional[Any] = None
        self.paragraph_store: Optional[Any] = None
        self.test_data: Optional[pd.DataFrame] = None

        self.documents: List[Dict[str, Any]] = []
        self.title_to_content: Dict[str, str] = {}

        self._rerank_semaphore = threading.Semaphore(5)
        self._rerank_cache: Dict[str, List[EvidenceUnit]] = {}

    # ------------------------------------------------------------------
    # 资源加载
    # ------------------------------------------------------------------
    def load_resources(self) -> None:
        logger.info("加载 HotpotQA 资源（自适应多尺度 RAG 实验）")
        self._load_documents()
        self._load_test_data()
        self._load_vector_stores()
        self._load_embedding_client()
        self._load_graph_store()
        self._load_optional_clients()

    def _load_documents(self) -> None:
        with open(self.config.documents_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        self.title_to_content = {
            doc.get("title", ""): doc.get("sentence_total", doc.get("content", ""))
            for doc in self.documents
            if doc.get("title")
        }
        logger.info("已加载 %s 个标题级文档", len(self.title_to_content))

    def _load_test_data(self) -> None:
        data = pd.read_parquet(self.config.test_data_path)
        if self.config.sample_size and self.config.sample_size < len(data):
            data = data.sample(n=self.config.sample_size, random_state=self.config.random_seed)
        self.test_data = data.reset_index(drop=True)
        logger.info("已加载 %s 条 HotpotQA 样本", len(self.test_data))

    def _load_vector_stores(self) -> None:
        from src.storage.vector_store.faiss_store import FAISSVectorStore

        self.sentence_store = FAISSVectorStore()
        self.sentence_store.load(str(self.config.sentence_store_path))
        logger.info("已加载句子向量库：%s 条向量", self.sentence_store.count())

        self.paragraph_store = FAISSVectorStore()
        self.paragraph_store.load(str(self.config.paragraph_store_path))
        logger.info("已加载段落向量库：%s 条向量", self.paragraph_store.count())

    def _load_embedding_client(self) -> None:
        from src.llms.embedding_client import EmbeddingClient

        self.embedding_client = EmbeddingClient()
        logger.info("Embedding 客户端就绪：dimension=%s", self.embedding_client.dimension)

    def _load_graph_store(self) -> None:
        try:
            from src.storage.graph_store.neo4j_store import Neo4jGraphStore

            self.graph_store = Neo4jGraphStore()
            node_count = self.graph_store.count_nodes()
            edge_count = self.graph_store.count_edges()
            logger.info("Neo4j 就绪：nodes=%s, edges=%s", node_count, edge_count)
        except Exception as exc:
            if self.config.require_neo4j:
                raise
            logger.warning("Neo4j 不可用，图扩展方法将降级：%s", exc)
            self.graph_store = None

    def _load_optional_clients(self) -> None:
        if self.config.run_generation or self.config.run_judge:
            from src.llms.deepseek_client import DeepSeekClient

            self.llm_client = DeepSeekClient()

        if self.config.use_api_reranker:
            try:
                from src.retrievers.reranker import create_reranker

                self.reranker = create_reranker(mode="api")
            except Exception as exc:
                logger.warning("API 重排序器不可用，改用词汇重排序：%s", exc)
                self.reranker = None

    # ------------------------------------------------------------------
    # 核心检索原语
    # ------------------------------------------------------------------
    def vector_retrieve(self, query: str, store_name: str = "sentence", top_k: Optional[int] = None) -> List[EvidenceUnit]:
        store = self.sentence_store if store_name == "sentence" else self.paragraph_store
        if store is None or self.embedding_client is None:
            return []

        qv = self.embedding_client.embed(query)
        if len(qv.shape) == 2:
            qv = qv[0]
        results = store.search(qv, top_k=top_k or self.config.k1)

        units = []
        for result in results:
            title = result.metadata.extra.get("title", result.metadata.doc_id)
            sentence_id = result.metadata.extra.get("sentence_id", "")
            content = result.metadata.content
            units.append(
                EvidenceUnit(
                    id=result.id,
                    title=title,
                    content=content,
                    score=float(result.score),
                    source=f"vector_{store_name}",
                    granularity="sentence" if sentence_id else "paragraph",
                    is_sentence_level=bool(sentence_id),
                    sentence_id=sentence_id,
                    metadata=dict(result.metadata.extra),
                )
            )
        return units

    def keyword_retrieve(self, query: str, top_k: Optional[int] = None) -> List[EvidenceUnit]:
        keywords = self.extract_keywords(query)
        if not keywords or self.graph_store is None:
            return []

        limit = top_k or self.config.k2
        cypher = """
        MATCH (n:Section)
        WHERE any(k IN $keywords WHERE toLower(n.title) CONTAINS toLower(k))
        RETURN n.title AS title, n.sentence_total AS content
        LIMIT $limit
        """
        try:
            rows = self.graph_store.query(cypher, {"keywords": keywords, "limit": limit})
        except Exception as exc:
            logger.warning("关键词图检索失败：%s", exc)
            return []

        units = []
        for row in rows:
            title = str(row.get("title", "")).strip('"')
            content = str(row.get("content", "") or self.title_to_content.get(title, ""))
            if title and content:
                units.append(
                    EvidenceUnit(
                        id=f"kw::{title}",
                        title=title,
                        content=content,
                        score=0.65,
                        source="keyword_graph",
                        granularity="paragraph",
                    )
                )
        return unique_by_title(units)

    def extract_keywords(self, query: str) -> List[str]:
        entities = sorted(self.complexity_scorer._extract_entities(query), key=len, reverse=True)
        keywords = [entity.strip() for entity in entities if entity.strip()]
        if keywords:
            return keywords[:10]

        stop = QueryComplexityScorer.QUESTION_WORDS | {"with", "that", "this", "into", "have", "has"}
        tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", query) if t.lower() not in stop and len(t) > 3]
        return list(dict.fromkeys(tokens))[:10]

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
                logger.warning("API 重排序失败，改用词汇重排序：%s", exc)

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

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """对英文/中文文本进行轻量句子切分。"""
        if not text:
            return []
        parts = re.split(r"(?<=[.!?。！？])\s+|(?<=[。！？])", text)
        return [part.strip() for part in parts if part and part.strip()]

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip().lower()

    def _local_parent_window(self, full_text: str, child_text: str) -> Tuple[str, str]:
        """返回命中句附近的局部父级窗口；无法定位时回退到完整父文本。"""
        sentences = self._split_sentences(full_text)
        if not sentences:
            return full_text, "full_parent_fallback"

        child_norm = self._normalize_for_match(child_text)
        if not child_norm:
            return full_text, "full_parent_fallback"

        best_idx = -1
        best_overlap = 0.0
        child_terms = set(re.findall(r"[A-Za-z0-9]+", child_norm))

        for idx, sentence in enumerate(sentences):
            sentence_norm = self._normalize_for_match(sentence)
            if child_norm in sentence_norm or sentence_norm in child_norm:
                best_idx = idx
                best_overlap = 1.0
                break

            sentence_terms = set(re.findall(r"[A-Za-z0-9]+", sentence_norm))
            overlap = len(child_terms & sentence_terms) / max(len(child_terms), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx

        if best_idx < 0 or best_overlap < 0.35:
            return full_text, "full_parent_fallback"

        window = max(0, int(self.config.parent_window_size))
        start = max(0, best_idx - window)
        end = min(len(sentences), best_idx + window + 1)
        local_text = " ".join(sentences[start:end]).strip()
        return local_text or full_text, "local_parent_window"

    def parent_map(self, units: Sequence[EvidenceUnit]) -> List[EvidenceUnit]:
        """结构层级父级回升。

        该函数实现结构图谱中的“子节点 -> 父级文本边界”操作。当前工程中
        使用 ``title_to_content`` 作为结构图父级索引的缓存实现，而不是每次
        在线查询 Neo4j；因此父级回升是结构图谱支持的一项操作，不等同于
        整个结构图谱。
        """
        mapped: List[EvidenceUnit] = []
        for unit in units:
            full = self.title_to_content.get(unit.title)
            if full and (unit.is_sentence_level or len(unit.content) < len(full)):
                parent_content, parent_mode = self._local_parent_window(full, unit.content)
                mapped.append(
                    EvidenceUnit(
                        id=f"parent::{unit.title}",
                        title=unit.title,
                        content=parent_content,
                        score=unit.score,
                        source=f"{unit.source}+parent",
                        granularity="paragraph",
                        metadata={
                            "parent_of": unit.id,
                            "parent_mode": parent_mode,
                            "trigger_sentence": unit.content,
                            "structure_source": "title_parent_index",
                        },
                    )
                )
            else:
                mapped.append(unit)
        return unique_by_title(mapped)

    @staticmethod
    def title_variants(title: str) -> List[str]:
        """生成可用于 Neo4j 精确匹配的标题变体。

        HTML 实体在 Python 侧处理，避免对数据库属性调用 lower/replace，
        从而保留 ``Section.title`` 索引的可用性。
        """
        raw = str(title or "").strip()
        if not raw:
            return []
        decoded = html.unescape(raw)
        encoded = html.escape(decoded, quote=False)
        variants = [raw, decoded, encoded]
        return list(dict.fromkeys(v for v in variants if v))

    def graph_expand(
        self,
        seed_titles: Sequence[str],
        hops: int = 1,
        limit_per_seed: Optional[int] = None,
    ) -> List[EvidenceUnit]:
        """沿语义图 ``SEMANTIC_LINKS`` 扩展邻居证据。

        每个标题先在 Python 侧生成 HTML 实体变体，再使用属性精确匹配。
        建议在 Neo4j 中建立索引：

        ``CREATE INDEX section_title_index IF NOT EXISTS FOR (n:Section) ON (n.title)``
        """
        if self.graph_store is None:
            return []

        hops = max(1, min(int(hops), self.config.hmax))
        limit = int(limit_per_seed or self.config.max_graph_neighbors)
        max_degree = int(self.config.graph_max_degree)
        ordered_seeds = unique_nonempty_titles(seed_titles)
        expanded: List[EvidenceUnit] = []
        seen = set(ordered_seeds)

        cypher = f"""
        MATCH (start:Section {{title: $title}})
        MATCH (start)-[:SEMANTIC_LINKS]-(first)
        MATCH p = (first)-[r*0..{hops}]-(last)
        MATCH (last)-[:SEMANTIC_LINKS]-(n:Section)
        WHERE n <> start
          AND ALL(rel IN r WHERE type(rel) <> 'SEPARATES')
          AND ALL(x IN nodes(p) WHERE COUNT {{ (x)--() }} <= {max_degree})
        RETURN DISTINCT
            n.title AS title,
            n.sentence_total AS content,
            length(p) AS path_len,
            [node IN nodes(p) |
                coalesce(node.title, node.name, node.id, head(labels(node)))] AS path_nodes
        ORDER BY path_len ASC
        LIMIT $limit
        """

        for seed_title in ordered_seeds:
            rows: List[Dict[str, Any]] = []
            matched_variant = seed_title
            for variant in self.title_variants(seed_title):
                try:
                    candidate_rows = self.graph_store.query(
                        cypher,
                        {"title": variant, "limit": limit},
                    )
                except Exception as exc:
                    logger.warning("图扩展失败（%s）：%s", seed_title, exc)
                    candidate_rows = []
                if candidate_rows:
                    rows = candidate_rows
                    matched_variant = variant
                    break

            if not rows:
                logger.debug("语义图中未匹配到种子标题：%s", seed_title)
                continue

            for row in rows:
                new_title = str(row.get("title", "")).strip('"')
                content = str(row.get("content", "") or self.title_to_content.get(new_title, ""))
                if not new_title or not content or new_title in seen:
                    continue

                raw_path_nodes = row.get("path_nodes", [])
                if hasattr(raw_path_nodes, "tolist"):
                    raw_path_nodes = raw_path_nodes.tolist()
                path_nodes = [str(node) for node in raw_path_nodes] if isinstance(raw_path_nodes, list) else []
                if not path_nodes or path_nodes[0] not in self.title_variants(seed_title):
                    path_nodes = [seed_title] + path_nodes
                if path_nodes[-1] != new_title:
                    path_nodes.append(new_title)

                seen.add(new_title)
                expanded.append(
                    EvidenceUnit(
                        id=f"graph::{new_title}",
                        title=new_title,
                        content=content,
                        score=0.55,
                        source=f"graph_{hops}hop",
                        granularity="paragraph",
                        metadata={
                            "seed_title": seed_title,
                            "matched_seed_variant": matched_variant,
                            "requested_hops": hops,
                            "path_len": row.get("path_len"),
                            "path_nodes": path_nodes,
                        },
                    )
                )
        return expanded

    def add_summary_evidence(self, seed_titles: Sequence[str], query: str = "") -> List[EvidenceUnit]:
        """提取关键句子作为摘要证据

        Args:
            seed_titles: 种子标题列表
            query: 查询文本，用于提取相关句子
        """
        summaries = []
        query_terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)) if query else set()

        for title in seed_titles:
            content = self.title_to_content.get(title, "")
            if not content:
                continue

            if query_terms:
                sentences = re.split(r'(?<=[.!?])\s+', content)
                scored_sentences = []
                for sent in sentences:
                    sent_terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", sent))
                    overlap = len(query_terms & sent_terms)
                    if overlap > 0:
                        scored_sentences.append((overlap, len(sent), sent))

                scored_sentences.sort(key=lambda x: (-x[0], x[1]))
                summary_sentences = [sent for _, _, sent in scored_sentences[:3]]
                summary = ' '.join(summary_sentences) if summary_sentences else content[:600]
            else:
                summary = content[:600]

            summary = summary[:900]

            summaries.append(
                EvidenceUnit(
                    id=f"summary::{title}",
                    title=title,
                    content=summary,
                    score=0.30,
                    source="summary",
                    granularity="summary",
                )
            )
        return summaries

    def select_with_budget(self, query: str, units: Sequence[EvidenceUnit], max_units: Optional[int] = None) -> List[EvidenceUnit]:
        """在上下文预算内动态计算相关性、新颖度与长度成本并选择证据。

        原实现在线性排序前一次性计算 novelty，导致所有候选初始新颖度几乎相同。
        此处改为逐轮贪心更新，使补全价值真正依赖已选证据集合。
        """
        selected: List[EvidenceUnit] = []
        selected_terms: Set[str] = set()
        selected_titles: Set[str] = set()
        budget = self.config.context_budget
        max_units = max_units or self.config.max_context_units
        total_tokens = 0

        query_terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", query))
        candidates: List[Tuple[EvidenceUnit, Set[str]]] = []
        for unit in units:
            if not unit.title or not unit.content:
                continue
            terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", f"{unit.title} {unit.content}"))
            candidates.append((unit, terms))

        remaining = list(candidates)
        while remaining and len(selected) < max_units:
            best_idx: Optional[int] = None
            best_score = -float("inf")

            for idx, (unit, terms) in enumerate(remaining):
                if unit.title in selected_titles:
                    continue

                unit_tokens = unit.token_count
                if total_tokens + unit_tokens > budget and selected:
                    continue

                relevance = len(query_terms & terms) / max(len(query_terms), 1)
                novelty = 1.0 if not selected_terms else len(terms - selected_terms) / max(len(terms), 1)
                cost = unit_tokens / max(budget, 1)

                source_bonus = 0.04 if unit.granularity == "sentence" else 0.0
                if unit.granularity == "summary":
                    source_bonus -= 0.03

                score = (
                    0.45 * unit.score
                    + 0.30 * relevance
                    + 0.20 * novelty
                    - 0.15 * cost
                    + source_bonus
                )
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            unit, terms = remaining.pop(best_idx)
            selected.append(unit)
            selected_titles.add(unit.title)
            selected_terms.update(terms)
            total_tokens += unit.token_count
            remaining = [(u, t) for u, t in remaining if u.title not in selected_titles]

        return selected

    def hard_context_limit(
        self,
        units: Sequence[EvidenceUnit],
        max_tokens: Optional[int] = None,
        max_units: Optional[int] = None,
    ) -> List[EvidenceUnit]:
        """仅执行模型安全上限，不进行预算感知价值排序。

        该函数用于 ``w/o Budget Control``：候选按原始分数排序，
        只保留硬 token 上限和最大单元数，避免超出生成模型上下文。
        """
        token_limit = int(max_tokens or self.config.context_budget)
        unit_limit = int(max_units or self.config.max_context_units)
        ordered = sorted(
            unique_by_title(units, keep_content_distinct=False),
            key=lambda unit: float(unit.score),
            reverse=True,
        )
        selected: List[EvidenceUnit] = []
        total_tokens = 0
        for unit in ordered:
            if len(selected) >= unit_limit:
                break
            if total_tokens + unit.token_count > token_limit:
                continue
            selected.append(unit)
            total_tokens += unit.token_count
        return selected

    def _select_route_candidates(
        self,
        query: str,
        units: Sequence[EvidenceUnit],
        route_max_units: int,
        enable_budget_control: bool,
    ) -> List[EvidenceUnit]:
        if enable_budget_control:
            return self.select_with_budget(query, units, max_units=route_max_units)
        return self.hard_context_limit(
            units,
            max_tokens=self.config.context_budget,
            max_units=self.config.max_context_units,
        )

    # ------------------------------------------------------------------
    # 自适应多尺度方法（本文方法）
    # ------------------------------------------------------------------
    def retrieve_adaptive(
        self,
        query: str,
        enable_graph_expansion: bool = True,
        enable_parent: bool = True,
        enable_summary: bool = True,
        enable_budget_control: bool = True,
        forced_route: Optional[str] = None,
        forced_hops: Optional[int] = None,
        force_parent: bool = False,
        fine_only: bool = False,
        ablation_name: str = "Full",
    ) -> MethodResult:
        """运行完整方法或某一消融设置。

        消融语义：
        - ``enable_parent=False``：删除结构图谱支持的父级回升；
        - ``enable_graph_expansion=False``：删除语义图扩展；
        - ``forced_route='graph_expansion'``：统一采用父级回升 + 一跳语义扩展；
        - ``enable_budget_control=False``：删除价值—成本筛选，仅保留硬上限。
        """
        start = time.perf_counter()
        initial = self.vector_retrieve(query, "sentence", self.config.k1) + self.keyword_retrieve(query, self.config.k2)
        initial = unique_by_title(initial, keep_content_distinct=True)
        reranked = self.rerank_units(query, initial, self.config.k3)

        if fine_only:
            units = self._select_route_candidates(
                query,
                unique_by_title(reranked),
                self.config.fine_max_units,
                enable_budget_control,
            )
            stats = self._stats(start, units, expanded_nodes=0, route="fine_only", candidate_count=len(reranked))
            stats.update({"ablation": ablation_name, "budget_control": enable_budget_control})
            return MethodResult(units=units, stats=stats)

        if force_parent and enable_parent:
            parent_candidates = self.parent_map(reranked)
            units = self._select_route_candidates(
                query,
                parent_candidates,
                self.config.parent_max_units,
                enable_budget_control,
            )
            stats = self._stats(start, units, expanded_nodes=0, route="parent_all", candidate_count=len(parent_candidates))
            stats.update({"ablation": ablation_name, "budget_control": enable_budget_control})
            return MethodResult(units=units, stats=stats)

        original_route, route_detail = self.choose_adaptive_route(query, reranked)
        route = original_route
        if forced_route is not None:
            if forced_route not in {"fine_grained", "local_parent", "graph_expansion"}:
                raise ValueError(f"不支持的 forced_route：{forced_route}")
            route = forced_route
            route_detail["forced_route"] = forced_route

        # 删除结构图谱后，原本的父级回升路径回退为细粒度保持，
        # 不能额外触发语义扩展，否则会混淆消融结论。
        if route == "local_parent" and not enable_parent:
            route = "fine_grained"
            route_detail["structural_fallback"] = "fine_grained"

        # 删除语义图谱后，原本的图扩展路径回退为父级回升；
        # 若结构图也被删除，则继续回退为细粒度保持。
        if route == "graph_expansion" and not enable_graph_expansion:
            route = "local_parent" if enable_parent else "fine_grained"
            route_detail["semantic_fallback"] = route

        route_detail.update(
            {
                "original_route": original_route,
                "route": route,
                "enable_parent": enable_parent,
                "enable_graph_expansion": enable_graph_expansion,
                "enable_budget_control": enable_budget_control,
                "ablation": ablation_name,
            }
        )

        selected: List[EvidenceUnit]
        expanded: List[EvidenceUnit] = []
        hops = 0
        graph_seed_titles: List[str] = []
        candidate_count = len(reranked)

        if route == "fine_grained":
            fine_candidates = unique_by_title(reranked)
            candidate_count = len(fine_candidates)
            selected = self._select_route_candidates(
                query,
                fine_candidates,
                self.config.fine_max_units,
                enable_budget_control,
            )

        elif route == "local_parent":
            parent_candidates = self.parent_map(reranked) if enable_parent else unique_by_title(reranked)
            candidate_count = len(parent_candidates)
            selected = self._select_route_candidates(
                query,
                parent_candidates,
                self.config.parent_max_units,
                enable_budget_control,
            )

        else:
            seeds = self.parent_map(reranked) if enable_parent else list(reranked)
            graph_seed_titles = unique_nonempty_titles(u.title for u in seeds)[: self.config.max_graph_seeds]
            hops = int(forced_hops or self._dynamic_hops(route_detail))
            if enable_graph_expansion and graph_seed_titles:
                expanded = self.graph_expand(
                    graph_seed_titles,
                    hops=hops,
                    limit_per_seed=self.config.max_graph_neighbors,
                )

            all_candidates = list(seeds) + expanded
            current_tokens = context_len(all_candidates)
            if enable_summary and current_tokens >= self.config.context_budget * 0.85:
                summary_titles = unique_nonempty_titles(u.title for u in seeds)
                summaries = self.add_summary_evidence(summary_titles, query)
            else:
                summaries = []

            all_candidates = all_candidates + summaries
            candidate_count = len(all_candidates)
            selected = self._select_route_candidates(
                query,
                all_candidates,
                self.config.graph_max_units,
                enable_budget_control,
            )

        stats = self._stats(
            start,
            selected,
            expanded_nodes=len(expanded),
            route=route,
            candidate_count=candidate_count,
        )
        stats.update(
            {
                "ablation": ablation_name,
                "original_route": original_route,
                "route_detail": route_detail,
                "graph_seed_count": len(graph_seed_titles),
                "graph_hops": hops,
                "selected_units": len(selected),
                "budget_control": enable_budget_control,
                "budget_utilization": round(context_len(selected) / max(self.config.context_budget, 1), 4),
                "local_parent_windows": sum(
                    1 for unit in selected if unit.metadata.get("parent_mode") == "local_parent_window"
                ),
                "full_parent_fallbacks": sum(
                    1 for unit in selected if unit.metadata.get("parent_mode") == "full_parent_fallback"
                ),
            }
        )
        return MethodResult(units=selected, stats=stats)

    def choose_adaptive_route(self, query: str, candidates: Sequence[EvidenceUnit]) -> Tuple[str, Dict[str, Any]]:
        """依据查询复杂度与证据状态选择最低成本的充分路径。"""
        complexity = self.complexity_scorer.compute(query)
        evidence_status = self.evaluate_evidence_status(candidates, query)

        evidence_sufficient = (
            evidence_status.score >= self.config.parent_threshold
            and evidence_status.short_sentence_ratio < 0.4
            and evidence_status.complementarity >= 0.55
        )
        graph_needed = (
            (complexity.score >= self.config.complexity_threshold or complexity.multi_hop_indicator)
            and (
                evidence_status.fragmentation >= self.config.fragmentation_threshold
                or evidence_status.complementarity < 0.55
            )
        )

        if evidence_sufficient and not graph_needed:
            route = "fine_grained"
        elif graph_needed:
            route = "graph_expansion"
        else:
            route = "local_parent"

        return route, {
            "complexity_score": complexity.score,
            "entity_count": complexity.entity_count,
            "relation_constraint_count": complexity.relation_constraint_count,
            "multi_hop_indicator": complexity.multi_hop_indicator,
            "evidence_score": evidence_status.score,
            "concentration": evidence_status.concentration,
            "fragmentation": evidence_status.fragmentation,
            "complementarity": evidence_status.complementarity,
            "short_sentence_ratio": evidence_status.short_sentence_ratio,
            "shared_parent_ratio": evidence_status.shared_parent_ratio,
            "evidence_sufficient": evidence_sufficient,
            "graph_needed": graph_needed,
            "route": route,
        }

    def evaluate_evidence_status(self, candidates: Sequence[EvidenceUnit], query: str) -> EvidenceStatus:
        if not candidates:
            return EvidenceStatus(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, {"reason": "empty"})

        titles = [u.title for u in candidates if u.title]
        unique_titles = set(titles)
        max_same_title = max((titles.count(t) for t in unique_titles), default=0)
        concentration = max_same_title / len(titles) if titles else 0.0
        fragmentation = (len(unique_titles) - 1) / max(len(titles), 1)

        query_terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", query))
        content = " ".join(f"{u.title} {u.content}" for u in candidates).lower()
        complementarity = len([t for t in query_terms if t in content]) / max(len(query_terms), 1)

        short_sentence_ratio = sum(1 for u in candidates if u.is_sentence_level and u.token_count < 18) / len(candidates)
        shared_parent_ratio = 1.0 - len(unique_titles) / max(len(titles), 1)
        evidence_score = 0.40 * concentration + 0.35 * complementarity + 0.25 * (1.0 - fragmentation)

        return EvidenceStatus(
            score=round(evidence_score, 4),
            concentration=round(concentration, 4),
            fragmentation=round(fragmentation, 4),
            complementarity=round(complementarity, 4),
            short_sentence_ratio=round(short_sentence_ratio, 4),
            shared_parent_ratio=round(shared_parent_ratio, 4),
            detail={"unique_titles": len(unique_titles), "candidate_count": len(candidates)},
        )

    def _dynamic_hops(self, route_detail: Dict[str, Any]) -> int:
        """严格控制二跳：仅在复杂、多跳、候选互补不足且较分散时触发。"""
        high_complexity = route_detail.get("complexity_score", 0.0) >= self.config.complexity_threshold
        explicit_multi_hop = bool(route_detail.get("multi_hop_indicator", False))
        weak_complement = (
            route_detail.get("complementarity", 0.0)
            < self.config.second_hop_complementarity_threshold
        )
        highly_fragmented = (
            route_detail.get("fragmentation", 0.0)
            >= self.config.second_hop_fragmentation_threshold
        )
        use_second_hop = high_complexity and explicit_multi_hop and weak_complement and highly_fragmented
        return self.config.hmax if use_second_hop else 1

    def _stats(
        self,
        start: float,
        units: Sequence[EvidenceUnit],
        expanded_nodes: int,
        route: str,
        candidate_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "time_ms": (time.perf_counter() - start) * 1000,
            "avg_len": context_len(units),
            "expanded_nodes": expanded_nodes,
            "candidate_count": candidate_count if candidate_count is not None else len(units),
            "selected_units": len(units),
            "route": route,
        }

    # ------------------------------------------------------------------
    # 生成和判断
    # ------------------------------------------------------------------
    def generate_answer(self, question: str, units: Sequence[EvidenceUnit]) -> str:
        if not self.config.run_generation or self.llm_client is None:
            return ""

        evidence = "\n\n".join(
            f"[{idx}] {unit.title}: {unit.content}" for idx, unit in enumerate(units, start=1)
        )
        prompt = (
            "Answer the question using only the evidence below. "
            "Perform any necessary comparison, temporal reasoning, or relation linking "
            "across the provided evidence. Do not refuse merely because the answer is not "
            "stated in one sentence. Say that it cannot be determined only when a required "
            "fact is genuinely absent from the evidence. Return a concise answer.\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer:"
        )
        from src.llms.base_client import Message

        response = self.llm_client.generate([Message(role="user", content=prompt)], temperature=0.0, max_tokens=256)
        return response.content.strip() if response.success else ""

    @staticmethod
    def is_refusal(answer: str) -> bool:
        text = re.sub(r"\s+", " ", str(answer or "").strip().lower())
        patterns = [
            "cannot be determined",
            "can't be determined",
            "can not be determined",
            "insufficient evidence",
            "insufficient information",
            "not enough information",
            "unable to determine",
        ]
        return any(pattern in text for pattern in patterns)

    def judge_answer(
        self,
        question: str,
        ground_truth: str,
        answer: str,
        units: Sequence[EvidenceUnit],
    ) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "correctness": None,
            "faithfulness": None,
            "answer_relevance": None,
            "context_relevance": None,
            "_judge_valid": False,
            "_judge_attempts": 0,
            "_is_refusal": self.is_refusal(answer),
        }
        if not self.config.run_judge or self.llm_client is None or not answer:
            if base["_is_refusal"]:
                base["correctness"] = 0.0
                base["answer_relevance"] = 0.0
            return base

        context = "\n\n".join(f"{u.title}: {u.content}" for u in units)[:5000]
        prompt = f"""You are a strict RAG evaluator. Score each metric from 0 to 1.

Question: {question}
Reference answer: {ground_truth}
Generated answer: {answer}

Retrieved context:
{context}

Rules:
1. A refusal such as "cannot be determined" is incorrect when the reference answer is specific.
2. For such a refusal, correctness must be 0 and answer_relevance must not exceed 0.25.
3. Correctness measures agreement with the reference answer, not merely factual safety.
4. Return only one JSON object with keys correctness, faithfulness,
   answer_relevance, context_relevance.
"""
        from src.llms.base_client import Message

        parsed: Optional[Dict[str, Any]] = None
        max_attempts = max(1, int(self.config.judge_max_retries))
        for attempt in range(1, max_attempts + 1):
            base["_judge_attempts"] = attempt
            response = self.llm_client.generate(
                [Message(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=180,
            )
            if response.success:
                parsed = self._parse_json_object(response.content)
                if parsed:
                    break
            if attempt < max_attempts:
                time.sleep(0.5 * attempt)

        if parsed:
            for key in ["correctness", "faithfulness", "answer_relevance", "context_relevance"]:
                try:
                    base[key] = max(0.0, min(1.0, float(parsed.get(key))))
                except (TypeError, ValueError):
                    base[key] = None
            base["_judge_valid"] = all(
                isinstance(base[key], (int, float))
                for key in ["correctness", "faithfulness", "answer_relevance", "context_relevance"]
            )

        # 确定性后处理，避免评价模型把拒答误判为正确。
        if base["_is_refusal"]:
            base["correctness"] = 0.0
            current_relevance = base.get("answer_relevance")
            base["answer_relevance"] = min(float(current_relevance), 0.25) if isinstance(current_relevance, (int, float)) else 0.0

        return base

    @staticmethod
    def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # 评估框架（样本×方法双层并发，检索与生成两阶段）
    # ------------------------------------------------------------------
    def evaluate_methods(self, method_fns: Dict[str, Callable[[str], MethodResult]], desc: str = "评估方法") -> Dict[str, List[Dict[str, Any]]]:
        """评估所有方法（支持样本×方法双层并发）

        Args:
            method_fns: 方法名称到函数的映射
            desc: 进度条描述
        """
        assert self.test_data is not None
        rows_by_method: Dict[str, List[Dict[str, Any]]] = {name: [] for name in method_fns}
        rows_lock = threading.Lock()

        samples = list(self.test_data.iterrows())
        method_names = list(method_fns.keys())

        def retrieval_phase(idx: int, sample: pd.Series, method_name: str) -> Tuple[str, pd.Series, MethodResult, Set[str]]:
            """阶段 1：检索（受重排序限制）"""
            fn = method_fns[method_name]
            question = str(sample.get("question", ""))
            relevant_titles = self.get_relevant_titles(sample)

            method_result = MethodResult(units=[], stats={"error": "uninitialized", "time_ms": 0.0, "avg_len": 0.0, "expanded_nodes": 0})
            for attempt in range(3):
                try:
                    method_result = fn(question)
                    break
                except Exception as exc:
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    logger.error("%s 在样本 %s 上失败：%s", method_name, sample.get("id"), exc)
                    method_result = MethodResult(units=[], stats={"error": str(exc), "time_ms": 0.0, "avg_len": 0.0, "expanded_nodes": 0})

            return method_name, sample, method_result, relevant_titles

        def generation_phase(method_name: str, sample: pd.Series, method_result: MethodResult, relevant_titles: Set[str]) -> Tuple[str, Dict[str, Any]]:
            """阶段 2：生成和评估（不受限制）"""
            question = str(sample.get("question", ""))
            ground_truth = str(sample.get("answer", ""))

            retrieved_titles = [u.title for u in method_result.units]
            metrics = compute_retrieval_metrics(
                retrieved_titles=retrieved_titles,
                relevant_titles=relevant_titles,
                avg_context_len=method_result.stats.get("avg_len", context_len(method_result.units)),
                latency_ms=method_result.stats.get("time_ms", 0.0),
                expanded_nodes=method_result.stats.get("expanded_nodes", 0),
            )

            answer = self.generate_answer(question, method_result.units)
            semantic = self.judge_answer(question, ground_truth, answer, method_result.units)

            route_detail = method_result.stats.get("route_detail", {})
            result = {
                "id": sample.get("id"),
                "question": question,
                "answer": ground_truth,
                "type": sample.get("type"),
                "level": sample.get("level"),
                "relevant_titles": sorted(relevant_titles),
                "retrieved_titles": retrieved_titles,
                "retrieved_contexts": [u.content for u in method_result.units],
                "generated_answer": answer,
                "retrieval_metrics": metrics,
                "semantic_metrics": semantic,
                "stats": method_result.stats,
                "complexity_score": route_detail.get("complexity_score", self.complexity_scorer.compute(question).score),
                "route": method_result.stats.get("route", route_detail.get("route", "")),
            }

            return method_name, result

        cache_file = None
        if self.config.retrieval_cache_dir:
            cache_dir = Path(self.config.retrieval_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"retrieval_{desc.replace(' ', '_')}.pkl"

        retrieval_results: List[Tuple[str, pd.Series, MethodResult, Set[str]]] = []
        generation_workers = self.config.max_workers

        if self.config.skip_retrieval and cache_file and cache_file.exists():
            logger.info("从缓存加载检索结果：%s", cache_file)
            import pickle
            with open(cache_file, "rb") as f:
                retrieval_results = pickle.load(f)
            logger.info("已加载 %d 条检索结果", len(retrieval_results))
        elif self.config.max_workers > 1:
            total_tasks = len(samples) * len(method_names)
            retrieval_workers = min(50, self.config.max_workers)

            pbar = tqdm(total=total_tasks, desc=f"{desc} - 检索阶段")

            retrieval_lock = threading.Lock()

            with ThreadPoolExecutor(max_workers=retrieval_workers) as retrieval_executor:
                futures = {}
                for idx, sample in samples:
                    for method_name in method_names:
                        future = retrieval_executor.submit(retrieval_phase, idx, sample, method_name)
                        futures[future] = (idx, method_name)

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        with retrieval_lock:
                            retrieval_results.append(result)
                        pbar.update(1)
                    except Exception as exc:
                        idx, method_name = futures[future]
                        logger.error("检索阶段异常 (idx=%s, method=%s)：%s", idx, method_name, exc)

            pbar.close()

            if cache_file:
                logger.info("保存检索结果到缓存：%s", cache_file)
                import pickle
                with open(cache_file, "wb") as f:
                    pickle.dump(retrieval_results, f)
        else:
            for idx, sample in tqdm(samples, desc=f"{desc} - 检索阶段"):
                for method_name in method_names:
                    method_name_r, sample_r, method_result, relevant_titles = retrieval_phase(idx, sample, method_name)
                    retrieval_results.append((method_name_r, sample_r, method_result, relevant_titles))

        if not self.config.run_generation and not self.config.run_judge:
            logger.info("跳过生成阶段（run_generation=False, run_judge=False）")
            for method_name, sample, method_result, relevant_titles in retrieval_results:
                question = str(sample.get("question", ""))
                ground_truth = str(sample.get("answer", ""))
                retrieved_titles = [u.title for u in method_result.units]
                metrics = compute_retrieval_metrics(
                    retrieved_titles=retrieved_titles,
                    relevant_titles=relevant_titles,
                    avg_context_len=method_result.stats.get("avg_len", context_len(method_result.units)),
                    latency_ms=method_result.stats.get("time_ms", 0.0),
                    expanded_nodes=method_result.stats.get("expanded_nodes", 0),
                )
                route_detail = method_result.stats.get("route_detail", {})
                result = {
                    "id": sample.get("id"),
                    "question": question,
                    "answer": ground_truth,
                    "type": sample.get("type"),
                    "level": sample.get("level"),
                    "relevant_titles": sorted(relevant_titles),
                    "retrieved_titles": retrieved_titles,
                    "retrieved_contexts": [u.content for u in method_result.units],
                    "generated_answer": "",
                    "retrieval_metrics": metrics,
                    "semantic_metrics": {},
                    "stats": method_result.stats,
                    "complexity_score": route_detail.get("complexity_score", self.complexity_scorer.compute(question).score),
                    "route": method_result.stats.get("route", route_detail.get("route", "")),
                }
                with rows_lock:
                    rows_by_method[method_name].append(result)
            return rows_by_method

        if self.config.max_workers > 1:
            pbar2 = tqdm(total=len(retrieval_results), desc=f"{desc} - 生成阶段")

            with ThreadPoolExecutor(max_workers=generation_workers) as generation_executor:
                futures = {generation_executor.submit(generation_phase, *r): r for r in retrieval_results}

                for future in as_completed(futures):
                    try:
                        method_name, result = future.result()
                        with rows_lock:
                            rows_by_method[method_name].append(result)
                    except Exception as exc:
                        logger.error("生成阶段异常：%s", exc)

                    pbar2.update(1)

            pbar2.close()
        else:
            for method_name, sample, method_result, relevant_titles in tqdm(retrieval_results, desc=f"{desc} - 生成阶段"):
                method_name_g, result = generation_phase(method_name, sample, method_result, relevant_titles)
                with rows_lock:
                    rows_by_method[method_name].append(result)

        return rows_by_method

    # ------------------------------------------------------------------
    # 结果汇总
    # ------------------------------------------------------------------
    def summarize_methods(self, rows_by_method: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return [
            {"Method": method_name, **self.aggregate_method_rows(rows)}
            for method_name, rows in rows_by_method.items()
        ]

    @staticmethod
    def _mean_all_rows(values: Iterable[Optional[float]], total_rows: int) -> Optional[float]:
        """缺失评价按 0 计，避免仅统计成功调用造成指标虚高。"""
        if total_rows <= 0:
            return None
        nums = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
        return sum(nums) / total_rows

    def aggregate_method_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = [row["retrieval_metrics"] for row in rows]
        semantic = [row.get("semantic_metrics", {}) for row in rows]
        total_rows = len(rows)
        valid_judges = sum(1 for s in semantic if bool(s.get("_judge_valid")))
        refusals = sum(1 for s in semantic if bool(s.get("_is_refusal")))

        return {
            "Samples": total_rows,
            "Valid Judge Samples": valid_judges,
            "Refusal Rate": round2(refusals / total_rows if total_rows else 0.0) or 0.0,
            "Recall": round2(mean_or_none(m.recall for m in metrics)) or 0.0,
            "Precision": round2(mean_or_none(m.precision for m in metrics)) or 0.0,
            "MRR": round2(mean_or_none(m.mrr for m in metrics)) or 0.0,
            "NDCG": round2(mean_or_none(m.ndcg for m in metrics)) or 0.0,
            "MAP": round2(mean_or_none(m.map_score for m in metrics)) or 0.0,
            "Avg Len": round2(mean_or_none(m.avg_len for m in metrics)) or 0.0,
            "Time/ms": round2(mean_or_none(m.time_ms for m in metrics)) or 0.0,
            "Expanded Nodes": round2(mean_or_none(m.expanded_nodes for m in metrics)) or 0.0,
            "correctness": round2(self._mean_all_rows((s.get("correctness") for s in semantic), total_rows)),
            "faithfulness": round2(self._mean_all_rows((s.get("faithfulness") for s in semantic), total_rows)),
            "answer_relevance": round2(self._mean_all_rows((s.get("answer_relevance") for s in semantic), total_rows)),
            "context_relevance": round2(self._mean_all_rows((s.get("context_relevance") for s in semantic), total_rows)),
        }

    def semantic_record_table(self, rows_by_method: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        table: List[Dict[str, Any]] = []
        for method, rows in rows_by_method.items():
            semantic = [row.get("semantic_metrics", {}) for row in rows]
            total_rows = len(rows)
            record: Dict[str, Any] = {
                "Method": method,
                "Samples": total_rows,
                "Valid Judge Samples": sum(1 for s in semantic if bool(s.get("_judge_valid"))),
                "Refusal Rate": round2(sum(1 for s in semantic if bool(s.get("_is_refusal"))) / total_rows if total_rows else 0.0),
            }
            for key in ["correctness", "faithfulness", "answer_relevance", "context_relevance"]:
                values = [s.get(key) for s in semantic]
                record[key] = round2(self._mean_all_rows(values, total_rows))
                record[f"{key}_ci95_valid"] = round2(ci95(values))
            table.append(record)
        return table

    def type_breakdown_table(self, rows_by_method: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """按方法和 HotpotQA 问题类型汇总，重点观察 Bridge/Comparison。"""
        out: List[Dict[str, Any]] = []
        for method, rows in rows_by_method.items():
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for row in rows:
                grouped.setdefault(str(row.get("type") or "unknown"), []).append(row)
            for question_type, group_rows in sorted(grouped.items()):
                metrics = [row["retrieval_metrics"] for row in group_rows]
                semantic = [row.get("semantic_metrics", {}) for row in group_rows]
                total = len(group_rows)
                out.append(
                    {
                        "Method": method,
                        "Type": question_type,
                        "Samples": total,
                        "Recall": round2(mean_or_none(m.recall for m in metrics)) or 0.0,
                        "Correctness": round2(self._mean_all_rows((s.get("correctness") for s in semantic), total)),
                        "Faithfulness": round2(self._mean_all_rows((s.get("faithfulness") for s in semantic), total)),
                        "Context Relevance": round2(self._mean_all_rows((s.get("context_relevance") for s in semantic), total)),
                        "Avg Len": round2(mean_or_none(m.avg_len for m in metrics)) or 0.0,
                        "Time/ms": round2(mean_or_none(m.time_ms for m in metrics)) or 0.0,
                        "Expanded Nodes": round2(mean_or_none(m.expanded_nodes for m in metrics)) or 0.0,
                        "Refusal Rate": round2(sum(1 for s in semantic if bool(s.get("_is_refusal"))) / total if total else 0.0),
                    }
                )
        return out

    def route_distribution_table(self, proposed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按问题类型与路由统计触发比例、质量和成本，便于分析自适应行为。"""
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        type_totals: Dict[str, int] = {}

        for row in proposed_rows:
            question_type = str(row.get("type") or "unknown")
            route = str(row.get("route") or "unknown")
            grouped.setdefault((question_type, route), []).append(row)
            type_totals[question_type] = type_totals.get(question_type, 0) + 1

        out: List[Dict[str, Any]] = []
        for (question_type, route), rows in sorted(grouped.items()):
            metrics = [row["retrieval_metrics"] for row in rows]
            semantic = [row.get("semantic_metrics", {}) for row in rows]
            stats = [row.get("stats", {}) for row in rows]
            out.append(
                {
                    "Type": question_type,
                    "Route": route,
                    "Samples": len(rows),
                    "Trigger Rate": round2(len(rows) / max(type_totals.get(question_type, 0), 1)) or 0.0,
                    "Recall": round2(mean_or_none(m.recall for m in metrics)) or 0.0,
                    "Correctness": round2(mean_or_none(s.get("correctness") for s in semantic)),
                    "Avg Len": round2(mean_or_none(m.avg_len for m in metrics)) or 0.0,
                    "Expanded Nodes": round2(mean_or_none(m.expanded_nodes for m in metrics)) or 0.0,
                    "Graph Hops": round2(mean_or_none(s.get("graph_hops") for s in stats)) or 0.0,
                    "Budget Utilization": round2(mean_or_none(s.get("budget_utilization") for s in stats)) or 0.0,
                }
            )
        return out

    def complexity_stratified_table(self, proposed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按查询复杂度分层统计路由触发情况与检索效果。"""
        buckets = {
            "low": [],
            "medium": [],
            "high": [],
        }
        for row in proposed_rows:
            score = float(row.get("complexity_score", 0.0))
            if score < 0.45:
                buckets["low"].append(row)
            elif score < self.config.complexity_threshold:
                buckets["medium"].append(row)
            else:
                buckets["high"].append(row)

        out = []
        for level, rows in buckets.items():
            metrics = [row["retrieval_metrics"] for row in rows]
            graph_count = sum(1 for row in rows if row.get("route") == "graph_expansion")
            out.append(
                {
                    "Complexity": level,
                    "Samples": len(rows),
                    "Graph Trigger Rate": round2(graph_count / len(rows) if rows else 0.0) or 0.0,
                    "Avg Len": round2(mean_or_none(m.avg_len for m in metrics)) or 0.0,
                    "Recall": round2(mean_or_none(m.recall for m in metrics)) or 0.0,
                }
            )
        return out

    # ------------------------------------------------------------------
    # IO 辅助函数
    # ------------------------------------------------------------------
    def _create_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.output_dir / f"adaptive_multiscale_rag_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def get_relevant_titles(sample: pd.Series) -> Set[str]:
        supporting = sample.get("supporting_facts", {})
        if hasattr(supporting, "get"):
            titles = supporting.get("title", [])
        else:
            titles = []
        if isinstance(titles, np.ndarray):
            titles = titles.tolist()
        return set(str(t) for t in titles)

    @staticmethod
    def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @classmethod
    def _json_safe(cls, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, ExperimentMetrics):
            return asdict(obj)
        if isinstance(obj, EvidenceUnit):
            return asdict(obj)
        if isinstance(obj, dict):
            return {str(k): cls._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._json_safe(v) for v in obj]
        return obj

    # ------------------------------------------------------------------
    # 实验主流程
    # ------------------------------------------------------------------
    def run(
        self,
        full_only: bool = False,
        include_summary_ablation: bool = False,
    ) -> Dict[str, Any]:
        """运行完整模型及核心消融实验。"""
        self.load_resources()
        assert self.test_data is not None

        run_dir = self._create_run_dir()

        method_fns: Dict[str, Callable[[str], MethodResult]] = {
            "Full": lambda q: self.retrieve_adaptive(q, ablation_name="Full"),
        }
        if not full_only:
            method_fns.update(
                {
                    "w/o Structural Graph": lambda q: self.retrieve_adaptive(
                        q,
                        enable_parent=False,
                        ablation_name="w/o Structural Graph",
                    ),
                    "w/o Semantic Graph": lambda q: self.retrieve_adaptive(
                        q,
                        enable_graph_expansion=False,
                        ablation_name="w/o Semantic Graph",
                    ),
                    "w/o Adaptive Routing": lambda q: self.retrieve_adaptive(
                        q,
                        forced_route="graph_expansion",
                        forced_hops=1,
                        ablation_name="w/o Adaptive Routing",
                    ),
                    "w/o Budget Control": lambda q: self.retrieve_adaptive(
                        q,
                        enable_budget_control=False,
                        ablation_name="w/o Budget Control",
                    ),
                }
            )
            if include_summary_ablation:
                method_fns["w/o Summary"] = lambda q: self.retrieve_adaptive(
                    q,
                    enable_summary=False,
                    ablation_name="w/o Summary",
                )

        rows_by_method = self.evaluate_methods(method_fns, desc="自适应多尺度 RAG 消融实验")
        full_rows = rows_by_method.get("Full", [])

        summary = self.summarize_methods(rows_by_method)
        self.write_csv(run_dir / "ablation_summary.csv", summary)

        semantic_records = self.semantic_record_table(rows_by_method)
        self.write_csv(run_dir / "ablation_semantic_records.csv", semantic_records)

        type_rows = self.type_breakdown_table(rows_by_method)
        self.write_csv(run_dir / "ablation_by_type.csv", type_rows)

        complexity_rows = self.complexity_stratified_table(full_rows)
        self.write_csv(run_dir / "full_complexity_stratified.csv", complexity_rows)

        route_rows = self.route_distribution_table(full_rows)
        self.write_csv(run_dir / "full_route_distribution.csv", route_rows)

        details = {
            "rows_by_method": rows_by_method,
            "summary": summary,
            "semantic_records": semantic_records,
            "ablation_by_type": type_rows,
            "full_complexity_stratified": complexity_rows,
            "full_route_distribution": route_rows,
        }
        details_path = run_dir / "ablation_details.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(details), f, ensure_ascii=False, indent=2)

        config_path = run_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(asdict(self.config)), f, ensure_ascii=False, indent=2)

        logger.info("自适应多尺度 RAG 消融实验完成，结果目录：%s", run_dir)
        return {
            "run_dir": str(run_dir),
            "tables": {
                "ablation_summary": str(run_dir / "ablation_summary.csv"),
                "ablation_semantic_records": str(run_dir / "ablation_semantic_records.csv"),
                "ablation_by_type": str(run_dir / "ablation_by_type.csv"),
                "full_complexity_stratified": str(run_dir / "full_complexity_stratified.csv"),
                "full_route_distribution": str(run_dir / "full_route_distribution.csv"),
                "details": str(details_path),
            },
            "summary": summary,
            "ablation_by_type": type_rows,
        }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="自适应多尺度 RAG 主实验与消融实验")
    parser.add_argument("--sample-size", type=int, default=None, help="采样数量，默认使用配置值")
    parser.add_argument("--max-workers", type=int, default=None, help="并发线程数，默认使用配置值")
    parser.add_argument("--no-generation", action="store_true", help="跳过答案生成")
    parser.add_argument("--no-judge", action="store_true", help="跳过语义评估")
    parser.add_argument("--no-neo4j", action="store_true", help="允许 Neo4j 不可用时降级运行")
    parser.add_argument("--context-budget", type=int, default=None, help="最大上下文 token 预算")
    parser.add_argument("--complexity-threshold", type=float, default=None, help="复杂度阈值 Tc")
    parser.add_argument("--parent-threshold", type=float, default=None, help="父级/细粒度判定阈值 Tp")
    parser.add_argument("--fragmentation-threshold", type=float, default=None, help="碎片度阈值")
    parser.add_argument("--max-graph-seeds", type=int, default=None, help="语义图扩展最大种子数")
    parser.add_argument("--max-graph-neighbors", type=int, default=None, help="每个种子最大邻居数")
    parser.add_argument("--parent-window-size", type=int, default=None, help="命中句前后保留句数")
    parser.add_argument("--judge-max-retries", type=int, default=None, help="评价模型最大重试次数")
    parser.add_argument("--full-only", action="store_true", help="只运行完整方法，不运行消融")
    parser.add_argument("--include-summary-ablation", action="store_true", help="额外运行 w/o Summary")
    args = parser.parse_args()

    config_kwargs: Dict[str, Any] = {}
    if args.sample_size is not None:
        config_kwargs["sample_size"] = args.sample_size
    if args.max_workers is not None:
        config_kwargs["max_workers"] = args.max_workers
    if args.no_generation:
        config_kwargs["run_generation"] = False
    if args.no_judge:
        config_kwargs["run_judge"] = False
    if args.no_neo4j:
        config_kwargs["require_neo4j"] = False
    for arg_name in [
        "context_budget",
        "complexity_threshold",
        "parent_threshold",
        "fragmentation_threshold",
        "max_graph_seeds",
        "max_graph_neighbors",
        "parent_window_size",
        "judge_max_retries",
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            config_kwargs[arg_name] = value

    config = RetrievalConfig(**config_kwargs)
    result = AdaptiveMultiscaleRAGExperiment(config).run(
        full_only=args.full_only,
        include_summary_ablation=args.include_summary_ablation,
    )

    print("\n主实验/消融实验完成。")
    print(f"结果目录: {result['run_dir']}")
    for name, path in result["tables"].items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
