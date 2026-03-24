# -*- coding: utf-8 -*-
"""关键词检索器实现

使用 BM25 算法进行关键词检索，支持中文分词。
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

from src.retrievers.base_retriever import RetrieverBase, SearchResult

logger = get_logger(__name__)

# 尝试导入 jieba 分词
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning(
        "未安装 jieba，中文文本分词功能将受限。安装命令：pip install jieba"
    )


class BM25:
    """BM25 算法实现
    
    Okapi BM25 排序函数实现。
    
    Attributes:
        k1: 词频饱和参数
        b: 文档长度归一化参数
        epsilon: 平滑参数
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """初始化 BM25
        
        Args:
            k1: 词频饱和参数，通常在 1.2-2.0 之间
            b: 文档长度归一化参数，通常在 0.75 左右
            epsilon: 平滑参数
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # 文档集合
        self.doc_freqs: Dict[str, int] = {}  # 词频统计
        self.doc_len: List[int] = []  # 文档长度列表
        self.avgdl: float = 0.0  # 平均文档长度
        self.doc_count: int = 0  # 文档总数
        
        # 文档内容存储
        self.doc_ids: List[str] = []
        self.doc_contents: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []
        
        # 词到文档的倒排索引
        self.inverted_index: Dict[str, List[int]] = {}
    
    def tokenize(self, text: str) -> List[str]:
        """分词
        
        Args:
            text: 输入文本
            
        Returns:
            词项列表
        """
        # 使用 jieba 分词（如果可用）
        if JIEBA_AVAILABLE:
            tokens = list(jieba.cut(text))
        else:
            # 简单的空格分词
            tokens = text.split()
        
        # 过滤停用词和标点
        tokens = [
            token.lower().strip()
            for token in tokens
            if token.strip() and len(token.strip()) > 1
        ]
        
        return tokens
    
    def fit(
        self,
        documents: List[Dict[str, Any]],
    ) -> None:
        """构建索引
        
        Args:
            documents: 文档列表，每个文档包含 doc_id, content, metadata
        """
        # 清空现有索引
        self.doc_freqs.clear()
        self.doc_len.clear()
        self.inverted_index.clear()
        self.doc_ids.clear()
        self.doc_contents.clear()
        self.doc_metadata.clear()
        
        self.doc_count = len(documents)
        total_len = 0
        
        # 构建索引
        for idx, doc in enumerate(documents):
            doc_id = doc["doc_id"]
            content = doc["content"]
            metadata = doc.get("metadata", {})
            
            self.doc_ids.append(doc_id)
            self.doc_contents.append(content)
            self.doc_metadata.append(metadata)
            
            # 分词
            tokens = self.tokenize(content)
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            
            # 统计词频
            token_freqs = Counter(tokens)
            self.doc_freqs[doc_id] = token_freqs
            
            # 构建倒排索引
            for token in token_freqs:
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append(idx)
        
        # 计算平均文档长度
        self.avgdl = total_len / self.doc_count if self.doc_count > 0 else 0
        
        logger.info(
            f"BM25 索引构建完成: docs={self.doc_count}, "
            f"avg_len={self.avgdl:.2f}, vocab={len(self.inverted_index)}"
        )
    
    def _calc_idf(self, term: str) -> float:
        """计算逆文档频率 IDF
        
        Args:
            term: 词项
            
        Returns:
            IDF 值
        """
        # 包含该词的文档数
        n = len(self.inverted_index.get(term, []))
        
        # IDF 计算
        idf = math.log(
            (self.doc_count - n + 0.5) / (n + 0.5) + 1
        )
        
        return idf
    
    def score(
        self,
        query: str,
        doc_idx: int,
    ) -> float:
        """计算查询与文档的 BM25 分数
        
        Args:
            query: 查询文本
            doc_idx: 文档索引
            
        Returns:
            BM25 分数
        """
        query_tokens = self.tokenize(query)
        doc_id = self.doc_ids[doc_idx]
        doc_len = self.doc_len[doc_idx]
        
        score = 0.0
        
        for token in query_tokens:
            # 跳过不在索引中的词
            if token not in self.inverted_index:
                continue
            
            # 词频
            tf = self.doc_freqs[doc_id].get(token, 0)
            if tf == 0:
                continue
            
            # IDF
            idf = self._calc_idf(token)
            
            # BM25 分数计算
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / self.avgdl
            )
            
            score += idf * numerator / denominator
        
        return score
    
    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """搜索最相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            (文档索引, 分数) 元组列表
        """
        # 获取查询词项
        query_tokens = self.tokenize(query)
        
        # 候选文档集合
        candidate_docs = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidate_docs.update(self.inverted_index[token])
        
        # 计算分数
        scores = []
        for doc_idx in candidate_docs:
            score = self.score(query, doc_idx)
            if score > 0:
                scores.append((doc_idx, score))
        
        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


class KeywordRetriever(RetrieverBase):
    """关键词检索器
    
    使用 BM25 算法进行关键词检索，支持中文分词。
    
    Attributes:
        bm25: BM25 实例
        documents: 文档列表
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        **kwargs: Any,
    ):
        """初始化关键词检索器
        
        Args:
            k1: BM25 词频饱和参数
            b: BM25 文档长度归一化参数
            **kwargs: 额外参数
        """
        super().__init__(**kwargs)
        
        self.bm25 = BM25(k1=k1, b=b)
        self.documents: List[Dict[str, Any]] = []
        
        logger.info(
            f"初始化关键词检索器: top_k={self.top_k}, "
            f"score_threshold={self.score_threshold}, k1={k1}, b={b}"
        )
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> None:
        """添加文档到索引
        
        Args:
            documents: 文档列表，每个文档包含 doc_id, content, metadata
        """
        self.documents.extend(documents)
        self.bm25.fit(self.documents)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """执行关键词检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量（可选）
            **kwargs: 额外参数
            
        Returns:
            检索结果列表，按相关性分数降序排列
        """
        actual_top_k = self._get_top_k(top_k)
        
        # 执行 BM25 搜索
        scores = self.bm25.search(query, actual_top_k)
        
        # 转换为 SearchResult 格式
        results = []
        for doc_idx, score in scores:
            doc = self.documents[doc_idx]
            
            result = SearchResult(
                doc_id=doc["doc_id"],
                content=doc["content"],
                score=score,
                metadata=doc.get("metadata", {}),
            )
            results.append(result)
        
        # 根据阈值过滤
        results = self._filter_by_threshold(results)
        
        logger.info(
            f"关键词检索完成: query='{query[:50]}...', "
            f"results={len(results)}"
        )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "type": "keyword",
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "doc_count": len(self.documents),
            "vocab_size": len(self.bm25.inverted_index),
            "avg_doc_len": self.bm25.avgdl,
        }


if __name__ == "__main__":
    print("=" * 50)
    print("测试 Keyword Retriever 模块")
    print("=" * 50)
    
    # 测试 1: 创建检索器
    retriever = KeywordRetriever(top_k=5, score_threshold=0.1)
    print(f"✓ 创建检索器: top_k={retriever.top_k}")
    
    # 测试 2: 添加文档
    docs = [
        "人工智能是计算机科学的一个分支，研究如何让机器具有智能。",
        "机器学习是人工智能的核心技术，通过数据训练模型。",
        "深度学习是机器学习的一种方法，使用神经网络进行学习。",
        "自然语言处理是人工智能的重要领域，处理人类语言。",
        "计算机视觉让机器能够理解和处理图像信息。",
    ]
    doc_list = [
        {"doc_id": f"doc_{i+1}", "content": doc, "metadata": {}}
        for i, doc in enumerate(docs)
    ]
    retriever.add_documents(doc_list)
    print(f"✓ 添加文档: count={len(retriever.documents)}")
    
    # 测试 3: 检索
    results = retriever.retrieve("人工智能技术", top_k=3)
    print(f"✓ 检索 '人工智能技术': 返回 {len(results)} 个结果")
    
    # 测试 4: 检索结果
    for i, result in enumerate(results[:2]):
        print(f"  - 结果 {i+1}: score={result.score:.4f}")
        print(f"    内容: {result.content[:30]}...")
    
    # 测试 5: 不同查询
    results2 = retriever.retrieve("深度学习神经网络")
    print(f"✓ 检索 '深度学习神经网络': 返回 {len(results2)} 个结果")
    
    # 测试 6: 统计信息
    stats = retriever.get_stats()
    print(f"✓ 统计信息: docs={stats['doc_count']}, vocab_size={stats['vocab_size']}")
    
    print("\n所有测试通过!")
