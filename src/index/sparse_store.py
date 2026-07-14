"""
================================================================================
Layer 3: 稀疏向量存储 —— BM25 关键词检索
================================================================================

BM25 与稠密向量检索互补：
  - 稠密检索（IVF-PQ）：语义相似，"汽车"能找到"轿车"
  - 稀疏检索（BM25）：精确匹配，人名/术语/编号不容模糊

设计：
  - Phase 2 用 Python jieba 分词 + C++ BM25Index 检索
  - 分词器和索引分离，分词可替换（jieba → cppjieba）
  - 如果 C++ 模块不可用，回退到纯 Python BM25 实现
"""

from typing import List, Tuple

from src.document import Chunk


class SparseRetriever:
    """稀疏检索器：BM25 关键词匹配

    使用示例:
        store = SparseRetriever()
        store.add_chunks(chunks)
        results = store.search("Transformer 自注意力", top_k=5)

    Phase 2 实现：
      - 如果 C++ agentrag_core 可用 → C++ BM25Index（快）
      - 不可用 → Python 简易 BM25（慢但可工作）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: BM25 词频饱和度参数（默认 1.5）
            b:  文档长度归一化参数（默认 0.75）
        """
        self.k1 = k1
        self.b = b
        self._chunks: List[Chunk] = []  # doc_id → Chunk
        self._index = None               # C++ BM25Index 或 Python 实现

        # 尝试加载 C++ 模块
        try:
            from agentrag_core import BM25Index as CppBM25
            self._index = CppBM25()
            self._index.set_params(k1, b)
            self._use_cpp = True
        except ImportError:
            self._use_cpp = False
            # Python 回退：简易倒排列表
            self._py_posting: dict[str, List[Tuple[int, int]]] = {}  # term → [(doc_id, tf)]
            self._py_lengths: List[int] = []  # 文档长度
            self._py_avg_len: float = 0.0
            self._py_total_docs: int = 0

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """分词并添加到 BM25 索引"""
        import jieba

        start_id = len(self._chunks)

        for i, chunk in enumerate(chunks):
            doc_id = start_id + i
            # 分词: jieba.lcut 精确模式
            tokens = jieba.lcut(chunk.content)

            if self._use_cpp and self._index is not None:
                self._index.add_document(doc_id, tokens)
            else:
                self._py_add(doc_id, tokens)

            self._chunks.append(chunk)

    def _py_add(self, doc_id: int, tokens: List[str]) -> None:
        """Python 回退：添加到简易倒排索引"""
        from collections import Counter
        tf_count = Counter(tokens)

        for term, freq in tf_count.items():
            if term not in self._py_posting:
                self._py_posting[term] = []
            self._py_posting[term].append((doc_id, freq))

        self._py_lengths.append(len(tokens))
        self._py_total_docs += 1
        self._py_avg_len = sum(self._py_lengths) / self._py_total_docs

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """BM25 检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            [(Chunk, bm25_score), ...] 按分数降序
        """
        import jieba
        tokens = jieba.lcut(query)

        if self._use_cpp and self._index is not None:
            results = self._index.search(tokens, top_k)
        else:
            results = self._py_search(tokens, top_k)

        # 映射回 Chunk
        output = []
        for r in results:
            doc_id = r.id
            if doc_id < len(self._chunks):
                output.append((self._chunks[doc_id], r.score))
        return output

    def _py_search(self, tokens: List[str], top_k: int) -> List:
        """Python 回退：简易 BM25 检索"""
        import math
        scores: dict[int, float] = {}

        for term in tokens:
            if term not in self._py_posting:
                continue

            postings = self._py_posting[term]
            df = len(postings)

            # IDF
            N = self._py_total_docs
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id, tf in postings:
                length = self._py_lengths[doc_id]
                norm = 1.0 - self.b + self.b * (length / self._py_avg_len)
                score = idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        # 排序取 top_k
        class SR:
            def __init__(self, i, s):
                self.id = i
                self.score = s

        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [SR(doc_id, score) for doc_id, score in sorted_scores]

    def remove_chunks(self, doc_ids: List[int]) -> None:
        """删除指定 chunks"""
        if self._use_cpp and self._index is not None:
            for doc_id in doc_ids:
                self._index.remove_document(doc_id)

    def __len__(self) -> int:
        return len(self._chunks)
