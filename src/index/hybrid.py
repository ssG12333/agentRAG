"""
================================================================================
Layer 4: 混合检索 —— RRF (Reciprocal Rank Fusion) 融合稠密+稀疏结果
================================================================================

为什么要混合检索？
  - 单一稠密检索会漏掉精确匹配（人名、编号、API 名等）
  - 单一稀疏检索会漏掉语义相近但用词不同的内容
  - 两者结合通常 Recall@10 提升 5-15%

RRF 算法：
  对每个 chunk，计算其在各检索器中的排名倒数之和：
    RRF(chunk) = sum(1 / (k + rank_i))
  其中 k=60 是平滑常数，rank_i 是 chunk 在第 i 个检索器中的排名（1-based）。

为什么用 RRF 而不是线性加权？
  - 稠密分数的范围是 [-1, 1]（余弦相似度）
  - BM25 分数的范围是 [0, +∞)
  - 两种分数的量纲和分布完全不同，直接加权毫无意义
  - RRF 只依赖排名，天然对量纲不敏感

参考：
  Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet and
  individual Rank Learning Methods" (SIGIR 2009)
"""

from typing import List, Tuple

from src.document import Chunk
from src.index.vector_store import BaseVectorStore
from src.index.sparse_store import SparseRetriever
from src.embedding.model import BaseEmbedding


class HybridRetriever:
    """混合检索器：稠密 + 稀疏 → RRF 融合

    使用示例:
        dense_store = NumpyVectorStore()  # 或 C++ IVF-PQ
        sparse_store = SparseRetriever()
        embedder = SentenceTransformerEmbedding(...)

        hybrid = HybridRetriever(embedder, dense_store, sparse_store)
        results = hybrid.search("Transformer 复杂度", top_k=5)
    """

    # RRF 平滑常数：防止单个检索器的高排名完全主导结果
    # k=60 是经典值（来自原始 RRF 论文）
    RRF_K = 60

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        dense_store: BaseVectorStore,
        sparse_store: SparseRetriever,
    ):
        self._embedding = embedding_model
        self._dense = dense_store
        self._sparse = sparse_store

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """混合检索

        流程:
          1. 稠密检索 → top_n_dense 个结果（默认取 2*top_k 以覆盖更多候选）
          2. 稀疏检索 → top_n_sparse 个结果
          3. RRF 融合：同一 chunk 在两边都出现则分数累加
          4. 按 RRF 分数排序取 top_k
        """
        # 粗排候选池大小（比 top_k 大，为融合保留空间）
        n_candidates = max(top_k * 3, 30)

        # 1. 稠密检索
        query_vec = self._embedding.embed_query(query)
        dense_results = self._dense.search(query_vec, top_k=n_candidates)

        # 2. 稀疏检索
        sparse_results = self._sparse.search(query, top_k=n_candidates)

        # 3. RRF 融合
        # chunk_id → RRF 累积分
        rrf_scores: dict[str, float] = {}
        # chunk_id → Chunk 对象（取首次出现的）
        chunk_map: dict[str, Chunk] = {}

        # 处理稠密结果：rank=1 是最佳
        for rank, (chunk, _dense_score) in enumerate(dense_results):
            chunk_id = chunk.id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            chunk_map[chunk_id] = chunk

        # 处理稀疏结果
        for rank, (chunk, _sparse_score) in enumerate(sparse_results):
            chunk_id = chunk.id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk

        # 4. 排序取 top_k
        sorted_items = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
        return [(chunk_map[cid], score) for cid, score in sorted_items]
