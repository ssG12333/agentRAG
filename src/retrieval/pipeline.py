"""统一检索管道：稠密检索、BM25/RRF 混合检索和可选重排序。"""

from __future__ import annotations

from typing import List, Tuple

from src.document import Chunk
from src.embedding.model import BaseEmbedding
from src.index.hybrid import HybridRetriever
from src.index.sparse_store import SparseRetriever
from src.index.vector_store import BaseVectorStore
from src.retrieval.reranker import BaseReranker
from src.retrieval.retriever import Retriever


class RetrievalPipeline:
    """将 dense store、可选 BM25/RRF 和 reranker 组装为统一接口。"""

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        dense_store: BaseVectorStore,
        *,
        hybrid: bool = False,
        reranker: BaseReranker | None = None,
        rerank_top_n: int = 30,
        sparse_k1: float = 1.5,
        sparse_b: float = 0.75,
    ):
        if rerank_top_n <= 0:
            raise ValueError("rerank_top_n 必须为正数")

        self._dense = Retriever(embedding_model, dense_store)
        self._hybrid = None
        self._reranker = reranker
        self._rerank_top_n = rerank_top_n

        if hybrid:
            sparse = SparseRetriever(k1=sparse_k1, b=sparse_b)
            sparse.add_chunks(dense_store.chunks)
            self._hybrid = HybridRetriever(embedding_model, dense_store, sparse)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        if top_k <= 0:
            return []

        candidate_k = self._rerank_top_n if self._reranker else top_k
        candidate_k = max(candidate_k, top_k)
        if self._hybrid is not None:
            candidates = self._hybrid.search(query, top_k=candidate_k)
        else:
            candidates = self._dense.retrieve(query, top_k=candidate_k)

        if self._reranker is not None:
            candidates = self._reranker.rerank(query, candidates)
        return candidates[:top_k]

    @property
    def mode(self) -> str:
        base = "hybrid" if self._hybrid is not None else "dense"
        return f"{base}+rerank" if self._reranker is not None else base

    @property
    def store(self) -> BaseVectorStore:
        return self._dense.store
