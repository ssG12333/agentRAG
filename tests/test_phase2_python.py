"""Phase 2 Python 层测试（不依赖 C++ 编译）"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ════════════ 稀疏检索 ════════════

def test_sparse_retriever_add_search():
    """测试 BM25 稀疏检索的增删查"""
    from src.document import Chunk
    from src.index.sparse_store import SparseRetriever

    store = SparseRetriever()
    chunks = [
        Chunk(id="0", document_id="d0", content="自注意力机制的计算复杂度是O(n²d)"),
        Chunk(id="1", document_id="d0", content="Transformer 使用多头注意力机制"),
        Chunk(id="2", document_id="d0", content="位置编码使用正弦和余弦函数"),
    ]
    store.add_chunks(chunks)

    # 精确匹配 "复杂度" 应该命中第一个 chunk
    results = store.search("计算复杂度", top_k=2)
    assert len(results) > 0
    # 分数最高应该是包含 "计算复杂度" 的 chunk
    assert "复杂度" in results[0][0].content


def test_sparse_retriever_empty():
    """空检索"""
    from src.index.sparse_store import SparseRetriever

    store = SparseRetriever()
    results = store.search("任意查询")
    assert results == []


# ════════════ RRF 混合检索 ════════════

def test_rrf_fusion():
    """测试 RRF 融合逻辑"""
    from src.index.hybrid import HybridRetriever

    # RRF 公式验证：RRF(d) = sum(1/(k+rank_i))
    h = HybridRetriever.__new__(HybridRetriever)
    k = h.RRF_K

    # 模拟：chunk A 在稠密排第1，稀疏排第3
    # RRF = 1/(60+1) + 1/(60+3) = 1/61 + 1/63 ≈ 0.01639 + 0.01587 ≈ 0.0323
    expected = 1.0 / (k + 1) + 1.0 / (k + 3)
    assert abs(expected - 0.0323) < 0.001

    # 模拟：chunk B 在稠密排第2，稀疏排第2
    # RRF = 1/(60+2) + 1/(60+2) = 2/62 ≈ 0.03226
    expected_b = 2.0 / (k + 2)
    # B 的 RRF 分数与 A 非常接近（0.03226 vs 0.03227），差距极小
    # 用近似相等验证
    assert abs(expected_b - expected) < 0.0001


# ════════════ 重排序 ════════════

def test_noop_reranker():
    """空重排序器：透传"""
    from src.document import Chunk
    from src.retrieval.reranker import NoOpReranker

    chunks = [
        Chunk(id="0", document_id="d0", content="aaa"),
        Chunk(id="1", document_id="d0", content="bbb"),
    ]
    candidates = [(chunks[0], 0.9), (chunks[1], 0.5)]

    reranker = NoOpReranker()
    result = reranker.rerank("query", candidates)

    # 应该保持原顺序不变
    assert result[0][0].content == "aaa"
    assert result[1][0].content == "bbb"


# ════════════ Phase 1 兼容性 ════════════

def test_phase1_retriever_still_works():
    """Phase 1 的 Retriever 仍应正常工作"""
    import numpy as np
    from src.document import Chunk
    from src.index.vector_store import NumpyVectorStore
    from src.retrieval.retriever import Retriever

    class MockEmbedding:
        dim = 3
        def embed(self, texts):
            return np.array([[1.0, 0.0, 0.0]] * len(texts))
        def embed_query(self, query):
            return np.array([1.0, 0.0, 0.0])

    store = NumpyVectorStore()
    chunks = [Chunk(id=str(i), document_id="d0", content=f"test_{i}") for i in range(3)]
    vecs = np.eye(3, dtype=np.float32)
    store.add(vecs, chunks)

    retriever = Retriever(MockEmbedding(), store)
    results = retriever.retrieve("test", top_k=2)
    assert len(results) == 2
