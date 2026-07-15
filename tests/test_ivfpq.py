"""残差 IVF-PQ C++ 索引与 Python VectorStore 测试。"""

import numpy as np
import pytest

from src.document import Chunk
from src.index.ivfpq_store import IVFPQVectorStore


def _clustered_vectors(seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = np.eye(4, 8, dtype=np.float32)
    vectors = []
    chunks = []
    for cluster, center in enumerate(centers):
        for item in range(12):
            vector = center + rng.normal(0, 0.02, size=8).astype(np.float32)
            vectors.append(vector)
            chunks.append(Chunk(
                id=f"{cluster}-{item}",
                document_id=f"doc-{cluster}",
                content=f"cluster {cluster} item {item}",
                metadata={"cluster": cluster},
            ))
    return np.asarray(vectors, dtype=np.float32), chunks, centers


def _build_store():
    vectors, chunks, centers = _clustered_vectors()
    store = IVFPQVectorStore(
        n_clusters=4,
        n_probe=4,
        n_subvectors=2,
        n_bits=4,
        n_iters=12,
    )
    store.add(vectors, chunks)
    return store, vectors, centers


def test_ivfpq_retrieves_correct_clusters():
    store, _, centers = _build_store()

    for cluster, query in enumerate(centers):
        results = store.search(query, top_k=5)
        assert len(results) == 5
        assert all(chunk.metadata["cluster"] == cluster for chunk, _ in results)
        assert all(results[i][1] >= results[i + 1][1] for i in range(4))


def test_ivfpq_compresses_vector_payload():
    store, vectors, _ = _build_store()

    assert store.codes_bytes == len(vectors) * 2
    assert store.estimated_memory_bytes < vectors.nbytes


def test_ivfpq_persistence_round_trip(tmp_path):
    store, _, centers = _build_store()
    index_path = tmp_path / "cluster.ivfpq"
    expected = store.search(centers[2], top_k=5)

    store.save(str(index_path))
    restored = IVFPQVectorStore(
        n_clusters=1,
        n_probe=1,
        n_subvectors=1,
        n_bits=1,
    )
    restored.load(str(index_path))
    actual = restored.search(centers[2], top_k=5)

    assert len(restored) == len(store)
    assert restored.dim == store.dim
    assert [chunk.id for chunk, _ in actual] == [
        chunk.id for chunk, _ in expected
    ]
    assert actual[0][0].metadata["cluster"] == 2


def test_ivfpq_rejects_invalid_usage():
    vectors, chunks, _ = _clustered_vectors()
    store = IVFPQVectorStore(
        n_clusters=4,
        n_probe=2,
        n_subvectors=2,
        n_bits=4,
    )

    with pytest.raises(ValueError, match="数量"):
        store.add(vectors, chunks[:-1])

    store.add(vectors, chunks)
    with pytest.raises(RuntimeError, match="增量"):
        store.add(vectors, chunks)
    with pytest.raises(ValueError, match="查询维度"):
        store.search(np.ones(4, dtype=np.float32))


def test_product_quantizer_rejects_codes_wider_than_uint8():
    import agentrag_core

    pq = agentrag_core.ProductQuantizer()
    with pytest.raises(ValueError, match="n_bits"):
        pq.train(np.ones((4, 8), dtype=np.float32), n_subvectors=2, n_bits=9)
