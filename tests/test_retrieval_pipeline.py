"""统一检索管道和 CLI 后端选择测试。"""

from pathlib import Path

import numpy as np
from click.testing import CliRunner

from src.document import Chunk
from src.index.vector_store import NumpyVectorStore
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.reranker import BaseReranker


class MockEmbedding:
    dim = 3

    def __init__(self, query_vector=None):
        self.query_vector = np.asarray(
            query_vector if query_vector is not None else [1.0, 0.0, 0.0],
            dtype=np.float32,
        )

    def embed_query(self, query):
        return self.query_vector


def _dense_store():
    store = NumpyVectorStore()
    chunks = [
        Chunk(id="0", document_id="d0", content="普通水果介绍"),
        Chunk(id="1", document_id="d1", content="手机精确型号 X100 参数"),
        Chunk(id="2", document_id="d2", content="其他内容"),
    ]
    vectors = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    store.add(vectors, chunks)
    return store


def test_pipeline_dense_mode_keeps_phase1_behavior():
    pipeline = RetrievalPipeline(MockEmbedding(), _dense_store())

    results = pipeline.retrieve("水果", top_k=2)

    assert pipeline.mode == "dense"
    assert [chunk.id for chunk, _ in results] == ["0", "2"]


def test_pipeline_hybrid_promotes_exact_term():
    pipeline = RetrievalPipeline(
        MockEmbedding(),
        _dense_store(),
        hybrid=True,
    )

    results = pipeline.retrieve("X100", top_k=2)

    assert pipeline.mode == "hybrid"
    assert results[0][0].id == "1"


def test_pipeline_combines_ivfpq_bm25_and_rrf():
    from src.index.ivfpq_store import IVFPQVectorStore

    source = _dense_store()
    store = IVFPQVectorStore(
        n_clusters=1,
        n_probe=1,
        n_subvectors=1,
        n_bits=2,
        n_iters=8,
    )
    store.add(np.eye(3, dtype=np.float32), source.chunks)
    pipeline = RetrievalPipeline(MockEmbedding(), store, hybrid=True)

    results = pipeline.retrieve("X100", top_k=2)

    assert pipeline.mode == "hybrid"
    assert results[0][0].id == "1"


class ReverseReranker(BaseReranker):
    def __init__(self):
        self.candidate_count = 0

    def rerank(self, query, candidates):
        self.candidate_count = len(candidates)
        return list(reversed(candidates))


def test_pipeline_applies_explicit_reranker_and_final_limit():
    reranker = ReverseReranker()
    pipeline = RetrievalPipeline(
        MockEmbedding(),
        _dense_store(),
        reranker=reranker,
        rerank_top_n=3,
    )

    results = pipeline.retrieve("query", top_k=2)

    assert pipeline.mode == "dense+rerank"
    assert reranker.candidate_count == 3
    assert [chunk.id for chunk, _ in results] == ["1", "2"]


def test_numpy_store_preserves_chunk_metadata(tmp_path):
    store = NumpyVectorStore()
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="metadata test",
        metadata={"file_name": "source.md", "page": 3},
    )
    store.add(np.ones((1, 3), dtype=np.float32), [chunk])
    path = tmp_path / "index.npz"

    store.save(str(path))
    restored = NumpyVectorStore()
    restored.load(str(path))

    assert restored.chunks[0].metadata == chunk.metadata


def test_numpy_store_loads_legacy_index_without_metadata(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        vectors=np.ones((1, 3), dtype=np.float32),
        chunk_ids=np.asarray(["legacy-1"]),
        chunk_contents=np.asarray(["legacy content"]),
        chunk_doc_ids=np.asarray(["legacy-doc"]),
    )

    store = NumpyVectorStore()
    store.load(str(path))

    assert len(store) == 1
    assert store.chunks[0].metadata == {}


def test_backend_auto_detection():
    from src.cli.main import _resolve_backend, _resolve_index_path

    assert _resolve_backend("auto", "data/demo.ivfpq") == "ivfpq"
    assert _resolve_backend("auto", "data/demo.npz") == "numpy"
    assert _resolve_backend("numpy", "data/demo.ivfpq") == "numpy"
    assert _resolve_index_path(None, "ivfpq").endswith(".ivfpq")
    assert _resolve_index_path(None, "numpy").endswith(".npz")


def test_index_cli_builds_static_backend_once(monkeypatch, tmp_path):
    import src.cli.main as cli

    document_path = tmp_path / "docs"
    document_path.mkdir()
    (document_path / "sample.txt").write_text("第一段。第二段。", encoding="utf-8")
    output_path = tmp_path / "index.ivfpq"

    class FakeEmbedding:
        hit_rate = 0.0

        def embed(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32)

    class FakeStore:
        def __init__(self):
            self.add_calls = 0
            self.saved_path = None
            self._chunks = []

        def add(self, embeddings, chunks):
            self.add_calls += 1
            self._chunks = list(chunks)
            assert embeddings.shape == (len(chunks), 4)

        def save(self, path):
            self.saved_path = path

        def __len__(self):
            return len(self._chunks)

        @property
        def dim(self):
            return 4

    fake_store = FakeStore()
    selected = {}

    def fake_store_factory(backend, **kwargs):
        selected["backend"] = backend
        selected["kwargs"] = kwargs
        return fake_store

    monkeypatch.setattr(cli, "_get_embedding", lambda *args, **kwargs: FakeEmbedding())
    monkeypatch.setattr(cli, "_get_vector_store", fake_store_factory)

    result = CliRunner().invoke(cli.main, [
        "index",
        "--path", str(document_path),
        "--backend", "ivfpq",
        "--n-clusters", "4",
        "--n-probe", "2",
        "--n-subvectors", "2",
        "--save", str(output_path),
    ])

    assert result.exit_code == 0, result.output
    assert selected["backend"] == "ivfpq"
    assert selected["kwargs"]["n_subvectors"] == 2
    assert fake_store.add_calls == 1
    assert fake_store.saved_path == str(output_path)
