"""C++ 残差 IVF-PQ 索引的 Python VectorStore 适配层。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.document import Chunk
from src.index.vector_store import BaseVectorStore


def _load_cpp_module():
    try:
        import agentrag_core
    except ImportError as exc:
        raise RuntimeError(
            "IVFPQVectorStore 需要已编译的 agentrag_core；"
            "请先运行 scripts/build_cpp.bat"
        ) from exc
    if not hasattr(agentrag_core, "IVFPQIndex"):
        raise RuntimeError(
            "当前 agentrag_core 未导出 IVFPQIndex；请重新编译 C++ 扩展"
        )
    return agentrag_core


class IVFPQVectorStore(BaseVectorStore):
    """一次性构建的残差 IVF-PQ 向量存储。

    索引只保存 PQ 编码，不在 Python 层保留原始 float32 向量。
    当前版本不支持增量添加；需要新增文档时应重建索引。
    """

    def __init__(
        self,
        n_clusters: int = 256,
        n_probe: int = 8,
        n_subvectors: int = 64,
        n_bits: int = 8,
        n_iters: int = 25,
    ):
        if min(n_clusters, n_probe, n_subvectors, n_bits, n_iters) <= 0:
            raise ValueError("IVF-PQ 参数必须为正数")
        if n_bits > 8:
            raise ValueError("n_bits 必须小于等于 8")

        self._cpp = _load_cpp_module()
        self._index = self._cpp.IVFPQIndex()
        self._chunks: List[Chunk] = []
        self._n_clusters = n_clusters
        self._n_probe = n_probe
        self._n_subvectors = n_subvectors
        self._n_bits = n_bits
        self._n_iters = n_iters

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dim(self) -> int | None:
        return int(self._index.dim) if len(self) else None

    @property
    def estimated_memory_bytes(self) -> int:
        return int(self._index.estimated_memory_bytes) if len(self) else 0

    @property
    def codes_bytes(self) -> int:
        return int(self._index.codes_bytes) if len(self) else 0

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if len(self):
            raise RuntimeError("IVFPQVectorStore 暂不支持增量添加，请重建索引")

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
            raise ValueError("embeddings 必须是非空二维数组")
        if vectors.shape[0] != len(chunks):
            raise ValueError("embeddings 数量必须与 chunks 数量一致")
        if vectors.shape[1] % self._n_subvectors != 0:
            raise ValueError("向量维度必须能被 n_subvectors 整除")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = np.ascontiguousarray(
            vectors / np.where(norms == 0, 1.0, norms),
            dtype=np.float32,
        )
        self._index.build(
            normalized,
            n_clusters=min(self._n_clusters, len(chunks)),
            n_probe=min(self._n_probe, self._n_clusters, len(chunks)),
            n_subvectors=self._n_subvectors,
            n_bits=self._n_bits,
            n_iters=self._n_iters,
        )
        self._chunks = list(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        if not len(self) or top_k <= 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(
                f"查询维度 {query.shape[0]} 与索引维度 {self.dim} 不一致"
            )
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        query = np.ascontiguousarray(query, dtype=np.float32)

        results = self._index.search(query, min(top_k, len(self)))
        return [
            (self._chunks[result.id], 1.0 / (1.0 + float(result.score)))
            for result in results
        ]

    def save(self, path: str) -> None:
        if not len(self):
            raise RuntimeError("不能保存空的 IVF-PQ 索引")

        index_path = Path(path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = Path(f"{index_path}.meta.npz")
        self._index.save(str(index_path))
        np.savez_compressed(
            metadata_path,
            chunk_ids=np.asarray([chunk.id for chunk in self._chunks]),
            chunk_document_ids=np.asarray(
                [chunk.document_id for chunk in self._chunks]
            ),
            chunk_contents=np.asarray([chunk.content for chunk in self._chunks]),
            chunk_metadata=np.asarray([
                json.dumps(chunk.metadata, ensure_ascii=False, default=str)
                for chunk in self._chunks
            ]),
        )

    def load(self, path: str) -> None:
        index_path = Path(path)
        metadata_path = Path(f"{index_path}.meta.npz")
        if not index_path.exists():
            raise FileNotFoundError(f"IVF-PQ 索引不存在: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"IVF-PQ 元数据不存在: {metadata_path}")

        index = self._cpp.IVFPQIndex()
        index.load(str(index_path))
        with np.load(metadata_path, allow_pickle=False) as data:
            required = {
                "chunk_ids",
                "chunk_document_ids",
                "chunk_contents",
                "chunk_metadata",
            }
            if not required.issubset(data.files):
                raise ValueError("IVF-PQ 元数据字段不完整")
            lengths = {len(data[name]) for name in required}
            if lengths != {len(index)}:
                raise ValueError("IVF-PQ 索引与 chunk 元数据数量不一致")

            chunks = [
                Chunk(
                    id=str(chunk_id),
                    document_id=str(document_id),
                    content=str(content),
                    metadata=json.loads(str(metadata)),
                )
                for chunk_id, document_id, content, metadata in zip(
                    data["chunk_ids"],
                    data["chunk_document_ids"],
                    data["chunk_contents"],
                    data["chunk_metadata"],
                )
            ]

        self._index = index
        self._chunks = chunks
        self._n_clusters = int(index.num_clusters)
        self._n_probe = int(index.n_probe)
        self._n_subvectors = int(index.n_subvectors)
        self._n_bits = int(index.n_bits)
