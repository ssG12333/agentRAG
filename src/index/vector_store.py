"""向量存储 —— numpy 内存索引（Phase 1），后续替换为 C++ IVF-PQ（Phase 2）"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.document import Chunk


class BaseVectorStore(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """添加向量和对应 chunk"""
        ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """检索 top_k 个最相似的 chunk 及分数"""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """持久化存储"""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """从持久化恢复"""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


class NumpyVectorStore(BaseVectorStore):
    """基于 numpy 的内存向量存储

    全量遍历搜索（Phase 1 简单实现），
    Phase 2 替换为 C++ IVF-PQ 索引。
    """

    def __init__(self):
        self._vectors: np.ndarray | None = None       # (n, dim)
        self._chunks: List[Chunk] = []
        self._normalized = False

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dim(self) -> int | None:
        if self._vectors is not None and self._vectors.shape[0] > 0:
            return int(self._vectors.shape[1])
        return None

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """添加向量和对应 chunk 到存储"""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # 存储时归一化，检索用余弦相似度 = dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # 避免零除
        normalized = embeddings / norms

        if self._vectors is None:
            self._vectors = normalized
        else:
            self._vectors = np.vstack([self._vectors, normalized])

        self._chunks.extend(chunks)
        self._normalized = True

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """全量余弦相似度检索"""
        if self._vectors is None or len(self._chunks) == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        # 归一化查询向量
        q_norm = np.linalg.norm(query)
        if q_norm > 0:
            query = query / q_norm

        # 余弦相似度 = dot(归一化向量)
        scores = np.dot(query, self._vectors.T)[0]  # (n,)

        # top-k
        if top_k >= len(scores):
            indices = np.argsort(scores)[::-1]
        else:
            indices = np.argpartition(scores, -top_k)[-top_k:]
            indices = indices[np.argsort(scores[indices])[::-1]]

        return [(self._chunks[i], float(scores[i])) for i in indices]

    def save(self, path: str) -> None:
        """保存向量和 chunk 元数据"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            vectors=self._vectors,
            chunk_ids=np.array([c.id for c in self._chunks]),
            chunk_contents=np.array([c.content for c in self._chunks]),
            chunk_doc_ids=np.array([c.document_id for c in self._chunks]),
        )

    def load(self, path: str) -> None:
        """从持久化恢复"""
        path = Path(path)
        if not path.with_suffix(".npz").exists() and not path.exists():
            raise FileNotFoundError(f"索引文件不存在: {path}")

        data = np.load(path, allow_pickle=True)
        self._vectors = data["vectors"]
        self._chunks = [
            Chunk(
                id=str(cid),
                document_id=str(did),
                content=str(content),
                metadata={},
            )
            for cid, did, content in zip(
                data["chunk_ids"], data["chunk_doc_ids"], data["chunk_contents"]
            )
        ]
        self._normalized = True
