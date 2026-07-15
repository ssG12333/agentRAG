"""
================================================================================
Layer 3: 向量存储 —— 存储嵌入向量并执行相似度检索
================================================================================

向量存储是 RAG 的"记忆"：将文档块对应的嵌入向量存下来，
查询时找出与问题向量最相似的 top-k 个块。

Phase 1 实现（NumpyVectorStore）：
  - 全量遍历 + 余弦相似度，O(n*d) 复杂度
  - 适合 < 10,000 个 chunk 的场景
  - 支持 np.savez 持久化到磁盘

Phase 2 升级（C++ IVF-PQ）：
  - K-Means 粗量化 + 乘积量化压缩
  - 索引构建 O(n*k*iter)，检索 O(n_probe * n/k * d)
  - 适合 > 100,000 个 chunk 的场景

为什么自己去实现而不是用 FAISS？
  - 理解 IVF 倒排索引的内部原理
  - 理解 PQ 乘积量化的码本训练和距离表查询
  - 可以针对中文文本做特定优化
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.document import Chunk


class BaseVectorStore(ABC):
    """向量存储抽象基类

    所有向量存储实现（numpy 全遍历 / C++ IVF-PQ）都遵循此接口。
    上层 Retriever 不关心底层怎么存的，只调 add() 和 search()。
    """

    @abstractmethod
    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """添加向量和对应文本块到存储

        Args:
            embeddings: (n, dim) 的 numpy 数组，已归一化
            chunks: 对应的 n 个 Chunk 对象
        """
        ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """检索与查询向量最相似的 top_k 个块

        Args:
            query_vector: (dim,) 的一维查询向量
            top_k: 返回数量

        Returns:
            [(Chunk, score), ...] 列表，按分数降序
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """持久化到磁盘（Phase 1 用 np.savez）"""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """从磁盘恢复"""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """存储中向量的数量"""
        ...

    @property
    @abstractmethod
    def chunks(self) -> List[Chunk]:
        """返回与索引 ID 对齐的 chunk 列表。"""
        ...


class NumpyVectorStore(BaseVectorStore):
    """基于 numpy 的内存向量存储

    内部存储：
      - _vectors: (n, dim) float32 数组，存储时已归一化
      - _chunks:  对应的 Chunk 对象列表

    检索算法：
      - 查询向量也做归一化
      - 余弦相似度 = 归一化向量的内积（dot product）
      - 全量计算 → argpartition 取 top_k（比 argsort 快，O(n) 选 top-k vs O(n log n) 全排序）
    """

    def __init__(self):
        self._vectors: np.ndarray | None = None   # (n, dim) 已归一化向量
        self._chunks: List[Chunk] = []            # 对应文本块
        self._normalized = False

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> List[Chunk]:
        return self._chunks

    @property
    def dim(self) -> int | None:
        """向量维度"""
        if self._vectors is not None and self._vectors.shape[0] > 0:
            return int(self._vectors.shape[1])
        return None

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """添加向量和 chunk

        注意：不会检查是否重复添加同一个 chunk（调用方应去重）。
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # L2 归一化：每个向量除以其模长。
        # 归一化后，余弦相似度 = 向量内积，避免检索时重复计算模长。
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # 避免除以零（零向量的模长为 0，直接保留零向量）
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings / norms

        if self._vectors is None:
            self._vectors = normalized
        else:
            # np.vstack：纵向拼接
            self._vectors = np.vstack([self._vectors, normalized])

        self._chunks.extend(chunks)
        self._normalized = True

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """全量余弦相似度检索

        步骤：
          1. 归一化查询向量
          2. 计算 query 与所有存储向量的内积（= 余弦相似度）
          3. np.argpartition 选 top_k（比 np.argsort 快，尤其 n 大时）
          4. 对 top_k 结果排序
        """
        if self._vectors is None or len(self._chunks) == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        # 归一化查询向量
        q_norm = np.linalg.norm(query)
        if q_norm > 0:
            query = query / q_norm

        # (1, dim) @ (n, dim).T → (1, n) → (n,)
        # 因为向量已归一化，内积 = 余弦相似度
        scores = np.dot(query, self._vectors.T)[0]

        # 用 argpartition 取 top_k：
        #   - 当 top_k << n 时，argpartition 比 argsort 快很多
        #   - 它不保证 top_k 内部有序，但第 top_k 个元素的位置是正确的
        #   - 然后只对 top_k 个元素排序
        if top_k >= len(scores):
            indices = np.argsort(scores)[::-1]  # 降序
        else:
            # 负号 tricks：argpartition 默认升序，用负号实现降序
            indices = np.argpartition(scores, -top_k)[-top_k:]
            # 对 top_k 内部排序
            indices = indices[np.argsort(scores[indices])[::-1]]

        return [(self._chunks[i], float(scores[i])) for i in indices]

    def save(self, path: str) -> None:
        """持久化向量索引到 .npz 文件

        存储内容：
          - vectors:  (n, dim) float32
          - chunk_ids / chunk_contents / chunk_doc_ids: 元数据
          - metadata 以 JSON 字符串持久化
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            vectors=self._vectors,
            chunk_ids=np.array([c.id for c in self._chunks]),
            chunk_contents=np.array([c.content for c in self._chunks]),
            chunk_doc_ids=np.array([c.document_id for c in self._chunks]),
            chunk_metadata=np.array([
                json.dumps(c.metadata, ensure_ascii=False, default=str)
                for c in self._chunks
            ]),
        )

    def load(self, path: str) -> None:
        """从 .npz 文件恢复索引

        兼容旧索引：缺少 chunk_metadata 字段时使用空字典。
        """
        path = Path(path)
        # 兼容传入带或不带 .npz 后缀的路径
        if not path.exists():
            candidate = path.with_suffix(".npz")
            if not candidate.exists():
                raise FileNotFoundError(f"索引文件不存在: {path}")
            path = candidate

        with np.load(path, allow_pickle=False) as data:
            self._vectors = data["vectors"].copy()
            metadata_values = (
                data["chunk_metadata"]
                if "chunk_metadata" in data.files
                else ["{}"] * len(data["chunk_ids"])
            )
            self._chunks = [
                Chunk(
                    id=str(cid),
                    document_id=str(did),
                    content=str(content),
                    metadata=json.loads(str(metadata)),
                )
                for cid, did, content, metadata in zip(
                    data["chunk_ids"],
                    data["chunk_doc_ids"],
                    data["chunk_contents"],
                    metadata_values,
                )
            ]
        self._normalized = True
