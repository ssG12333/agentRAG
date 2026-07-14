"""嵌入模型封装 —— 文本 → 向量"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List

import numpy as np


class BaseEmbedding(ABC):
    """嵌入模型抽象基类"""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """将文本列表转为嵌入向量，返回 (n, dim) 数组"""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """单条查询嵌入，返回 (dim,) 数组"""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度"""
        ...


class SentenceTransformerEmbedding(BaseEmbedding):
    """基于 sentence-transformers 的嵌入模型

    参数:
        model_name: HuggingFace 模型名
        device: "cpu" | "cuda"
        normalize: 是否 L2 归一化
    """

    def __init__(self, model_name: str, device: str = "cpu", normalize: bool = True):
        self._model_name = model_name
        self._device = device
        self._normalize = normalize

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "请安装 sentence-transformers: pip install sentence-transformers"
            )

        self._model = SentenceTransformer(model_name, device=device)

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    def __repr__(self) -> str:
        return f"SentenceTransformerEmbedding(model={self._model_name}, device={self._device})"


class CachedEmbedding(BaseEmbedding):
    """带 LRU 缓存的嵌入装饰器

    避免重复计算相同文本的嵌入向量。
    """

    def __init__(self, base: BaseEmbedding, max_size: int = 1024):
        self._base = base
        self._max_size = max_size
        self._cache: dict[str, np.ndarray] = {}
        self._hits = 0
        self._misses = 0

    @property
    def dim(self) -> int:
        return self._base.dim

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def embed(self, texts: List[str]) -> np.ndarray:
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text in self._cache:
                self._hits += 1
                results.append((i, self._cache[text]))
            else:
                self._misses += 1
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 批量计算未缓存的
        if uncached_texts:
            new_embeddings = self._base.embed(uncached_texts)
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                self._cache[text] = emb
                results.append((idx, emb))

            # LRU 淘汰
            while len(self._cache) > self._max_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]

        # 恢复原始顺序
        results.sort(key=lambda x: x[0])
        return np.stack([r[1] for r in results])

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    def clear_cache(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0
