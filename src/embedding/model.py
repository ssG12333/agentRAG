"""
================================================================================
Layer 2: 嵌入模型 —— 文本 → 稠密向量
================================================================================

嵌入层是整个 RAG 系统的"眼睛"：将文本映射到固定维度的向量空间，
在这个空间中语义相似的文本距离近。

为什么自建而不直接用 sentence-transformers？
  - 统一接口：BaseEmbedding 抽象，方便后续替换为 ONNX / API
  - 缓存层：CachedEmbedding 避免重复计算（查询改写等场景常见）
  - 量化准备：Phase 4 将替换为 ONNX INT8 推理

支持的模型：
  - BAAI/bge-small-zh-v1.5  (24M, 512维, ~95MB)  —— 轻量调试
  - BAAI/bge-base-zh-v1.5   (102M, 768维, ~390MB) —— 标准使用
  - all-MiniLM-L6-v2         (22M, 384维, ~90MB)  —— 英文备选
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbedding(ABC):
    """嵌入模型抽象基类

    所有嵌入模型必须实现 embed() 和 embed_query() 两个方法。
    Phase 1 用 SentenceTransformer，Phase 4 替换为 ONNXEmbedding，
    上层代码无需任何改动。
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """批量嵌入：将文本列表转为 (n, dim) 的 numpy 数组

        Args:
            texts: 文本列表

        Returns:
            np.ndarray, shape=(len(texts), dim), dtype=float32
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """单条查询嵌入：返回 (dim,) 的一维数组

        与 embed([query])[0] 等价，但更语义化。
        有些模型（如 BGE）对 query 和 passage 有不同的处理策略。
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度"""
        ...


class SentenceTransformerEmbedding(BaseEmbedding):
    """基于 sentence-transformers 的嵌入模型

    这是 Phase 1 的主力实现。内部使用 HuggingFace 的 sentence-transformers 库，
    支持 BGE、E5、MiniLM 等主流嵌入模型。

    参数:
        model_name: HuggingFace 模型名，如 "BAAI/bge-small-zh-v1.5"
        device: "cpu" | "cuda:0" | "cuda"。GPU 推理速度约 5-10 倍于 CPU
        normalize: 是否 L2 归一化输出向量。
                   归一化后余弦相似度 = 内积，检索更快（不需要再次归一化）

    BGE 模型使用提示：
      - BGE 模型对 query 建议加前缀 "为这个句子生成表示以用于检索相关文章："
      - 本类默认不做此处理，在 Retriever 层统一管理
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
        # sentence-transformers 提供此属性
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        """批量嵌入。show_progress_bar=False 避免日志刷屏。"""
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
    """带 LRU（最近最少使用）缓存的嵌入装饰器

    实现透明缓存：对上层完全无感知，但能避免重复计算相同文本的嵌入。

    为什么需要缓存？
      - 查询改写场景：改写后的 query 可能和之前的 query 完全相同
      - 增量索引：只对新文档做嵌入，已索引文档遇到相同内容时走缓存
      - RAG Agent 多步推理：同一检索结果可能被多次查询

    缓存策略：
      - 字典存储 text → vector 映射
      - 超过 max_size 时，删除最早插入的条目（近似 FIFO）

    使用:
        base = SentenceTransformerEmbedding("BAAI/bge-small-zh-v1.5")
        cached = CachedEmbedding(base, max_size=1024)
        # 后续使用 cached 对象，自动缓存
        # 通过 cached.hit_rate 查看缓存命中率
    """

    def __init__(self, base: BaseEmbedding, max_size: int = 1024):
        self._base = base                  # 被装饰的实际嵌入模型
        self._max_size = max_size           # 最大缓存条目数
        self._cache: dict[str, np.ndarray] = {}  # text → vector
        self._hits = 0                      # 缓存命中次数
        self._misses = 0                    # 缓存未命中次数

    @property
    def dim(self) -> int:
        return self._base.dim

    @property
    def hit_rate(self) -> float:
        """缓存命中率：0.0 ~ 1.0。用于评估缓存是否有效。"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def embed(self, texts: List[str]) -> np.ndarray:
        """带缓存的批量嵌入

        算法：
          1. 遍历输入，命中缓存的直接取出
          2. 未命中的收集起来，批量调用底层模型
          3. 将新结果存入缓存（可能触发 LRU 淘汰）
          4. 按原始顺序拼接所有结果返回
        """
        results = []           # [(原始索引, 向量)]
        uncached_texts = []    # 需要计算的文本
        uncached_indices = []  # 对应的原始索引

        # 第一步：分离命中/未命中
        for i, text in enumerate(texts):
            if text in self._cache:
                self._hits += 1
                results.append((i, self._cache[text]))
            else:
                self._misses += 1
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 第二步：批量计算未命中的
        if uncached_texts:
            new_embeddings = self._base.embed(uncached_texts)
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                self._cache[text] = emb
                results.append((idx, emb))

            # LRU 淘汰：超过容量限制时，删除最早插入的条目
            while len(self._cache) > self._max_size:
                oldest_key = next(iter(self._cache))  # dict 按插入顺序迭代
                del self._cache[oldest_key]

        # 第三步：按原始顺序排列（因为 may out of order due to cache hits）
        results.sort(key=lambda x: x[0])
        return np.stack([r[1] for r in results])

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    def clear_cache(self) -> None:
        """清空缓存和统计"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
