"""
================================================================================
Layer 4: 重排序器 —— Cross-Encoder 精细打分
================================================================================

为什么需要重排序？
  第一阶段（向量/BM25 检索）用的是双塔模型（Bi-Encoder）：
    - query 和 document 独立编码，然后点积
    - 速度快（O(1) 编码），但 query-doc 交互不充分

  第二阶段（Cross-Encoder 重排序）：
    - 将 (query, document) 拼接后送入 transformer 完整编码
    - query 和 doc 的每个 token 都可以互相 attend
    - 精度远高于双塔，但速度慢（每个 pair 都要跑一次模型）

  策略：
    - 第一阶段粗排取 50-100 个候选
    - 第二阶段精排取 top_k（如 5-10 个）送给 LLM

建议模型：
  - BAAI/bge-reranker-base：278M，适合中文
  - BAAI/bge-reranker-v2-m3：多语言，568M，更强但更慢

Phase 2 实现：
  - CrossEncoderReranker：基于 sentence-transformers 的 CrossEncoder
  - NoOpReranker：透传（Phase 1 使用，不做重排序）
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from src.document import Chunk


class BaseReranker(ABC):
    """重排序器抽象基类"""

    @abstractmethod
    def rerank(
        self, query: str, candidates: List[Tuple[Chunk, float]]
    ) -> List[Tuple[Chunk, float]]:
        """对候选列表重新排序

        Args:
            query: 用户查询
            candidates: [(chunk, old_score), ...] 第一阶段粗排结果

        Returns:
            [(chunk, new_score), ...] 重排序后的结果，按新分数降序
        """
        ...


class CrossEncoderReranker(BaseReranker):
    """基于 Cross-Encoder 的重排序器

    将 (query, chunk) pair 输入 Cross-Encoder，输出相关性 logit，
    logit 越高 = 越相关。

    使用示例:
        reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
        refined = reranker.rerank(query, coarse_results)
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model = None  # 延迟加载

    def _load_model(self):
        """延迟加载模型（首次调用时才加载）"""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name, device=self._device)
        except ImportError:
            raise ImportError(
                "重排序需要 sentence-transformers: pip install sentence-transformers"
            )

    def rerank(
        self, query: str, candidates: List[Tuple[Chunk, float]]
    ) -> List[Tuple[Chunk, float]]:
        if not candidates:
            return []

        self._load_model()

        # 构造 (query, doc) pair 列表
        pairs = [(query, chunk.content) for chunk, _score in candidates]

        # Cross-Encoder 推理
        scores = self._model.predict(pairs, show_progress_bar=False)

        # 按新分数重排
        reranked = [
            (chunk, float(score))
            for (chunk, _old), score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda x: -x[1])
        return reranked

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model={self._model_name})"


class NoOpReranker(BaseReranker):
    """空重排序器 —— 直接透传，Phase 1 使用"""

    def rerank(
        self, query: str, candidates: List[Tuple[Chunk, float]]
    ) -> List[Tuple[Chunk, float]]:
        return candidates
