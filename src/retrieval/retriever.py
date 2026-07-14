"""
================================================================================
Layer 4: 检索器 —— 连接嵌入层和索引层，返回相关文档块
================================================================================

检索器是 "用户问题 → 相关文档" 的桥梁：
  1. 嵌入模型将问题转为查询向量
  2. 向量存储检索 top_k 个相似块
  3. 返回 [(Chunk, score), ...]

Phase 1 只做单路稠密检索（dense only）。
Phase 2 升级为混合检索：稠密(IVF-PQ) + 稀疏(BM25) → RRF 融合 → 重排序。
"""

from typing import List, Tuple

from src.document import Chunk
from src.embedding.model import BaseEmbedding
from src.index.vector_store import BaseVectorStore


class Retriever:
    """基础检索器

    将嵌入模型和向量存储组装在一起，提供简洁的检索接口。

    使用示例:
        embedder = SentenceTransformerEmbedding("BAAI/bge-small-zh-v1.5")
        store = NumpyVectorStore()
        # ... 索引文档到 store ...
        retriever = Retriever(embedder, store)
        chunks = retriever.retrieve("什么是 Transformer？", top_k=5)
    """

    def __init__(self, embedding_model: BaseEmbedding, vector_store: BaseVectorStore):
        self._embedding = embedding_model  # 嵌入模型：文本 → 向量
        self._store = vector_store         # 向量存储：向量 → 相似块

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """检索与查询最相关的 top_k 个文档块

        流程:
          query ("什么是自注意力？")
            → embed_query → query_vector (512,)
            → store.search → [(chunk_0, 0.92), (chunk_1, 0.87), ...]

        Args:
            query: 用户查询字符串
            top_k: 返回的 chunk 数量

        Returns:
            [(Chunk, score), ...] 列表，按相似度分数降序
            score 范围 [-1, 1]，1 表示完全相似（嵌入已归一化时）
        """
        query_vec = self._embedding.embed_query(query)
        return self._store.search(query_vec, top_k=top_k)

    @property
    def store(self) -> BaseVectorStore:
        """暴露底层向量存储（供 CLI 查询统计信息）"""
        return self._store

    @property
    def embedding(self) -> BaseEmbedding:
        """暴露底层嵌入模型"""
        return self._embedding
