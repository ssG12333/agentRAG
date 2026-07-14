"""检索器 —— 查询嵌入 → 向量搜索 → 返回结果"""

from typing import List, Tuple

from src.document import Chunk
from src.embedding.model import BaseEmbedding
from src.index.vector_store import BaseVectorStore


class Retriever:
    """基础检索器：嵌入查询 → 向量相似度搜索

    用法:
        retriever = Retriever(embedding_model, vector_store)
        chunks, scores = retriever.retrieve("什么是自注意力？", top_k=5)
    """

    def __init__(self, embedding_model: BaseEmbedding, vector_store: BaseVectorStore):
        self._embedding = embedding_model
        self._store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """检索 top_k 个最相关 chunk

        Returns:
            List[Tuple[Chunk, float]]: (chunk, 相似度分数) 列表，按分数降序
        """
        query_vec = self._embedding.embed_query(query)
        return self._store.search(query_vec, top_k=top_k)

    @property
    def store(self) -> BaseVectorStore:
        return self._store

    @property
    def embedding(self) -> BaseEmbedding:
        return self._embedding
