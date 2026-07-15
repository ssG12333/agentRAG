"""快速离线课程实验：只复用生产模块，不下载模型。"""

from __future__ import annotations

import argparse
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from src.document import Chunk, Document
from src.document.chunker import FixedWindowChunker, RecursiveChunker
from src.embedding.model import BaseEmbedding, CachedEmbedding
from src.generation.engine import MockLLM
from src.generation.prompt import RAGPromptBuilder
from src.index.vector_store import NumpyVectorStore
from src.retrieval.retriever import Retriever


class ToyEmbedding(BaseEmbedding):
    """确定性关键词向量，仅用于讲解接口、缓存和余弦关系。"""

    TERMS = ("向量", "检索", "缓存", "注意力")

    @property
    def dim(self) -> int:
        return len(self.TERMS)

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            row = np.asarray([text.count(term) for term in self.TERMS], dtype=np.float32)
            if not np.any(row):
                row[-1] = 0.1
            norm = np.linalg.norm(row)
            rows.append(row / norm if norm else row)
        return np.asarray(rows, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


def _demo_chunks() -> list[Chunk]:
    return [
        Chunk("c0", "d0", "向量检索通过相似度寻找相关文本。", {"topic": "retrieval"}),
        Chunk("c1", "d0", "缓存可以避免重复计算相同输入。", {"topic": "cache"}),
        Chunk("c2", "d1", "注意力让 token 之间交换信息。", {"topic": "attention"}),
    ]


def lab_01() -> None:
    embedder = ToyEmbedding()
    chunks = _demo_chunks()
    store = NumpyVectorStore()
    store.add(embedder.embed([chunk.content for chunk in chunks]), chunks)
    results = Retriever(embedder, store).retrieve("什么是向量检索？", top_k=2)
    context = RAGPromptBuilder().format_chunks_for_prompt(results)
    prompt = RAGPromptBuilder().build("什么是向量检索？", context)
    answer = MockLLM("[Mock] 已完成检索到生成的数据流验证。").generate(prompt)
    print("LAB 01 | retrieved:", [chunk.id for chunk, _ in results])
    print("LAB 01 | prompt_has_context:", "向量检索" in prompt)
    print("LAB 01 | answer:", answer)


def lab_02() -> None:
    text = "第一段介绍向量检索。\n\n第二段介绍分块边界。\n\n第三段讨论重叠如何保留上下文。"
    document = Document("lesson", text, {"file_name": "lesson.md"})
    recursive = RecursiveChunker(chunk_size=24, chunk_overlap=6).chunk(document)
    fixed = FixedWindowChunker(chunk_size=24, chunk_overlap=6).chunk(document)
    print("LAB 02 | recursive_lengths:", [len(chunk.content) for chunk in recursive])
    print("LAB 02 | fixed_starts:", [chunk.metadata["start_char"] for chunk in fixed])
    print("LAB 02 | metadata_kept:", recursive[0].metadata["file_name"])


def lab_03() -> None:
    cached = CachedEmbedding(ToyEmbedding(), max_size=4)
    texts = ["向量检索", "向量检索系统", "缓存策略"]
    vectors = cached.embed(texts)
    similarity = float(vectors[0] @ vectors[1])
    cached.embed([texts[0]])
    print("LAB 03 | cosine:", round(similarity, 4))
    print("LAB 03 | cache_hit_rate:", round(cached.hit_rate, 4))
    print("LAB 03 | warning: ToyEmbedding is not a semantic model")


def lab_04() -> None:
    embedder = ToyEmbedding()
    chunks = _demo_chunks()
    store = NumpyVectorStore()
    store.add(embedder.embed([chunk.content for chunk in chunks]), chunks)
    expected = store.search(embedder.embed_query("缓存"), top_k=1)[0][0]
    with TemporaryDirectory(prefix="agentrag-learning-") as directory:
        path = Path(directory) / "index.npz"
        store.save(str(path))
        restored = NumpyVectorStore()
        restored.load(str(path))
        actual = restored.search(embedder.embed_query("缓存"), top_k=1)[0][0]
    print("LAB 04 | same_top1:", expected.id == actual.id)
    print("LAB 04 | metadata:", actual.metadata)


LABS = {"01": lab_01, "02": lab_02, "03": lab_03, "04": lab_04}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", choices=[*LABS, "all"], required=True)
    args = parser.parse_args(argv)
    selected = LABS.values() if args.lab == "all" else [LABS[args.lab]]
    for lab in selected:
        lab()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
