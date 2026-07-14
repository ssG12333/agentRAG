"""Phase 1 测试合集"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# 确保项目根在 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ════════════ 文档解析器 ════════════

def test_markdown_parser():
    from src.document.parser import MarkdownParser
    from src.document import Document

    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", encoding="utf-8", delete=False) as f:
        f.write("# 测试标题\n\n这是测试内容。\n\n## 第二段\n\n更多内容。")
        path = f.name

    try:
        parser = MarkdownParser()
        doc = parser.parse(path)

        assert isinstance(doc, Document)
        assert doc.metadata["title"] == "测试标题"
        assert "测试内容" in doc.content
        assert doc.metadata["format"] == "markdown"
    finally:
        os.unlink(path)


def test_text_parser():
    from src.document.parser import TextParser
    from src.document import Document

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as f:
        f.write("纯文本内容\n第二行")
        path = f.name

    try:
        parser = TextParser()
        doc = parser.parse(path)

        assert isinstance(doc, Document)
        assert "纯文本内容" in doc.content
        assert doc.metadata["format"] == "text"
    finally:
        os.unlink(path)


def test_get_parser():
    from src.document.parser import get_parser, MarkdownParser, TextParser

    assert isinstance(get_parser("test.md"), MarkdownParser)
    assert isinstance(get_parser("test.txt"), TextParser)

    with pytest.raises(ValueError):
        get_parser("test.pdf")


# ════════════ 分块器 ════════════

def test_recursive_chunker():
    from src.document import Document
    from src.document.chunker import RecursiveChunker

    doc = Document(
        id="test",
        content="第一段内容。\n\n第二段内容。\n\n第三段更长一些的内容放在这里。",
    )
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.document_id == "test"
        assert chunk.metadata.get("chunk_index") is not None


def test_fixed_window_chunker():
    from src.document import Document
    from src.document.chunker import FixedWindowChunker

    doc = Document(id="test", content="0123456789" * 20)  # 200 chars
    chunker = FixedWindowChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    # 每个 chunk 不应超过 chunk_size
    for chunk in chunks:
        assert len(chunk.content) <= 50


def test_chunker_overlap():
    from src.document import Document
    from src.document.chunker import RecursiveChunker

    doc = Document(id="test", content="A" * 100)
    chunker = RecursiveChunker(chunk_size=30, chunk_overlap=10)
    chunks = chunker.chunk(doc)

    if len(chunks) >= 2:
        # 前一块末尾应与后一块开头有重叠
        tail = chunks[0].content[-10:]
        head = chunks[1].content[:10]
        assert tail == head


# ════════════ 向量存储 ════════════

def test_vector_store_add_search():
    import numpy as np
    from src.document import Chunk
    from src.index.vector_store import NumpyVectorStore

    store = NumpyVectorStore()
    chunks = [
        Chunk(id="0", document_id="d0", content="aaa"),
        Chunk(id="1", document_id="d0", content="bbb"),
        Chunk(id="2", document_id="d0", content="ccc"),
    ]
    # 正态化的向量
    vecs = np.array([
        [1.0, 0.0, 0.0],   # 与 query [1,0,0] 最相似
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    store.add(vecs, chunks)
    assert len(store) == 3

    results = store.search(np.array([1.0, 0.0, 0.0]), top_k=2)
    assert len(results) == 2
    assert results[0][0].content == "aaa"  # 最相似
    assert results[0][1] > results[1][1]   # 分数递减


def test_vector_store_persist():
    import numpy as np
    from src.document import Chunk
    from src.index.vector_store import NumpyVectorStore

    store = NumpyVectorStore()
    chunks = [Chunk(id="0", document_id="d0", content="test")]
    vecs = np.array([[1.0, 0.0]], dtype=np.float32)
    store.add(vecs, chunks)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name

    try:
        store.save(path)

        store2 = NumpyVectorStore()
        store2.load(path)
        assert len(store2) == 1
        assert store2._chunks[0].content == "test"
    finally:
        os.unlink(path)


# ════════════ 检索器 ════════════

def test_retriever():
    import numpy as np
    from src.document import Chunk
    from src.index.vector_store import NumpyVectorStore
    from src.retrieval.retriever import Retriever

    # 用简单的 mock embedding
    class MockEmbedding:
        dim = 3

        def embed(self, texts):
            return np.array([[1.0, 0.0, 0.0]] * len(texts))

        def embed_query(self, query):
            return np.array([1.0, 0.0, 0.0])

    store = NumpyVectorStore()
    chunks = [Chunk(id=str(i), document_id="d0", content=f"content_{i}") for i in range(3)]
    vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    store.add(vecs, chunks)

    embedder = MockEmbedding()
    retriever = Retriever(embedder, store)
    results = retriever.retrieve("test", top_k=2)

    assert len(results) == 2
    assert results[0][0].content == "content_0"


# ════════════ 提示模板 ════════════

def test_prompt_builder():
    from src.generation.prompt import RAGPromptBuilder

    builder = RAGPromptBuilder()
    prompt = builder.build(
        query="什么是 Transformer？",
        chunks=["文档1内容", "文档2内容"],
    )

    assert "什么是 Transformer" in prompt
    assert "文档1内容" in prompt
    assert "文档2内容" in prompt
    assert "参考文档" in prompt
    assert "用户问题" in prompt


# ════════════ 生成引擎 ════════════

def test_mock_llm():
    from src.generation.engine import MockLLM

    llm = MockLLM("测试回答")
    assert llm.generate("任意提示") == "测试回答"

    tokens = list(llm.generate_stream("任意提示"))
    assert len(tokens) == 4  # "测试回答"
    assert "".join(tokens) == "测试回答"
