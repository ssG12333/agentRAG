"""快速离线课程实验：只复用生产模块，不下载模型。"""

from __future__ import annotations

import argparse
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document import Chunk, Document
from src.document.chunker import FixedWindowChunker, RecursiveChunker
from src.embedding.model import BaseEmbedding, CachedEmbedding
from src.generation.engine import MockLLM
from src.generation.kv_cache import KVCacheMonitor
from src.generation.prompt import RAGPromptBuilder
from src.index.vector_store import NumpyVectorStore
from src.index.sparse_store import SparseRetriever
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.reranker import BaseReranker
from src.retrieval.retriever import Retriever
from src.agent.loop import ReActAgent
from src.agent.tools import Tool, ToolRegistry


class ToyEmbedding(BaseEmbedding):
    """确定性关键词向量，仅用于讲解接口、缓存和余弦关系。"""

    TERMS = ("向量", "检索", "缓存", "设备", "注意力")

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


def lab_05() -> None:
    try:
        import agentrag_core
    except ImportError as exc:
        print(f"SKIPPED 05: agentrag_core unavailable: {exc}")
        return
    points = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0], [5.0, 5.1]],
        dtype=np.float32,
    )
    result = agentrag_core.kmeans(points, k=2, max_iters=20, seed=42)
    centroids = np.asarray(result.centroids, dtype=np.float32).reshape(2, 2)
    print("LAB 05 | centroids:", np.round(centroids, 3).tolist())
    print("LAB 05 | assignments:", list(result.assignments))
    print("LAB 05 | note: cluster labels may swap")


def lab_06() -> None:
    try:
        from scripts.benchmark_retrieval import run_benchmark
        low_probe = run_benchmark(
            n_vectors=256, dim=16, n_queries=16, top_k=5,
            n_clusters=8, n_probe=1, n_subvectors=4, n_bits=3,
            n_iters=5, seed=7,
        )[1]
        high_probe = run_benchmark(
            n_vectors=256, dim=16, n_queries=16, top_k=5,
            n_clusters=8, n_probe=8, n_subvectors=4, n_bits=3,
            n_iters=5, seed=7,
        )[1]
    except (ImportError, RuntimeError) as exc:
        print(f"SKIPPED 06: IVF-PQ unavailable: {exc}")
        return
    print("LAB 06 | recall n_probe=1:", low_probe["recall_at_k"])
    print("LAB 06 | recall n_probe=8:", high_probe["recall_at_k"])
    print("LAB 06 | index_bytes:", high_probe["index_data_bytes"])


def lab_07() -> None:
    chunks = [
        Chunk("x", "d", "设备精确型号 X100，支持本地推理。"),
        Chunk("y", "d", "设备适合进行向量检索实验。"),
        Chunk("z", "d", "通用设备说明与安装步骤。"),
    ]
    sparse = SparseRetriever(k1=1.5, b=0.75)
    sparse.add_chunks(chunks)
    results = sparse.search("X100", top_k=3)
    print("LAB 07 | ranking:", [(chunk.id, round(float(score), 4)) for chunk, score in results])


class KeywordReranker(BaseReranker):
    """课程测试替身：包含精确 query 的候选排在前面。"""

    def rerank(self, query, candidates):
        reranked = [
            (chunk, 1.0 if query.lower() in chunk.content.lower() else 0.0)
            for chunk, _ in candidates
        ]
        return sorted(reranked, key=lambda item: -item[1])


def lab_08() -> None:
    embedder = ToyEmbedding()
    chunks = [
        Chunk("exact", "d", "设备型号 X100 的专用维护手册。"),
        Chunk("dense", "d", "向量检索设备的通用维护手册。"),
        Chunk("other", "d", "缓存与注意力的说明。"),
    ]
    store = NumpyVectorStore()
    store.add(embedder.embed([chunk.content for chunk in chunks]), chunks)
    dense = RetrievalPipeline(embedder, store)
    hybrid = RetrievalPipeline(
        embedder, store, hybrid=True, reranker=KeywordReranker(), rerank_top_n=3,
    )
    print("LAB 08 | dense_top1:", dense.retrieve("X100", top_k=1)[0][0].id)
    print("LAB 08 | hybrid_rerank_top1:", hybrid.retrieve("X100", top_k=1)[0][0].id)
    print("LAB 08 | warning: KeywordReranker is a test double")


def lab_09() -> None:
    prompt = RAGPromptBuilder().build(
        "什么是 KV Cache？",
        ["KV Cache 保存历史 token 的 Key 和 Value。"],
        ["用户: 我正在学习自回归生成。"],
    )
    stream = "".join(MockLLM("[Mock stream]").generate_stream(prompt))
    monitor = KVCacheMonitor(total_slots=1024, warn_threshold=0.9)
    monitor.update(used_slots=768)
    print("LAB 09 | prompt_sections:", all(
        part in prompt for part in ["参考文档", "对话历史", "用户问题"]
    ))
    print("LAB 09 | stream:", stream)
    print("LAB 09 | usage_pct:", monitor.stats().usage_pct)
    print("LAB 09 | memory_bytes_unknown:", monitor.stats().memory_bytes)


class SequentialLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def generate(self, prompt, **kwargs):
        return self.outputs.pop(0)

    def generate_stream(self, prompt, **kwargs):
        yield self.generate(prompt, **kwargs)


def lab_10() -> None:
    calls = []
    registry = ToolRegistry()
    registry.register(Tool(
        name="lookup",
        description="查询课程事实",
        parameters={"query": {"type": "string", "description": "查询内容"}},
        func=lambda query: calls.append(query) or "逻辑缓存不会跳过真实 prefill。",
    ))
    llm = SequentialLLM([
        '<tool_call>{"name":"lookup","arguments":{"query":"Prefix Cache"}}</tool_call>',
        "Final Answer: 当前只有逻辑命中统计。",
    ])
    agent = ReActAgent(llm, registry, max_steps=3)
    answer = agent.run("当前 Prefix Cache 实现到什么程度？")
    print("LAB 10 | tool_calls:", calls)
    print("LAB 10 | answer:", answer)
    print("LAB 10 | cache_stats:", agent.cache_stats)


def lab_11() -> None:
    from click.testing import CliRunner
    from src.cli.main import main, _resolve_backend

    result = CliRunner().invoke(main, ["--help"])
    print("LAB 11 | cli_exit_code:", result.exit_code)
    print("LAB 11 | commands_present:", all(
        name in result.output for name in ["index", "ask", "chat", "serve"]
    ))
    print("LAB 11 | auto_ivfpq:", _resolve_backend("auto", "data/demo.ivfpq"))
    print("LAB 11 | auto_numpy:", _resolve_backend("auto", "data/demo.npz"))


LABS = {
    "01": lab_01, "02": lab_02, "03": lab_03, "04": lab_04,
    "05": lab_05, "06": lab_06, "07": lab_07, "08": lab_08,
    "09": lab_09, "10": lab_10, "11": lab_11,
}


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
