"""agentRAG CLI 入口"""

from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src import __version__

console = Console()


# ── 工厂函数 ──────────────────────────────────────────

def _get_embedding(model_name: str, device: str, cache: bool = True):
    from src.embedding.model import SentenceTransformerEmbedding, CachedEmbedding
    base = SentenceTransformerEmbedding(model_name, device=device)
    return CachedEmbedding(base) if cache else base


def _get_vector_store(backend: str, **kwargs):
    if backend == "numpy":
        from src.index.vector_store import NumpyVectorStore
        return NumpyVectorStore()
    if backend == "ivfpq":
        from src.index.ivfpq_store import IVFPQVectorStore
        return IVFPQVectorStore(**kwargs)
    raise ValueError(f"未知索引后端: {backend}")


def _resolve_backend(backend: str, index_path: str | None) -> str:
    if backend != "auto":
        return backend
    if index_path and Path(index_path).suffix.lower() == ".ivfpq":
        return "ivfpq"
    return "numpy"


def _resolve_index_path(index_path: str | None, backend: str) -> str:
    if index_path:
        return index_path
    if backend == "ivfpq":
        return "./data/vector_index.ivfpq"
    return "./data/vector_index.npz"


def _build_retrieval_pipeline(
    embedder,
    store,
    *,
    hybrid: bool,
    reranker_model: str,
    reranker_device: str,
    rerank_top_n: int,
):
    from src.retrieval.pipeline import RetrievalPipeline
    from src.retrieval.reranker import CrossEncoderReranker

    reranker = None
    if reranker_model:
        reranker = CrossEncoderReranker(reranker_model, device=reranker_device)
    return RetrievalPipeline(
        embedder,
        store,
        hybrid=hybrid,
        reranker=reranker,
        rerank_top_n=rerank_top_n,
    )


def _get_llm(model_path: str, n_ctx: int, n_threads: int, temperature: float, top_p: float):
    from src.generation.engine import LlamaCppEngine, GenerationConfig, MockLLM
    if not model_path:
        console.print("[yellow]警告: 未指定模型路径，使用 MockLLM（固定回答）[/yellow]")
        return MockLLM()
    config = GenerationConfig(n_ctx=n_ctx, n_threads=n_threads, temperature=temperature, top_p=top_p)
    return LlamaCppEngine(model_path, config)


# ── CLI ──────────────────────────────────────────────

@click.group()
@click.version_option(version=__version__, prog_name="agentrag")
def main():
    """agentRAG -- 本地轻量 RAG Agent"""
    pass


@main.command()
@click.option("--path", "-p", required=True, help="文档目录路径")
@click.option("--chunk-size", default=512, help="分块大小（字符数）")
@click.option("--chunk-overlap", default=64, help="分块重叠（字符数）")
@click.option("--model", default="BAAI/bge-small-zh-v1.5", help="嵌入模型名")
@click.option("--device", default="cpu", help="设备: cpu / cuda")
@click.option("--backend", type=click.Choice(["numpy", "ivfpq"]), default="numpy", show_default=True)
@click.option("--n-clusters", default=256, show_default=True, help="IVF 聚类数")
@click.option("--n-probe", default=8, show_default=True, help="IVF 检索探测簇数")
@click.option("--n-subvectors", default=64, show_default=True, help="PQ 子向量数")
@click.option("--n-bits", default=8, show_default=True, help="PQ 每段编码位数")
@click.option("--save", default=None, help="索引持久化路径")
def index(
    path: str,
    chunk_size: int,
    chunk_overlap: int,
    model: str,
    device: str,
    backend: str,
    n_clusters: int,
    n_probe: int,
    n_subvectors: int,
    n_bits: int,
    save: str | None,
):
    """索引文档目录，构建向量知识库"""
    from pathlib import Path
    from src.document.parser import get_parser
    from src.document.chunker import RecursiveChunker

    embedder = _get_embedding(model, device)
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = _get_vector_store(
        backend,
        n_clusters=n_clusters,
        n_probe=n_probe,
        n_subvectors=n_subvectors,
        n_bits=n_bits,
    )

    doc_dir = Path(path)
    files = list(doc_dir.rglob("*"))
    supported = [f for f in files if f.suffix.lower() in {".md", ".markdown", ".txt"}]

    if not supported:
        console.print(f"[red]目录 {path} 中没有支持的文档 (.md, .txt)[/red]")
        return

    all_chunks = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task_p = progress.add_task("解析文档并分块...", total=len(supported))
        task_e = progress.add_task("生成嵌入向量...", total=None)
        for fp in supported:
            try:
                parser = get_parser(str(fp))
                doc = parser.parse(str(fp))
                all_chunks.extend(chunker.chunk(doc))
            except Exception as e:
                console.print(f"[red]解析失败 {fp.name}: {e}[/red]")
            progress.advance(task_p)
        if not all_chunks:
            console.print("[red]没有可索引的内容[/red]")
            return
        texts = [c.content for c in all_chunks]
        embedding_batches = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            embedding_batches.append(np.asarray(embedder.embed(batch), dtype=np.float32))
            progress.update(task_e, completed=i+len(batch), total=len(texts))
        store.add(np.vstack(embedding_batches), all_chunks)

    save = _resolve_index_path(save, backend)
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    store.save(save)

    console.print(f"\n[bold green]索引完成[/bold green]")
    console.print(
        f"  后端: {backend}  文档: {len(supported)}  分块: {len(store)}  "
        f"维度: {store.dim}  缓存: {embedder.hit_rate:.0%}"
    )
    console.print(f"  索引: {save}")


@main.command()
@click.argument("query")
@click.option("--top-k", default=5, help="检索数量")
@click.option("--temperature", default=0.7, help="生成温度")
@click.option("--show-sources", is_flag=True, help="显示引用来源")
@click.option("--index-path", default=None, help="索引文件路径")
@click.option("--backend", type=click.Choice(["auto", "numpy", "ivfpq"]), default="auto", show_default=True)
@click.option("--hybrid/--dense-only", default=False, show_default=True, help="启用 BM25 + RRF")
@click.option("--reranker-model", default="", help="Cross-Encoder 模型；留空关闭")
@click.option("--reranker-device", default="cpu", help="重排序设备")
@click.option("--rerank-top-n", default=30, show_default=True, help="送入重排序的候选数")
@click.option("--model-path", default="", help="GGUF 模型路径")
@click.option("--n-ctx", default=4096, help="上下文窗口大小")
@click.option("--n-threads", default=8, help="CPU 线程数")
@click.option("--embed-model", default="BAAI/bge-small-zh-v1.5", help="嵌入模型名")
@click.option("--device", default="cpu", help="设备: cpu / cuda")
def ask(
    query, top_k, temperature, show_sources, index_path, backend, hybrid,
    reranker_model, reranker_device, rerank_top_n, model_path, n_ctx,
    n_threads, embed_model, device,
):
    """单次问答：从知识库检索并生成回答"""
    from src.generation.prompt import RAGPromptBuilder

    backend = _resolve_backend(backend, index_path)
    index_path = _resolve_index_path(index_path, backend)
    store = _get_vector_store(backend)
    index_file = Path(index_path)
    if not index_file.exists():
        console.print(f"[red]索引文件不存在: {index_path}[/red]")
        return

    store.load(str(index_file))
    embedder = _get_embedding(embed_model, device)
    retriever = _build_retrieval_pipeline(
        embedder,
        store,
        hybrid=hybrid,
        reranker_model=reranker_model,
        reranker_device=reranker_device,
        rerank_top_n=rerank_top_n,
    )
    llm = _get_llm(model_path, n_ctx, n_threads, temperature, 0.9)
    prompt_builder = RAGPromptBuilder()

    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        console.print("[red]未检索到相关内容[/red]")
        return

    chunk_texts = prompt_builder.format_chunks_for_prompt(results)
    prompt = prompt_builder.build(query, chunk_texts)
    console.print("\n---")
    try:
        for token in llm.generate_stream(prompt):
            console.print(token, end="")
    except Exception as e:
        console.print(f"[red]生成失败: {e}[/red]")
    console.print("\n---")

    if show_sources:
        console.print("\n引用来源:")
        for i, (chunk, score) in enumerate(results):
            console.print(f"  [{i+1}] {chunk.metadata.get('file_name','?')} (相关度: {score:.3f})")


@main.command()
@click.option("--index-path", default=None, help="索引文件路径")
@click.option("--backend", type=click.Choice(["auto", "numpy", "ivfpq"]), default="auto", show_default=True)
@click.option("--hybrid/--dense-only", default=False, show_default=True, help="启用 BM25 + RRF")
@click.option("--reranker-model", default="", help="Cross-Encoder 模型；留空关闭")
@click.option("--reranker-device", default="cpu", help="重排序设备")
@click.option("--rerank-top-n", default=30, show_default=True, help="送入重排序的候选数")
@click.option("--model-path", default="", help="GGUF 模型路径")
@click.option("--n-ctx", default=4096, help="上下文窗口大小")
@click.option("--n-threads", default=8, help="CPU 线程数")
@click.option("--embed-model", default="BAAI/bge-small-zh-v1.5", help="嵌入模型名")
@click.option("--device", default="cpu", help="设备: cpu / cuda")
@click.option("--verbose", is_flag=True, help="显示 Agent 思考过程")
def chat(
    index_path, backend, hybrid, reranker_model, reranker_device,
    rerank_top_n, model_path, n_ctx, n_threads, embed_model, device, verbose,
):
    """交互式对话 —— Agent 模式（工具调用 + 多轮记忆）"""
    from src.agent.tools import Tool, ToolRegistry
    from src.agent.memory import ConversationMemory
    from src.agent.loop import ReActAgent
    from src.retrieval.rewriter import QueryRewriter

    # 加载索引和检索器
    backend = _resolve_backend(backend, index_path)
    index_path = _resolve_index_path(index_path, backend)
    store = _get_vector_store(backend)
    index_file = Path(index_path)
    if index_file.exists():
        store.load(str(index_file))
        console.print(f"[dim]已加载 {backend} 索引: {len(store)} chunks[/dim]")
    embedder = _get_embedding(embed_model, device)
    retriever = _build_retrieval_pipeline(
        embedder,
        store,
        hybrid=hybrid,
        reranker_model=reranker_model,
        reranker_device=reranker_device,
        rerank_top_n=rerank_top_n,
    )
    llm = _get_llm(model_path, n_ctx, n_threads, 0.7, 0.9)
    rewriter = QueryRewriter(llm)

    # 构建工具注册表
    registry = ToolRegistry()
    registry.register(Tool(
        name="search_knowledge_base",
        description="在本地知识库中搜索相关文档。当需要查找事实、概念、定义时使用。",
        parameters={
            "query": {"type": "string", "description": "搜索查询（关键词或自然语言）"},
            "top_k": {"type": "integer", "description": "返回文档数量，默认5"},
        },
        func=lambda query, top_k=5: _format_search_results(retriever.retrieve(query, top_k=top_k)),
    ))
    registry.register(Tool(
        name="list_documents",
        description="列出知识库中已索引的文档列表",
        parameters={},
        func=lambda: _list_docs(store),
    ))

    # 创建 Agent
    memory = ConversationMemory()
    agent = ReActAgent(llm, registry, memory, max_steps=5, verbose=verbose)

    console.print("[bold]agentRAG 交互对话[/bold] (输入 /exit 退出, /clear 清空记忆, /cache 查看缓存, /docs 列出文档)")
    console.print(f"已加载 {len(registry)} 个工具，检索模式: {retriever.mode}")

    while True:
        try:
            user_input = console.input("\n[bold green]>>[/bold green] ")
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input.strip():
            continue

        # 特殊命令
        if user_input.lower() in ("/exit", "/quit"):
            break
        elif user_input.lower() == "/clear":
            memory.clear()
            console.print("[dim]记忆已清空[/dim]")
            continue
        elif user_input.lower() == "/cache":
            stats = agent.cache_stats
            console.print(
                f"  Prefix Cache ({stats['mode']}): "
                f"{stats['entries']}/{stats['max_entries']} 条, "
                f"命中率: {stats['hit_rate']:.0%}"
            )
            continue
        elif user_input.lower() == "/docs":
            console.print(_list_docs(store))
            continue

        # 查询改写
        history = memory.get_history_text(n_last=8)
        query = rewriter.rewrite(user_input, [history] if history else None)

        if verbose and query != user_input:
            console.print(f"[dim]改写: {query}[/dim]")

        # Agent 运行
        try:
            answer = agent.run(query)
            console.print(f"\n{answer}")
        except Exception as e:
            console.print(f"[red]Agent 错误: {e}[/red]")


def _format_search_results(results):
    """格式化检索结果为字符串"""
    if not results:
        return "未找到相关文档。"
    lines = []
    for i, (chunk, score) in enumerate(results):
        source = chunk.metadata.get("file_name", "?")
        lines.append(f"[文档{i+1}] (来源: {source}, 相关度: {score:.3f})\n{chunk.content[:300]}")
    return "\n\n".join(lines)


def _list_docs(store):
    """列出已索引文档（去重）"""
    from collections import Counter
    sources = Counter()
    for c in store.chunks:
        sources[c.metadata.get("file_name", "?")] += 1
    if not sources:
        return "(无已索引文档)"
    lines = ["已索引文档:"]
    for name, count in sources.items():
        lines.append(f"  {name} ({count} chunks)")
    return "\n".join(lines)


@main.command()
@click.option("--host", default="127.0.0.1", help="绑定地址")
@click.option("--port", default=8000, help="监听端口")
def serve(host: str, port: int):
    """启动 REST API 服务"""
    console.print(f"[yellow]开发中 (Phase 5)[/yellow]")


if __name__ == "__main__":
    main()
