"""agentRAG CLI 入口"""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src import __version__

console = Console()


# ── 延迟导入，避免缺少可选依赖时 CLI 都打不开 ──

def _get_embedding(model_name: str, device: str, cache: bool = True):
    from src.embedding.model import SentenceTransformerEmbedding, CachedEmbedding

    base = SentenceTransformerEmbedding(model_name, device=device)
    return CachedEmbedding(base) if cache else base


def _get_vector_store():
    from src.index.vector_store import NumpyVectorStore

    return NumpyVectorStore()


def _get_llm(model_path: str, n_ctx: int, n_threads: int, temperature: float, top_p: float):
    from src.generation.engine import LlamaCppEngine, GenerationConfig, MockLLM

    if not model_path:
        console.print("[yellow]⚠ 未指定模型路径，使用 MockLLM（固定回答）[/yellow]")
        return MockLLM()

    config = GenerationConfig(
        n_ctx=n_ctx,
        n_threads=n_threads,
        temperature=temperature,
        top_p=top_p,
    )
    return LlamaCppEngine(model_path, config)


# ── CLI ──

@click.group()
@click.version_option(version=__version__, prog_name="agentrag")
def main():
    """agentRAG —— 本地轻量 RAG Agent

    从零搭建，理解全链路：文档解析 → 嵌入 → 向量检索 → 生成回答
    """
    pass


@main.command()
@click.option("--path", "-p", required=True, help="文档目录路径")
@click.option("--chunk-size", default=512, help="分块大小（字符数）")
@click.option("--chunk-overlap", default=64, help="分块重叠（字符数）")
@click.option("--model", default="BAAI/bge-small-zh-v1.5", help="嵌入模型名")
@click.option("--device", default="cpu", help="设备: cpu / cuda")
@click.option("--save", default="./data/vector_index.npz", help="索引持久化路径")
def index(path: str, chunk_size: int, chunk_overlap: int, model: str, device: str, save: str):
    """索引文档目录，构建知识库"""
    from pathlib import Path

    from src.document.parser import get_parser
    from src.document.chunker import RecursiveChunker

    # 1. 初始化组件
    embedder = _get_embedding(model, device)
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = _get_vector_store()

    # 2. 扫描文件
    doc_dir = Path(path)
    files = list(doc_dir.rglob("*"))
    supported = [f for f in files if f.suffix.lower() in {".md", ".markdown", ".txt"}]

    if not supported:
        console.print(f"[red]目录 {path} 中没有支持的文档 (.md, .txt)[/red]")
        return

    console.print(f"📂 找到 {len(supported)} 个文档")

    # 3. 解析 + 分块 + 嵌入
    all_chunks = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_parse = progress.add_task("解析文档...", total=len(supported))
        task_embed = progress.add_task("嵌入向量...", total=None)

        for file_path in supported:
            file_path_str = str(file_path)
            try:
                parser = get_parser(file_path_str)
                doc = parser.parse(file_path_str)
                chunks = chunker.chunk(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                console.print(f"[red]解析失败 {file_path.name}: {e}[/red]")
            progress.advance(task_parse)

        if not all_chunks:
            console.print("[red]没有可索引的内容[/red]")
            return

        # 批量嵌入
        chunk_texts = [c.content for c in all_chunks]
        batch_size = 32
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i : i + batch_size]
            embeddings = embedder.embed(batch)
            store.add(embeddings, all_chunks[i : i + batch_size])
            progress.update(task_embed, completed=i + len(batch), total=len(chunk_texts))

    # 4. 持久化
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    store.save(save)

    # 5. 统计
    console.print(f"\n✅ 索引完成！")
    console.print(f"   文档数: {len(supported)}")
    console.print(f"   分块数: {len(store)}")
    console.print(f"   向量维度: {store.dim}")
    console.print(f"   缓存命中率: {embedder.hit_rate:.0%}")
    console.print(f"   已保存到: {save}")


@main.command()
@click.argument("query")
@click.option("--top-k", default=5, help="检索数量")
@click.option("--temperature", default=0.7, help="生成温度")
@click.option("--show-sources", is_flag=True, help="显示引用来源")
@click.option("--index-path", default="./data/vector_index.npz", help="索引文件路径")
@click.option("--model-path", default="", help="GGUF 模型路径（不指定则用 MockLLM）")
@click.option("--n-ctx", default=4096, help="上下文窗口大小")
@click.option("--n-threads", default=8, help="CPU 线程数")
@click.option("--embed-model", default="BAAI/bge-small-zh-v1.5", help="嵌入模型名")
@click.option("--device", default="cpu", help="设备: cpu / cuda")
def ask(
    query: str,
    top_k: int,
    temperature: float,
    show_sources: bool,
    index_path: str,
    model_path: str,
    n_ctx: int,
    n_threads: int,
    embed_model: str,
    device: str,
):
    """单次问答：从知识库检索并生成回答"""
    from src.retrieval.retriever import Retriever
    from src.generation.prompt import RAGPromptBuilder

    # 1. 加载索引
    store = _get_vector_store()
    index_file = Path(index_path)
    if not index_file.exists():
        console.print(f"[red]索引文件不存在: {index_path}[/red]")
        console.print("请先运行 [bold]agentrag index --path ./docs/[/bold]")
        return

    console.print("📖 加载索引...")
    store.load(str(index_file))

    # 2. 初始化组件
    embedder = _get_embedding(embed_model, device)
    retriever = Retriever(embedder, store)
    llm = _get_llm(model_path, n_ctx, n_threads, temperature, 0.9)
    prompt_builder = RAGPromptBuilder()

    # 3. 检索
    console.print(f"🔍 检索中 (top_k={top_k})...")
    results = retriever.retrieve(query, top_k=top_k)

    if not results:
        console.print("[red]未检索到相关内容[/red]")
        return

    # 4. 构建 prompt
    chunk_texts = prompt_builder.format_chunks_for_prompt(results)
    prompt = prompt_builder.build(query, chunk_texts)

    # 5. 生成
    console.print("🤔 生成回答...\n")
    try:
        response = llm.generate_stream(prompt)
        for token in response:
            console.print(token, end="")
        console.print()
    except Exception as e:
        console.print(f"[red]生成失败: {e}[/red]")

    # 6. 显示来源
    if show_sources:
        console.print("\n── 引用来源 ──")
        for i, (chunk, score) in enumerate(results):
            source = chunk.metadata.get("file_name", "unknown")
            console.print(f"  [{i+1}] {source} (相关度: {score:.3f})")
            console.print(f"       {chunk.content[:100]}...", style="dim")


@main.command()
def chat():
    """交互式对话模式"""
    console.print("💬 交互对话模式")
    console.print("[yellow]🚧 开发中 (Phase 3)[/yellow]")


@main.command()
@click.option("--host", default="127.0.0.1", help="绑定地址")
@click.option("--port", default=8000, help="监听端口")
def serve(host: str, port: int):
    """启动 REST API 服务"""
    console.print(f"🌐 API 服务: http://{host}:{port}")
    console.print("[yellow]🚧 开发中 (Phase 5)[/yellow]")


if __name__ == "__main__":
    main()
