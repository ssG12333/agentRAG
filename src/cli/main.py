"""
================================================================================
CLI 入口 —— agentRAG 命令行工具
================================================================================

提供四个子命令：
  agentrag index  — 索引文档目录，构建向量知识库
  agentrag ask   — 单次问答：从知识库检索并生成回答
  agentrag chat  — 交互式多轮对话（Phase 3 实现）
  agentrag serve — 启动 REST API 服务（Phase 5 实现）

设计原则：
  - 使用 click 库（轻量、类型安全、自动生成 --help）
  - 延迟导入（lazy import）：避免缺少可选依赖时连 --help 都打不开
  - rich.Console 美化终端输出（进度条、颜色、表格）
  - Windows 终端兼容：避免使用 emoji（GBK 编码问题）
"""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# 工厂函数 —— 延迟导入，避免缺少可选依赖时 CLI 无法启动
# ═══════════════════════════════════════════════════════════════════════════════

def _get_embedding(model_name: str, device: str, cache: bool = True):
    """创建嵌入模型实例

    延迟导入 embedding 模块，因为 sentence-transformers 是可选依赖。
    """
    from src.embedding.model import SentenceTransformerEmbedding, CachedEmbedding

    base = SentenceTransformerEmbedding(model_name, device=device)
    # 默认开启缓存——检索和索引场景大量重复文本
    return CachedEmbedding(base) if cache else base


def _get_vector_store():
    """创建向量存储实例"""
    from src.index.vector_store import NumpyVectorStore

    return NumpyVectorStore()


def _get_llm(model_path: str, n_ctx: int, n_threads: int, temperature: float, top_p: float):
    """创建 LLM 生成引擎实例

    如果未指定模型路径，自动回退到 MockLLM（方便无模型时测试检索效果）。
    """
    from src.generation.engine import LlamaCppEngine, GenerationConfig, MockLLM

    if not model_path:
        console.print("[yellow]警告: 未指定模型路径，使用 MockLLM（固定回答）[/yellow]")
        return MockLLM()

    config = GenerationConfig(
        n_ctx=n_ctx,
        n_threads=n_threads,
        temperature=temperature,
        top_p=top_p,
    )
    return LlamaCppEngine(model_path, config)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 命令组
# ═══════════════════════════════════════════════════════════════════════════════

@click.group()
@click.version_option(version="0.1.0", prog_name="agentrag")
def main():
    """agentRAG —— 本地轻量 RAG Agent

    从零搭建，理解全链路：文档解析 -> 嵌入 -> 向量检索 -> 生成回答
    """
    pass


@main.command()
@click.option("--path", "-p", required=True, help="文档目录路径")
@click.option("--chunk-size", default=512, help="分块大小（字符数）")
@click.option("--chunk-overlap", default=64, help="分块重叠（字符数）")
@click.option("--model", default="BAAI/bge-small-zh-v1.5", help="嵌入模型名（HuggingFace ID）")
@click.option("--device", default="cpu", help="推理设备: cpu / cuda")
@click.option("--save", default="./data/vector_index.npz", help="索引持久化路径")
def index(path: str, chunk_size: int, chunk_overlap: int, model: str, device: str, save: str):
    """索引文档目录，构建向量知识库。

    扫描目录中的 .md / .txt 文件，逐文件解析 -> 分块 -> 嵌入 -> 存入向量索引，
    最后持久化到磁盘。

    示例:
        agentrag index --path ./my_docs/
        agentrag index --path ./docs/ --chunk-size 1024 --save ./data/my_index.npz
    """
    from pathlib import Path

    from src.document.parser import get_parser
    from src.document.chunker import RecursiveChunker

    # 1. 初始化各个组件
    embedder = _get_embedding(model, device)
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = _get_vector_store()

    # 2. 扫描目录，筛选支持的文档格式
    doc_dir = Path(path)
    files = list(doc_dir.rglob("*"))
    supported = [f for f in files if f.suffix.lower() in {".md", ".markdown", ".txt"}]

    if not supported:
        console.print(f"[red]目录 {path} 中没有支持的文档 (.md, .txt)[/red]")
        return

    console.print(f"目录中共找到 {len(supported)} 个支持的文档")

    # 3. 逐文档：解析 -> 分块 -> 嵌入 -> 存入
    all_chunks = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_parse = progress.add_task("解析文档并分块...", total=len(supported))
        task_embed = progress.add_task("生成嵌入向量...", total=None)

        # 阶段一：解析 + 分块（不涉及大模型，较快）
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

        # 阶段二：批量嵌入（涉及模型推理，较慢）
        # 分批处理避免 OOM（大文档目录可能有数千个 chunk）
        chunk_texts = [c.content for c in all_chunks]
        batch_size = 32  # 嵌入模型的推荐批处理大小
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i : i + batch_size]
            embeddings = embedder.embed(batch)
            store.add(embeddings, all_chunks[i : i + batch_size])
            progress.update(task_embed, completed=i + len(batch), total=len(chunk_texts))

    # 4. 持久化到磁盘
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    store.save(save)

    # 5. 输出统计信息
    console.print(f"\n[bold green]索引完成！[/bold green]")
    console.print(f"  文档数量: {len(supported)}")
    console.print(f"  分块数量: {len(store)}")
    console.print(f"  向量维度: {store.dim}")
    console.print(f"  嵌入缓存命中率: {embedder.hit_rate:.0%}")
    console.print(f"  索引文件: {save}")


@main.command()
@click.argument("query")
@click.option("--top-k", default=5, help="检索返回的块数量")
@click.option("--temperature", default=0.7, help="生成温度 (0.0~1.0)")
@click.option("--show-sources", is_flag=True, help="显示引用来源")
@click.option("--index-path", default="./data/vector_index.npz", help="索引文件路径")
@click.option("--model-path", default="", help="GGUF 模型路径（不指定则使用 MockLLM）")
@click.option("--n-ctx", default=4096, help="上下文窗口大小 (tokens)")
@click.option("--n-threads", default=8, help="CPU 推理线程数")
@click.option("--embed-model", default="BAAI/bge-small-zh-v1.5", help="嵌入模型名")
@click.option("--device", default="cpu", help="推理设备: cpu / cuda")
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
    """单次问答：从知识库中检索相关文档，由 LLM 生成回答。

    前提：需要先运行 'agentrag index' 构建索引。

    示例:
        agentrag ask "Transformer 的计算复杂度是多少？"
        agentrag ask "什么是 KV Cache？" --top-k 3 --show-sources
        agentrag ask "..." --model-path ./models/qwen2.5-3b-q4.gguf
    """
    from src.retrieval.retriever import Retriever
    from src.generation.prompt import RAGPromptBuilder

    # 1. 加载已有索引
    store = _get_vector_store()
    index_file = Path(index_path)
    if not index_file.exists():
        console.print(f"[red]索引文件不存在: {index_path}[/red]")
        console.print("请先运行: [bold]agentrag index --path ./你的文档目录/[/bold]")
        return

    console.print("加载索引...")
    store.load(str(index_file))
    console.print(f"  已加载 {len(store)} 个文本块")

    # 2. 初始化检索和生成组件
    embedder = _get_embedding(embed_model, device)
    retriever = Retriever(embedder, store)
    llm = _get_llm(model_path, n_ctx, n_threads, temperature, 0.9)
    prompt_builder = RAGPromptBuilder()

    # 3. 检索：将用户问题转为向量，搜索 top_k 相关块
    console.print(f"检索中...")
    results = retriever.retrieve(query, top_k=top_k)

    if not results:
        console.print("[red]未检索到相关内容。请确认索引中是否有相关文档。[/red]")
        return

    # 4. 构建 RAG Prompt
    chunk_texts = prompt_builder.format_chunks_for_prompt(results)
    prompt = prompt_builder.build(query, chunk_texts)

    # 5. 流式生成（打字机效果）
    console.print("\n回答:\n---")
    try:
        # 使用流式输出：逐 token 打印
        response = llm.generate_stream(prompt)
        for token in response:
            console.print(token, end="")
        console.print()  # 最后换行
        console.print("---")
    except Exception as e:
        console.print(f"[red]生成失败: {e}[/red]")

    # 6. 显示引用来源（--show-sources 开启）
    if show_sources:
        console.print("\n引用来源:")
        for i, (chunk, score) in enumerate(results):
            source = chunk.metadata.get("file_name", "unknown")
            console.print(f"  [{i+1}] [bold]{source}[/bold] (相关度: {score:.3f})")
            # 只显示前 120 个字符避免刷屏
            preview = chunk.content[:120].replace("\n", " ")
            console.print(f"      {preview}...", style="dim")


@main.command()
def chat():
    """交互式对话模式（Phase 3 实现）

    支持多轮对话、工具调用、Prefix Caching。
    """
    console.print("交互对话模式")
    console.print("[yellow]开发中 (Phase 3)[/yellow]")


@main.command()
@click.option("--host", default="127.0.0.1", help="绑定地址")
@click.option("--port", default=8000, help="监听端口")
def serve(host: str, port: int):
    """启动 REST API 服务（Phase 5 实现）"""
    console.print(f"API 服务: http://{host}:{port}")
    console.print("[yellow]开发中 (Phase 5)[/yellow]")


# ═══════════════════════════════════════════════════════════════════════════════
# 直接运行脚本时的入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
