"""真实模型可选实验；缺少依赖或模型时明确输出 SKIPPED。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def embedding_lab() -> bool:
    try:
        from src.embedding.model import SentenceTransformerEmbedding
        model = SentenceTransformerEmbedding("BAAI/bge-small-zh-v1.5", device="cpu")
    except Exception as exc:
        print(f"SKIPPED embedding: {exc}")
        return False
    vectors = model.embed(["向量检索", "语义搜索"])
    print("REAL embedding | shape:", vectors.shape)
    return True


def generation_lab(model_path: Path) -> bool:
    if not model_path.is_file():
        print(f"SKIPPED generation: GGUF not found: {model_path}")
        return False
    try:
        from src.generation.engine import GenerationConfig, LlamaCppEngine
        engine = LlamaCppEngine(
            str(model_path),
            GenerationConfig(n_ctx=512, n_threads=8, max_tokens=32, temperature=0.0),
        )
    except Exception as exc:
        print(f"SKIPPED generation: {exc}")
        return False
    answer = engine.generate("请用一句话解释检索增强生成。", max_tokens=32, temperature=0.0)
    print("REAL generation | answer:", answer.strip())
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=["embedding", "generation"], required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/qwen2.5-3b-instruct-q4_k_m.gguf"),
    )
    args = parser.parse_args(argv)
    if args.component == "embedding":
        embedding_lab()
    else:
        generation_lab(args.model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
