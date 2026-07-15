"""校验系统学习课程的文件、统一章节结构和相对链接。"""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEARNING_DIR = PROJECT_ROOT / "docs" / "learning"
CHAPTERS = [
    "00_course_guide.md",
    "01_rag_architecture.md",
    "02_document_processing.md",
    "03_embeddings.md",
    "04_exact_vector_search.md",
    "05_cpp_pybind_kmeans.md",
    "06_residual_ivfpq.md",
    "07_bm25.md",
    "08_hybrid_rerank.md",
    "09_prompt_generation_kv.md",
    "10_agent.md",
    "11_cli_integration.md",
    "12_evaluation.md",
    "13_quantization_preresearch.md",
    "14_prefix_kv_preresearch.md",
    "15_production_preresearch.md",
]
SUPPORTING_DOCS = ["README.md", "glossary.md", "references.md", "_chapter_template.md"]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def _check_required_files(errors: list[str]) -> None:
    for filename in CHAPTERS + SUPPORTING_DOCS:
        if not (LEARNING_DIR / filename).is_file():
            errors.append(f"缺少课程文件: docs/learning/{filename}")


def _check_chapter_structure(errors: list[str]) -> None:
    markers = (
        "## 学习目标与先修知识",
        "> 状态：",
        "<details><summary>参考答案</summary>",
    )
    for filename in CHAPTERS:
        path = LEARNING_DIR / filename
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in content:
                errors.append(f"{path.relative_to(PROJECT_ROOT)} 缺少结构标记: {marker}")


def _iter_link_docs() -> list[Path]:
    return [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "README-cn.md",
        *sorted(LEARNING_DIR.glob("*.md")),
    ]


def _check_relative_links(errors: list[str]) -> None:
    for path in _iter_link_docs():
        content = path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(content):
            target = match.group(1).strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative_target = target.split("#", maxsplit=1)[0]
            resolved = (path.parent / relative_target).resolve()
            if not resolved.exists():
                errors.append(
                    f"{path.relative_to(PROJECT_ROOT)} 存在失效链接: {target}"
                )


def main() -> int:
    errors: list[str] = []
    _check_required_files(errors)
    _check_chapter_structure(errors)
    _check_relative_links(errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {len(CHAPTERS)} 章课程文件、统一结构和相对链接均有效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
