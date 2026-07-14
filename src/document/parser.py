"""
================================================================================
Layer 1: 文档解析器 —— 从文件系统读取文档
================================================================================

职责：将目录中的文件解析为统一的 Document 对象。

设计原则：
  - 工厂模式 + 注册机制：新增格式只需实现 BaseParser 并注册即可
  - 每个解析器声明自己支持的扩展名（supports 类方法）
  - 使用 pathlib 跨平台处理路径

当前支持：
  - Markdown (.md / .markdown)：提取一级标题作为文档标题
  - 纯文本 (.txt / .text / .log / .csv / .json / .xml)

使用示例：
    parser = get_parser("/path/to/doc.md")  # 自动选择 MarkdownParser
    doc = parser.parse("/path/to/doc.md")    # 返回 Document 对象
"""

from abc import ABC, abstractmethod
from pathlib import Path

from src.document import Document


class BaseParser(ABC):
    """解析器抽象基类

    所有文档解析器必须继承此类，实现 parse() 和 supports() 两个方法。
    新增格式时，只需：
        1. 继承 BaseParser
        2. 实现两个方法
        3. 调用 register_parser() 注册
    """

    @abstractmethod
    def parse(self, file_path: str) -> Document:
        """解析文件，返回 Document 对象

        Args:
            file_path: 文件绝对路径

        Returns:
            Document 对象，id 为文件路径，content 为全文
        """
        ...

    @classmethod
    @abstractmethod
    def supports(cls, ext: str) -> bool:
        """判断此解析器是否支持该文件扩展名

        Args:
            ext: 带点的扩展名，如 ".md", ".txt"

        Returns:
            True 表示可以解析此格式
        """
        ...


class MarkdownParser(BaseParser):
    """Markdown 文档解析器

    解析 .md / .markdown 文件。
    提取第一个一级标题（# 开头的行）作为文档标题。
    """

    @classmethod
    def supports(cls, ext: str) -> bool:
        return ext.lower() in {".md", ".markdown"}

    def parse(self, file_path: str) -> Document:
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # 尝试从文档中提取第一个一级标题作为标题
        # 如果没有一级标题，回退到文件名（不含扩展名）
        title = path.stem
        for line in content.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break

        return Document(
            id=str(path.absolute()),
            content=content,
            metadata={
                "file_path": str(path.absolute()),
                "file_name": path.name,
                "title": title,
                "format": "markdown",
                "size_bytes": path.stat().st_size,
            },
        )


class TextParser(BaseParser):
    """纯文本解析器

    解析 .txt / .text / .log / .csv / .json / .xml 等纯文本文件。
    不做结构解析，直接读取全文。
    """

    @classmethod
    def supports(cls, ext: str) -> bool:
        return ext.lower() in {".txt", ".text", ".log", ".csv", ".json", ".xml"}

    def parse(self, file_path: str) -> Document:
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return Document(
            id=str(path.absolute()),
            content=content,
            metadata={
                "file_path": str(path.absolute()),
                "file_name": path.name,
                "title": path.stem,
                "format": "text",
                "size_bytes": path.stat().st_size,
            },
        )


# ── 解析器注册表 ──────────────────────────────────────────
# 注册顺序决定匹配优先级：先注册的先匹配
# Phase 5 新增 PDF 解析器时，只需在此列表添加即可

_PARSERS: list[type[BaseParser]] = [MarkdownParser, TextParser]


def register_parser(parser_cls: type[BaseParser]) -> None:
    """注册自定义解析器到全局注册表

    新解析器会插入到列表头部，优先级高于已有解析器。

    使用示例:
        class PDFParser(BaseParser):
            @classmethod
            def supports(cls, ext): return ext == ".pdf"
            def parse(self, path): ...

        register_parser(PDFParser)  # 注册后 get_parser() 即可识别 .pdf
    """
    _PARSERS.insert(0, parser_cls)


def get_parser(file_path: str) -> BaseParser:
    """根据文件扩展名自动选择解析器

    遍历注册表中的解析器，返回第一个 supports() 返回 True 的解析器实例。

    Args:
        file_path: 文件路径

    Returns:
        匹配的解析器实例

    Raises:
        ValueError: 没有解析器支持此文件格式
    """
    ext = Path(file_path).suffix.lower()
    for parser_cls in _PARSERS:
        if parser_cls.supports(ext):
            return parser_cls()
    raise ValueError(f"不支持的文件格式: {ext} (文件: {file_path})")
