"""文档解析器 —— 支持 Markdown / TXT 格式"""

from abc import ABC, abstractmethod
from pathlib import Path

from src.document import Document


class BaseParser(ABC):
    """解析器抽象基类"""

    @abstractmethod
    def parse(self, file_path: str) -> Document:
        """解析文件，返回 Document 对象"""
        ...

    @classmethod
    @abstractmethod
    def supports(cls, ext: str) -> bool:
        """判断是否支持该扩展名"""
        ...


class MarkdownParser(BaseParser):
    """Markdown 文档解析器"""

    @classmethod
    def supports(cls, ext: str) -> bool:
        return ext.lower() in {".md", ".markdown"}

    def parse(self, file_path: str) -> Document:
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # 提取标题作为文档名
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
    """纯文本解析器"""

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


# 注册解析器
_PARSERS: list[type[BaseParser]] = [MarkdownParser, TextParser]


def register_parser(parser_cls: type[BaseParser]) -> None:
    """注册自定义解析器"""
    _PARSERS.insert(0, parser_cls)  # 优先匹配


def get_parser(file_path: str) -> BaseParser:
    """根据文件扩展名选择解析器"""
    ext = Path(file_path).suffix.lower()
    for parser_cls in _PARSERS:
        if parser_cls.supports(ext):
            return parser_cls()
    raise ValueError(f"不支持的文件格式: {ext} (文件: {file_path})")
