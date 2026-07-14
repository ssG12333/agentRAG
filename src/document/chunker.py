"""文本分块器 —— 递归字符分割 / 固定窗口"""

import re
from abc import ABC, abstractmethod
from typing import List

from src.document import Document, Chunk


class BaseChunker(ABC):
    """分块器抽象基类"""

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """将文档切分为块列表"""
        ...


class RecursiveChunker(BaseChunker):
    """递归字符分块器

    按分隔符优先级逐级切分，直到每个块 <= chunk_size。
    相邻块保持 chunk_overlap 字符的重叠。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: List[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ". ", " ", ""]

    def chunk(self, document: Document) -> List[Chunk]:
        # 递归切分
        splits = self._split_text(document.content, self.separators)
        # 合并为指定大小的块
        chunks_text = self._merge_splits(splits)
        # 构造 Chunk 对象
        return [
            Chunk(
                id=f"{document.id}#{i}",
                document_id=document.id,
                content=text,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "chunk_count": len(chunks_text),
                },
            )
            for i, text in enumerate(chunks_text)
        ]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归按分隔符切分"""
        if not separators:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            # 最终回退：按字符切
            return list(text)

        if sep not in text:
            return self._split_text(text, remaining_seps)

        # 按分隔符切开，保留分隔符附在前一段末尾
        parts = text.split(sep)
        result = []
        for i, part in enumerate(parts):
            if i > 0:
                result[-1] += sep  # 分隔符归前一段
            if part:
                # 对每段继续用下级分隔符切分
                sub_splits = self._split_text(part, remaining_seps)
                result.extend(sub_splits)
        return result

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """合并小段到大块，保持大小约束和重叠"""
        if not splits:
            return []

        chunks = []
        current = ""
        for split in splits:
            if not current:
                current = split
            elif len(current) + len(split) <= self.chunk_size:
                current += split
            else:
                chunks.append(current)
                # 重叠：从当前块末尾取 overlap 字符作为新块开头
                if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                    current = current[-self.chunk_overlap:] + split
                else:
                    current = split

        if current:
            chunks.append(current)

        return chunks


class FixedWindowChunker(BaseChunker):
    """固定窗口分块器（最简单，用于验证）"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            raise ValueError("chunk_size 必须大于 chunk_overlap")

        chunks = []
        i = 0
        while i < len(text):
            chunk_text = text[i : i + self.chunk_size]
            chunks.append(
                Chunk(
                    id=f"{document.id}#{i}",
                    document_id=document.id,
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": i,
                        "start_char": i,
                    },
                )
            )
            i += step

        return chunks
