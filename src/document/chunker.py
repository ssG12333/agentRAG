"""
================================================================================
Layer 1: 文本分块器 —— 将长文档切分为可检索的语义块
================================================================================

为什么需要分块？
  1. 嵌入模型有最大输入长度限制（BGE 系列为 512 tokens）
  2. 检索时小块更精准——用户问题通常只匹配文档的某个局部
  3. 小块送入 LLM 上下文更经济（token 成本更低）

两种策略：
  RecursiveChunker:  按分隔符优先级逐级切分（推荐，语义更完整）
  FixedWindowChunker: 固定窗口滑动切分（最简单，用于基线对比）

分块质量对最终 RAG 效果影响极大，需要根据文档类型调参 chunk_size 和 overlap。
"""

from abc import ABC, abstractmethod
from typing import List

from src.document import Document, Chunk


class BaseChunker(ABC):
    """分块器抽象基类

    chunk(doc) → List[Chunk]：输入一个 Document，输出若干 Chunk
    每个 Chunk 保留了对原始 Document 的引用（document_id）和继承的元数据。
    """

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """将文档切分为块列表"""
        ...


class RecursiveChunker(BaseChunker):
    """递归字符分块器

    算法：
      1. 按分隔符优先级列表逐级切分
         - 优先按段落（\\n\\n）切 → 保持段落完整
         - 段落太长则按换行（\\n）切 → 保持句子完整
         - 句子太长则按句号/空格切 → 保持词组完整
         - 最后按字符切 → 兜底保证不超过 chunk_size
      2. 切分后使用贪心合并：累积片段直到接近 chunk_size，然后开始新块
      3. 块之间保留 overlap：前一块末尾的 overlap 个字符复制到下一块开头
         这保证了检索时不会丢失跨块边界的语义

    参数说明：
      chunk_size:  每个块的目标最大字符数（不是精确值，贪心合并后会略小）
      chunk_overlap: 相邻块的重叠字符数。增大可减少信息断裂，但会增加索引大小
      separators:  分隔符优先级列表，排在越前面的优先用来切分

    示例：
      输入: 一篇 2000 字的文章
      参数: chunk_size=512, overlap=64
      输出: 约 4-5 个 chunk，相邻块末尾和开头有 64 个字符重叠
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: List[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 默认分隔符优先级：段落 → 换行 → 句号 → 空格 → 逐字符
        self.separators = separators or ["\n\n", "\n", "。", ". ", " ", ""]

    def chunk(self, document: Document) -> List[Chunk]:
        # 第一步：递归按分隔符切为片段
        splits = self._split_text(document.content, self.separators)
        # 第二步：贪心合并片段为目标大小的块
        chunks_text = self._merge_splits(splits)
        # 第三步：构造 Chunk 对象，附带元数据
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
        """递归按分隔符切分文本

        核心逻辑：
          - 取第一个分隔符，用它切开文本
          - 对切开的每一段，用剩余分隔符继续切（递归）
          - 分隔符附着在前一段的末尾（不丢失标点信息）
          - 最后的分隔符 ""（空字符串）触发逐字符切分——保底策略
        """
        if not separators:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        # 逐字符切分（保底）：当所有语义分隔符都无法切分时
        if sep == "":
            return list(text)

        # 当前分隔符在文本中不存在，跳过，用下一个分隔符
        if sep not in text:
            return self._split_text(text, remaining_seps)

        parts = text.split(sep)
        result = []
        for i, part in enumerate(parts):
            if i > 0:
                # 分隔符归前一段（如 "段落1\\n\\n" 而不是 "\\n\\n段落2"）
                result[-1] += sep
            if part:
                # 递归用下级分隔符继续切
                sub_splits = self._split_text(part, remaining_seps)
                result.extend(sub_splits)
        return result

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """贪心合并：将小片段聚合成不超过 chunk_size 的块

        策略：从头开始累积，当前块加下一段不超过限制就加进去，
        超过就封存当前块并开始新块。
        overlap 从上一块的末尾截取，拼到下一块开头。
        """
        if not splits:
            return []

        chunks = []
        current = ""
        for split in splits:
            if not current:
                current = split
            elif len(current) + len(split) <= self.chunk_size:
                # 还没有超过大小限制，继续累积
                current += split
            else:
                # 超过限制了，封存当前块，开始新块
                chunks.append(current)
                # 重叠策略：上一块末尾的 overlap 个字符放到新块开头
                if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                    current = current[-self.chunk_overlap:] + split
                else:
                    current = split

        # 最后一段（可能不为空）
        if current:
            chunks.append(current)

        return chunks


class FixedWindowChunker(BaseChunker):
    """固定窗口分块器

    最朴素的分块方式：窗口滑动，不关心中文语义边界。
    主要用于基线对比——衡量 RecursiveChunker 带来了多少提升。

    参数:
      chunk_size:  窗口大小（字符数）
      chunk_overlap: 窗口滑动步长 = chunk_size - overlap
                     overlap 越大步长越小，块越多，检索越细

    示例：
      chunk_size=100, overlap=20 → 步长=80
      文本前 3 个块: [0:100], [80:180], [160:260]
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            raise ValueError(
                f"chunk_size({self.chunk_size}) 必须大于 chunk_overlap({self.chunk_overlap})"
            )

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
