"""
================================================================================
Layer 1: 文档处理 —— 数据模型
================================================================================

定义整个 RAG 系统的核心数据结构。Document 表示原始文档，Chunk 表示
文档被分块后的文本片段。所有下游组件（解析器、分块器、嵌入层、索引层）
都围绕这两个模型工作。

设计原则：
  - 使用 dataclass 保持数据结构简洁
  - metadata 字典保持灵活性（不同解析器可存不同字段）
  - Chunk 的 embedding 字段可选（索引后才填充），避免内存浪费
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Document:
    """原始文档对象

    由解析器（parser）从文件创建。一个文件对应一个 Document。

    Attributes:
        id: 文档唯一标识，通常用文件绝对路径，保证不重复
        content: 文档完整原始文本
        metadata: 附加信息字典，包含来源文件、标题、格式、大小等
    """
    id: str                              # 唯一标识（文件路径）
    content: str                         # 原始全文
    metadata: dict = field(default_factory=dict)  # {file_path, file_name, title, format, size_bytes}


@dataclass
class Chunk:
    """文本块对象

    由分块器（chunker）从 Document 切分而来。一个 Document 对应多个 Chunk。

    Attributes:
        id: 块唯一标识，格式为 "{文档ID}#{块序号}"
        document_id: 反向引用，指向所属 Document
        content: 块内文本
        metadata: 继承自 Document 的元数据 + 块级别信息（序号、位置等）
        embedding: 嵌入向量，索引阶段由嵌入模型填充。None 表示尚未嵌入
    """
    id: str                              # 唯一标识（如 /path/doc.md#3）
    document_id: str                     # 所属文档 ID
    content: str                         # 块文本
    metadata: dict = field(default_factory=dict)  # 继承文档元数据 + 块序号
    embedding: Optional[np.ndarray] = None       # 嵌入向量（n,）或空
