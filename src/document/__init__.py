"""Layer 1: 文档处理 —— 解析 / 分块 / 元数据"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Document:
    """文档对象"""
    id: str                              # 唯一标识
    content: str                         # 原始文本
    metadata: dict = field(default_factory=dict)  # 来源、标题等


@dataclass
class Chunk:
    """文本块对象"""
    id: str                              # 唯一标识
    document_id: str                     # 所属文档
    content: str                         # 块文本
    metadata: dict = field(default_factory=dict)  # 位置、标题等
    embedding: Optional[np.ndarray] = None       # 嵌入向量（索引后填充）
