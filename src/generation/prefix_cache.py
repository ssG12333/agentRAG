"""
================================================================================
Layer 5: Prefix Cache 管理 —— 为后续真实 KV 复用准备稳定前缀
================================================================================

核心思想：
  Transformer 推理分两步——prefill（处理所有输入 token）和 decode（逐个生成）。
  Prefill 计算 KV Cache。如果前缀内容在多轮对话中相同，可以复用之前的 Cache。

RAG 场景特别适合 Prefix Caching：
  1. 系统提示固定不变
  2. 同一知识库的多次检索结果可能高度重叠
  3. 多步 Agent 推理中，工具描述始终不变

Phase 3 实现边界：
  - 只实现前缀哈希、LRU 和逻辑命中统计
  - 每次命中仍把完整 prompt 交给 LLM，不会跳过真实 prefill
  - 实际 KV Cache 复用需要 llama.cpp C API，留作研究性实验
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Dict, Optional


class PrefixCache:
    """前缀缓存管理器

    维护 "前缀 hash → 缓存条目" 的 LRU 映射。

    Phase 3 实现：缓存管理逻辑 + 命中率统计。
    实际 KV Cache 操作需要 llama.cpp C API 支持（Phase 3+）。
    """

    def __init__(self, max_entries: int = 4):
        """
        Args:
            max_entries: 最大缓存条目数（LRU 淘汰）
        """
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")
        self._max_entries = max_entries
        self._cache: OrderedDict[str, Any] = OrderedDict()  # key → cache_data
        self._hits = 0
        self._misses = 0

    @staticmethod
    def cache_key(prompt_prefix: str) -> str:
        """生成前缀的哈希键

        对系统提示 + 检索上下文的内容做 SHA256，确保不同内容产生不同键。
        """
        return hashlib.sha256(prompt_prefix.encode("utf-8")).hexdigest()[:16]

    def has(self, key: str) -> bool:
        """检查前缀是否已缓存"""
        return key in self._cache

    def get(self, key: str) -> Optional[Any]:
        """获取缓存条目（LRU 更新）"""
        if key in self._cache:
            self._hits += 1
            # 移到末尾（最近使用）
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def store(self, key: str, cache_data: Any) -> None:
        """存储缓存条目

        Phase 3: cache_data 是占位符，
        后续研究：若底层 API 可用，再存储可复用的 KV Cache 句柄
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)  # LRU 淘汰最旧的
        self._cache[key] = cache_data

    def evict(self, key: str) -> None:
        """手动驱逐指定条目"""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """清空全部缓存"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self._cache),
            "max_entries": self._max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "mode": "logical",
        }


class PrefixAwareEngine:
    """带 Prefix Caching 意识的生成引擎包装器

    包装 BaseLLM，自动管理前缀缓存。
    在 Agent 多步推理循环中，系统提示 + 工具描述始终相同 → 始终命中。

    Phase 3: 管理缓存逻辑，实际 prefill 跳过需要 llama.cpp API 支持。
    """

    def __init__(self, llm, prefix_cache: PrefixCache | None = None):
        self._llm = llm
        self._cache = prefix_cache or PrefixCache()

    def _touch_prefix(self, prefix: str) -> None:
        key = PrefixCache.cache_key(prefix)
        cached = self._cache.get(key)
        if cached is None:
            self._cache.store(key, {"prefix_len": len(prefix)})

    def generate_prompt_with_cache(
        self,
        prefix: str,
        suffix: str,
        *,
        separator: str = "",
        **kwargs,
    ) -> str:
        """记录逻辑前缀命中，并始终调用底层 LLM 生成完整 prompt。"""
        self._touch_prefix(prefix)
        return self._llm.generate(prefix + separator + suffix, **kwargs)

    def generate_with_cache(
        self,
        system_prompt: str,
        context: str,
        query: str,
        **kwargs,
    ) -> str:
        """带前缀缓存的生成

        Args:
            system_prompt: 系统提示（固定部分）
            context: 检索到的文档上下文（可能重复部分）
            query: 用户当前问题（变化部分）

        Returns:
            LLM 生成的回答
        """
        # 构建前缀
        prefix = system_prompt + "\n\n" + context

        return self.generate_prompt_with_cache(
            prefix, query, separator="\n\n", **kwargs)

    def generate_stream_with_cache(
        self,
        system_prompt: str,
        context: str,
        query: str,
        **kwargs,
    ):
        """流式版本"""
        prefix = system_prompt + "\n\n" + context
        self._touch_prefix(prefix)

        full_prompt = prefix + "\n\n" + query
        for token in self._llm.generate_stream(full_prompt, **kwargs):
            yield token

    @property
    def cache_stats(self) -> Dict[str, Any]:
        return self._cache.stats
