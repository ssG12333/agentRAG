"""
================================================================================
Layer 4: 查询改写器 —— 指代消解 / 多轮查询优化
================================================================================

为什么需要查询改写？
  在多轮对话中，用户的追问常包含指代：
    Q1: "Transformer 有什么优点？"
    Q2: "它的计算复杂度呢？"
    这里的 "它" 指 "Transformer"。

  直接拿 Q2 去检索会搜到无关内容。
  改写后: "Transformer 的计算复杂度是多少？"

实现方式：
  - 使用 LLM 做指代消解（几行 prompt 即可）
  - 带缓存避免重复改写相同 query
  - 如果 LLM 不可用，回退为原查询（不阻塞检索）
"""

from typing import List, Optional


class QueryRewriter:
    """查询改写器 —— 指代消解 + 上下文补全

    使用示例:
        rewriter = QueryRewriter(llm)
        rewritten = rewriter.rewrite("它的复杂度呢？", [
            "用户: Transformer 有什么优点？",
            "助手: Transformer 的主要优点是并行化和长程依赖建模。",
        ])
        # → "Transformer 的计算复杂度是多少？"
    """

    REWRITE_PROMPT = """你是一个查询改写助手。根据对话历史，将用户的最新问题改写为独立可理解的查询。

规则：
1. 将指代词（它、这、那、其等）替换为具体对象
2. 补全省略的主语和上下文
3. 保持原问题的语义和意图不变
4. 只输出改写后的查询，不要解释

## 对话历史
{history}

## 用户最新问题
{query}

## 改写后的查询"""

    def __init__(self, llm=None, cache_size: int = 256):
        """
        Args:
            llm: BaseLLM 实例（可选）。如果为 None，rewrite() 原样返回。
            cache_size: 缓存容量
        """
        self._llm = llm
        self._cache: dict[str, str] = {}  # (query + history_key) → rewritten

    def rewrite(
        self,
        query: str,
        chat_history: Optional[List[str]] = None,
    ) -> str:
        """改写查询

        Args:
            query: 用户当前问题
            chat_history: 对话历史文本列表（可选）

        Returns:
            改写后的查询。如果改写失败或无 LLM，返回原查询。
        """
        # 无 LLM 时直接返回
        if self._llm is None:
            return query

        # 无历史时不需要改写
        if not chat_history:
            return query

        # 检查缓存
        history_key = "\n".join(chat_history[-3:])  # 只用最近 3 条
        cache_key = f"{query}|{history_key}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 调用 LLM 改写
        history_text = "\n".join(chat_history)
        prompt = self.REWRITE_PROMPT.format(
            history=history_text,
            query=query,
        )

        try:
            rewritten = self._llm.generate(prompt, max_tokens=128).strip()
            # 清理输出（去掉可能的引号、多余空白）
            rewritten = rewritten.strip('"\'').strip()

            # 缓存
            if len(self._cache) >= 256:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = rewritten

            return rewritten if rewritten else query
        except Exception:
            return query

    def clear_cache(self) -> None:
        self._cache.clear()
