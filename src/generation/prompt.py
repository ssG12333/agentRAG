"""
================================================================================
Layer 5: 提示词模板 —— 将检索结果拼装为 LLM 可理解的 Prompt
================================================================================

RAG 的 Prompt 通常包含四部分：
  1. 系统提示（System Prompt）：定义 LLM 的行为准则
  2. 参考文档（Context）：检索到的 chunk 内容
  3. 对话历史（Chat History）：多轮对话上下文
  4. 用户问题（Query）：用户当前的问题

为什么 Prompt 工程在 RAG 中很重要？
  - 指令边界：LLM 需要明确区分"文档内容"和"用户问题"
  - 防止幻觉：明确告诉 LLM 不要编造文档中不存在的信息
  - 来源引用：引导 LLM 在回答中标注引用了哪个文档
"""

from typing import List


# 默认系统提示 —— 强调"只看文档、不编造"
DEFAULT_SYSTEM_PROMPT = """你是一个知识助手，只根据下面提供的参考文档内容回答问题。
规则：
1. 如果文档中有明确答案，直接引用并注明来源。
2. 如果文档中没有直接答案，说"根据提供的文档，无法确定"。
3. 不要编造文档中不存在的信息。
4. 回答使用中文，简洁准确。"""


class RAGPromptBuilder:
    """RAG 提示构建器

    将检索结果、对话历史、用户问题组装为 LLM 可接受的 prompt 字符串。

    使用示例:
        builder = RAGPromptBuilder()
        prompt = builder.build(
            query="自注意力的复杂度是多少？",
            chunks=["自注意力的计算复杂度为 O(n²*d)...", "多头注意力..."],
        )
        response = llm.generate(prompt)
    """

    def __init__(self, system_prompt: str | None = None):
        """可自定义系统提示，默认使用 DEFAULT_SYSTEM_PROMPT"""
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def build(
        self,
        query: str,
        chunks: List[str],
        chat_history: List[str] | None = None,
    ) -> str:
        """构建完整 Prompt

        Prompt 结构:
            <系统提示>

            ## 参考文档
            [文档1] chunk 内容
            [文档2] chunk 内容

            ## 对话历史         (可选)
            用户: 上一个问题
            助手: 上一个回答

            ## 用户问题
            当前问题

        Args:
            query: 用户当前问题
            chunks: 检索到的文本块内容列表（已排序，最佳在前）
            chat_history: 可选的历史对话轮次

        Returns:
            可直接传入 LLM.generate() 的 prompt 字符串
        """
        parts = [self.system_prompt, ""]

        # 参考文档部分
        parts.append("## 参考文档")
        for i, chunk in enumerate(chunks):
            parts.append(f"[文档{i+1}] {chunk}")
        parts.append("")

        # 对话历史部分（可选）
        if chat_history:
            parts.append("## 对话历史")
            for msg in chat_history:
                parts.append(msg)
            parts.append("")

        # 用户问题部分
        parts.append("## 用户问题")
        parts.append(query)

        return "\n".join(parts)

    def format_chunks_for_prompt(self, chunks_with_scores) -> List[str]:
        """将检索结果 [(Chunk, score), ...] 转为纯文本列表

        提取 chunk.content，不包含分数（LLM 不需要看分数）。
        """
        result = []
        for chunk, score in chunks_with_scores:
            result.append(chunk.content)
        return result
