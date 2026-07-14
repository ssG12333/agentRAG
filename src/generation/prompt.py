"""提示词模板 —— RAG 系统提示 + 用户提示构建"""

from typing import List


# 默认系统提示
DEFAULT_SYSTEM_PROMPT = """你是一个知识助手，只根据下面提供的参考文档内容回答问题。
规则：
1. 如果文档中有明确答案，直接引用并注明来源。
2. 如果文档中没有直接答案，说"根据提供的文档，无法确定"。
3. 不要编造文档中不存在的信息。
4. 回答使用中文，简洁准确。"""


class RAGPromptBuilder:
    """RAG 提示构建器"""

    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def build(
        self,
        query: str,
        chunks: List[str],
        chat_history: List[str] | None = None,
    ) -> str:
        """构建完整的 RAG 提示

        格式:
            <系统提示>

            ## 参考文档
            [文档1] ...
            [文档2] ...

            ## 对话历史
            ...

            ## 用户问题
            ...
        """
        parts = [self.system_prompt, ""]

        # 参考文档
        parts.append("## 参考文档")
        for i, chunk in enumerate(chunks):
            parts.append(f"[文档{i+1}] {chunk}")
        parts.append("")

        # 对话历史
        if chat_history:
            parts.append("## 对话历史")
            for msg in chat_history:
                parts.append(msg)
            parts.append("")

        # 用户问题
        parts.append("## 用户问题")
        parts.append(query)

        return "\n".join(parts)

    def format_chunks_for_prompt(self, chunks_with_scores) -> List[str]:
        """将检索结果格式化为提示用的文本列表"""
        result = []
        for chunk, score in chunks_with_scores:
            result.append(chunk.content)
        return result
