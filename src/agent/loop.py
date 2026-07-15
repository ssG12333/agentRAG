"""
================================================================================
Layer 6: ReAct Agent 循环 —— Reasoning + Acting
================================================================================

ReAct (Reasoning and Acting) 是经典的 LLM Agent 范式：
  Thought → Action → Observation → Thought → ... → Final Answer

流程：
  1. 用户输入问题
  2. LLM 思考是否需要调用工具
  3. 如果需要：输出 Action → 执行工具 → 得到 Observation → 回到 2
  4. 如果不需要：输出 Final Answer → 返回

约束：
  - max_steps 防止无限循环（默认 5 步）
  - 每步拼接完整的 Thought/Action/Observation 历史

Prompt 格式（ReAct 经典）：
  Thought: 我需要搜索知识库来回答这个问题
  Action: search_knowledge_base
  Action Input: {"query": "自注意力计算复杂度", "top_k": 5}

  Observation: [文档1] 自注意力复杂度为 O(n^2*d)...

  Thought: 我已经找到了答案
  Final Answer: 自注意力的计算复杂度是 O(n^2*d)
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent.tools import ToolRegistry
from src.agent.memory import ConversationMemory
from src.generation.prefix_cache import PrefixCache, PrefixAwareEngine


class ReActAgent:
    """ReAct Agent —— 思考-行动循环

    使用示例:
        agent = ReActAgent(llm, tool_registry, memory)
        answer = agent.run("自注意力复杂度是多少？")
        print(answer)
    """

    def __init__(
        self,
        llm,                           # BaseLLM 实例
        tool_registry: ToolRegistry,
        memory: ConversationMemory | None = None,
        max_steps: int = 5,
        verbose: bool = False,         # 是否打印思考过程
    ):
        self._llm = llm
        self._tools = tool_registry
        self._memory = memory or ConversationMemory()
        self._max_steps = max_steps
        self._verbose = verbose
        self._prefix_cache = PrefixCache()

    def run(self, user_query: str) -> str:
        """执行一次 ReAct 循环

        Args:
            user_query: 用户问题

        Returns:
            Agent 最终回答
        """
        # 记录用户输入
        self._memory.add_user(user_query)

        # 构建系统 Prompt（含工具描述 + 格式说明）
        system_prompt = self._build_system_prompt()

        # 对话历史作为上下文
        history_text = self._memory.get_history_text(n_last=6)

        # ReAct 循环
        step = 0
        observations: List[str] = []  # 当前轮的观察结果

        while step < self._max_steps:
            step += 1

            # 构建本轮 Prompt
            prompt = self._build_step_prompt(
                system_prompt, history_text, user_query, observations
            )

            if self._verbose:
                print(f"\n{'='*40}\nStep {step}\n{'='*40}")

            # 调用 LLM
            raw_output = self._llm.generate(prompt)

            if self._verbose:
                print(f"LLM output:\n{raw_output[:200]}...")

            # 解析输出
            parsed = self._parse_output(raw_output)

            # 检查是否 Final Answer
            if parsed["type"] == "final_answer":
                answer = parsed["content"]
                self._memory.add_assistant(answer)
                return answer

            # 检查是否 Action
            if parsed["type"] == "action":
                tool_name = parsed["tool_name"]
                arguments = parsed["arguments"]

                if self._verbose:
                    print(f"Tool: {tool_name}({arguments})")

                # 执行工具
                result = self._tools.call(tool_name, arguments)
                observations.append(
                    f"Action: {tool_name}\n"
                    f"Action Input: {json.dumps(arguments, ensure_ascii=False)}\n\n"
                    f"Observation: {result}"
                )

                if self._verbose:
                    print(f"Result: {result[:100]}...")
                continue

            # 未识别的输出：当作中间思考，继续
            observations.append(f"Thought: {raw_output[:200]}")
            if self._verbose:
                print("(未解析到 Action 或 Final Answer，继续)")

        # 超过最大步数，强制要求回答
        force_prompt = (
            system_prompt + "\n\n"
            + history_text + "\n\n"
            + "你已经尝试了多步，请基于已有信息给出最终回答。"
        )
        final = self._llm.generate(force_prompt)
        self._memory.add_assistant(final)
        return final

    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        tools_desc = self._tools.get_tools_prompt()
        call_format = self._tools.get_call_format_prompt()

        return f"""你是一个智能助手，可以使用工具来回答用户的问题。

## 可用工具

{tools_desc}

## 调用格式

{call_format}

## 思考流程

对于每个问题:
1. Thought: 分析需要什么工具、什么参数
2. 如果需要工具，使用 <tool_call> 格式调用
3. 收到 <observation> 后，判断信息是否足够
4. 如果足够，给出 Final Answer；如果不够，继续调用工具

## 规则
- 每次只调用一个工具
- 工具执行结果在 <observation> 中
- 回答使用中文，简洁准确
- 如果工具返回错误，尝试调整参数重试"""

    def _build_step_prompt(
        self,
        system_prompt: str,
        history: str,
        query: str,
        observations: List[str],
    ) -> str:
        """构建单步 Prompt"""
        parts = [system_prompt, ""]

        if history:
            parts.append(f"## 对话历史\n{history}\n")

        parts.append(f"## 用户问题\n{query}\n")

        if observations:
            parts.append("## 已执行的操作\n" + "\n\n".join(observations) + "\n")
            parts.append("请继续思考：是调用更多工具，还是给出最终回答？")
        else:
            parts.append("请开始思考。")

        return "\n".join(parts)

    def _parse_output(self, text: str) -> Dict[str, Any]:
        """解析 LLM 输出

        支持两种格式:
          1. <tool_call>{"name": "...", "arguments": {...}}</tool_call>
          2. Final Answer: <文本>

        Returns:
            {"type": "action", "tool_name": ..., "arguments": ...}
            {"type": "final_answer", "content": ...}
            {"type": "unknown", "content": ...}
        """
        # 1. 检查 tool_call
        tool_match = re.search(
            r'<tool_call>\s*\n?\s*(\{.*?\})\s*\n?\s*</tool_call>',
            text,
            re.DOTALL,
        )
        if tool_match:
            try:
                call_data = json.loads(tool_match.group(1))
                return {
                    "type": "action",
                    "tool_name": call_data["name"],
                    "arguments": call_data.get("arguments", {}),
                }
            except (json.JSONDecodeError, KeyError):
                pass

        # 2. 检查 Final Answer
        answer_match = re.search(
            r'Final Answer\s*[:：]\s*(.+)',
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if answer_match:
            return {
                "type": "final_answer",
                "content": answer_match.group(1).strip(),
            }

        # 3. 如果文本末尾没有工具调用也没有 Final Answer，取最后一段作为回答
        # （LLM 有时会直接回答而不标记 Final Answer）
        if not re.search(r'<tool_call>', text):
            # 尝试取最后有意义的一段
            lines = text.strip().split("\n")
            # 从后往前找第一个非空行
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith(("Thought", "Action", "Observation")):
                    return {"type": "final_answer", "content": line}

        return {"type": "unknown", "content": text}

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    @property
    def tools(self) -> ToolRegistry:
        return self._tools
