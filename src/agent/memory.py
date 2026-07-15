"""
================================================================================
Layer 6: Agent 记忆系统 —— 对话历史 / 工作记忆
================================================================================

记忆分两类：
  ConversationMemory: 长期对话上下文（所有轮次），用于 Prompt 拼接
  WorkingMemory:     短期工作区（当前任务的目标、中间结果），非持久化

设计原则：
  - 记忆不嵌入 Agent 循环中，Agent 显式读写
  - ConversationMemory 用 role+content 格式，兼容 LLM 的 OpenAI 格式
  - Phase 3 简单实现，Phase 5 可升级为向量持久化
"""

from typing import Any, Dict, List, Optional


class ConversationMemory:
    """对话历史记忆

    存储所有对话轮次，格式为 [{"role": "user", "content": "..."}, ...]
    """

    def __init__(self, max_turns: int = 20):
        self._messages: List[Dict[str, str]] = []
        self._max_turns = max_turns  # 最多保留的轮次数（1 turn = user + assistant）

    def add_user(self, message: str) -> None:
        """添加用户消息"""
        self._messages.append({"role": "user", "content": message})

    def add_assistant(self, message: str) -> None:
        """添加助手消息"""
        self._messages.append({"role": "assistant", "content": message})

    def add_system(self, message: str) -> None:
        """添加系统消息（通常只有第一条）"""
        self._messages.insert(0, {"role": "system", "content": message})

    def get_history(self, n_last: int = 10) -> List[Dict[str, str]]:
        """取最近 n 条消息（按原始顺序）

        Args:
            n_last: 取最近 n 条。0 表示取出全部（受 max_turns 限制）

        Returns:
            [{"role": ..., "content": ...}, ...]
        """
        if n_last <= 0:
            return self._messages[-self._max_turns * 2:]  # 2 = user + assistant per turn
        return self._messages[-n_last:]

    def get_history_text(self, n_last: int = 10) -> str:
        """取最近 n 条消息的文本形式

        格式:
            用户: 用户的上一轮问题
            助手: 助手的上一轮回答
        """
        history = self.get_history(n_last)
        lines = []
        for msg in history:
            if msg["role"] == "user":
                lines.append(f"用户: {msg['content']}")
            elif msg["role"] == "assistant":
                lines.append(f"助手: {msg['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """清空全部记忆"""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self._messages


class WorkingMemory:
    """工作记忆（短期，不持久化）

    用于存储当前任务的临时状态，如：
      - 当前目标
      - 中间检索结果
      - 已尝试的工具列表
    """

    def __init__(self):
        self._store: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        return self._store.pop(key, default)

    def clear(self) -> None:
        self._store.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def snapshot(self) -> Dict[str, Any]:
        """返回当前所有键值的快照"""
        return dict(self._store)
