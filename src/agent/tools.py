"""
================================================================================
Layer 6: 工具系统 —— Agent 可调用的函数集合
================================================================================

工具是 Agent 的"手"——LLM 决定用哪个工具，工具执行具体操作并返回结果。

设计：
  - Tool: 描述工具的名称、用途、参数（给 LLM 看的 JSON Schema）
  - ToolRegistry: 注册/查找/列出工具，生成工具描述 Prompt
  - 工具调用格式使用 XML 标签 <tool_call>（比 JSON 更友好，LLM 更少出错）

使用示例:
    registry = ToolRegistry()
    registry.register(Tool(name="search", description="...", func=my_search))
    prompt = registry.get_tools_prompt()  # 给 LLM 的工具描述
    result = registry.call("search", {"query": "test"})  # 执行工具
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Tool:
    """工具定义

    Attributes:
        name: 工具名称（LLM 调用时使用的标识符）
        description: 工具功能描述（LLM 看这个决定是否使用）
        parameters: JSON Schema 格式的参数描述
        func: 实际执行的 Python 函数，签名应为 (kwargs) -> str 或 (arg1, arg2, ...) -> str
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    func: Optional[Callable] = None


class ToolRegistry:
    """工具注册表

    管理所有可用工具，提供注册、查找、调用、Prompt 生成功能。
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具。同名覆盖。"""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """移除工具"""
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        """按名称获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """列出所有已注册工具"""
        return list(self._tools.values())

    def call(self, name: str, arguments: Dict[str, Any]) -> str:
        """调用工具并返回结果字符串

        Args:
            name: 工具名
            arguments: 参数字典

        Returns:
            工具执行结果字符串。如果工具不存在或出错，返回错误描述。
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"错误: 未找到工具 '{name}'"

        if tool.func is None:
            return f"错误: 工具 '{name}' 未绑定函数"

        try:
            result = tool.func(**arguments)
            return str(result)
        except Exception as e:
            return f"错误: 调用工具 '{name}' 时出错: {e}"

    def get_tools_prompt(self) -> str:
        """生成给 LLM 的工具描述 Prompt

        格式（XML 标签式，比 JSON 更不容易出错）:

        <tools>
        <tool name="search">在知识库中搜索相关文档</tool>
        <tool name="calculator">计算数学表达式</tool>
        ...
        </tools>
        """
        if not self._tools:
            return "(没有可用的工具)"

        lines = ["<tools>"]
        for tool in self._tools.values():
            params_desc = ""
            if tool.parameters:
                param_strs = []
                for pname, pinfo in tool.parameters.items():
                    param_strs.append(
                        f'  <parameter name="{pname}" type="{pinfo.get("type","string")}">'
                        f'{pinfo.get("description","")}</parameter>'
                    )
                params_desc = "\n" + "\n".join(param_strs)

            lines.append(
                f'  <tool name="{tool.name}">'
                f'{tool.description}{params_desc}'
                f'  </tool>'
            )
        lines.append("</tools>")
        return "\n".join(lines)

    def get_call_format_prompt(self) -> str:
        """生成工具调用格式说明"""
        return """当你需要使用工具时，使用以下格式:

<tool_call>
{"name": "工具名称", "arguments": {"参数名": "参数值"}}
</tool_call>

工具执行后你会收到:

<observation>
工具返回的结果
</observation>

如果不需要使用工具，直接回答用户问题。"""

    def __len__(self) -> int:
        return len(self._tools)
