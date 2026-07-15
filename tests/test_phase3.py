"""Phase 3 Agent 层测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ════════════ 工具系统 ════════════

def test_tool_register_and_call():
    from src.agent.tools import Tool, ToolRegistry

    reg = ToolRegistry()
    reg.register(Tool(
        name="echo",
        description="Echo back",
        parameters={"msg": {"type": "string"}},
        func=lambda msg: f"Echo: {msg}",
    ))

    assert len(reg) == 1
    assert reg.get("echo") is not None
    assert reg.get("nonexist") is None
    assert reg.call("echo", {"msg": "hello"}) == "Echo: hello"


def test_tool_registry_list():
    from src.agent.tools import Tool, ToolRegistry

    reg = ToolRegistry()
    reg.register(Tool(name="a", description="Tool A"))
    reg.register(Tool(name="b", description="Tool B"))
    assert len(reg.list_tools()) == 2


def test_tools_prompt_generation():
    from src.agent.tools import Tool, ToolRegistry

    reg = ToolRegistry()
    reg.register(Tool(
        name="search",
        description="Search knowledge base",
        parameters={"query": {"type": "string", "description": "Search query"}},
    ))
    prompt = reg.get_tools_prompt()
    assert "<tools>" in prompt
    assert "search" in prompt
    assert "'query'" in prompt or '"query"' in prompt


def test_tool_call_errors():
    from src.agent.tools import ToolRegistry

    reg = ToolRegistry()
    # 调用不存在的工具
    result = reg.call("nonexist", {})
    assert "未找到" in result


# ════════════ 记忆系统 ════════════

def test_conversation_memory():
    from src.agent.memory import ConversationMemory

    mem = ConversationMemory()
    mem.add_user("你好")
    mem.add_assistant("你好！")
    assert len(mem) == 2
    assert "你好" in mem.get_history_text()
    mem.clear()
    assert len(mem) == 0


def test_working_memory():
    from src.agent.memory import WorkingMemory

    wm = WorkingMemory()
    wm.set("goal", "find answer")
    assert wm.get("goal") == "find answer"
    assert "goal" in wm
    wm.clear()
    assert wm.get("goal") is None


# ════════════ Prefix Cache ════════════

def test_prefix_cache():
    from src.generation.prefix_cache import PrefixCache

    cache = PrefixCache(max_entries=3)
    k1 = PrefixCache.cache_key("system prompt + doc1")
    k2 = PrefixCache.cache_key("system prompt + doc2")

    assert not cache.has(k1)
    cache.store(k1, {"data": 1})
    assert cache.has(k1)
    assert cache.get(k1) == {"data": 1}

    cache.store(k2, {"data": 2})
    # 缓存未命中: k2 是新键
    cache.get("unknown_key")
    assert cache.hit_rate > 0


def test_prefix_cache_eviction():
    from src.generation.prefix_cache import PrefixCache

    cache = PrefixCache(max_entries=2)
    for i in range(4):
        cache.store(PrefixCache.cache_key(f"prefix_{i}"), i)
    assert cache.stats["entries"] <= 2  # LRU 淘汰


def test_prefix_aware_engine_tracks_miss_then_hit():
    from src.generation.prefix_cache import PrefixAwareEngine

    llm = MockLLMForAgent("ok")
    engine = PrefixAwareEngine(llm)
    engine.generate_prompt_with_cache("stable", "step-1")
    engine.generate_prompt_with_cache("stable", "step-2")

    assert engine.cache_stats["mode"] == "logical"
    assert engine.cache_stats["misses"] == 1
    assert engine.cache_stats["hits"] == 1
    assert llm.prompts == ["stablestep-1", "stablestep-2"]


# ════════════ ReAct 解析 ════════════

def test_react_parse_final_answer():
    from src.agent.loop import ReActAgent
    from src.agent.tools import ToolRegistry

    agent = ReActAgent(MockLLMForAgent(), ToolRegistry())
    result = agent._parse_output("Thought: 我知道答案了。\nFinal Answer: 自注意力复杂度是O(n^2*d)")
    assert result["type"] == "final_answer"
    assert "O(n^2*d)" in result["content"]


def test_react_parse_tool_call():
    from src.agent.loop import ReActAgent
    from src.agent.tools import ToolRegistry

    agent = ReActAgent(MockLLMForAgent(), ToolRegistry())
    result = agent._parse_output(
        'Thought: 需要搜索。\n'
        '<tool_call>\n'
        '{"name": "search_knowledge_base", "arguments": {"query": "测试"}}\n'
        '</tool_call>'
    )
    assert result["type"] == "action"
    assert result["tool_name"] == "search_knowledge_base"
    assert result["arguments"] == {"query": "测试"}


def test_react_max_steps():
    from src.agent.tools import Tool, ToolRegistry
    from src.agent.loop import ReActAgent

    reg = ToolRegistry()
    reg.register(Tool(
        name="loop",
        description="Always loop",
        func=lambda: "looping",
    ))
    llm = MockLLMForAgent(
        '<tool_call>\n{"name": "loop", "arguments": {}}\n</tool_call>'
    )
    agent = ReActAgent(llm, reg, max_steps=3)
    answer = agent.run("test")
    # 应该因为超过 max_steps 而被强制终止
    assert isinstance(answer, str)


def test_react_tool_observation_final_answer_and_cache_hit():
    from src.agent.tools import Tool, ToolRegistry
    from src.agent.loop import ReActAgent

    calls = []
    registry = ToolRegistry()
    registry.register(Tool(
        name="lookup",
        description="lookup value",
        parameters={"query": {"type": "string"}},
        func=lambda query: calls.append(query) or "工具结果42",
    ))
    llm = SequentialMockLLM([
        '<tool_call>{"name": "lookup", "arguments": {"query": "答案"}}</tool_call>',
        "Final Answer: 最终答案是42",
    ])
    agent = ReActAgent(llm, registry)

    answer = agent.run("唯一问题XYZ")

    assert answer == "最终答案是42"
    assert calls == ["答案"]
    assert "Observation: 工具结果42" in llm.prompts[1]
    assert llm.prompts[0].count("唯一问题XYZ") == 1
    assert agent.cache_stats["misses"] == 1
    assert agent.cache_stats["hits"] == 1
    assert agent.memory.messages[-1] == {
        "role": "assistant",
        "content": "最终答案是42",
    }


# ════════════ 查询改写 ════════════

def test_rewriter_no_llm():
    from src.retrieval.rewriter import QueryRewriter
    rw = QueryRewriter(llm=None)
    assert rw.rewrite("它的复杂度？", ["用户: Transformer是什么？"]) == "它的复杂度？"


def test_rewriter_no_history():
    from src.retrieval.rewriter import QueryRewriter
    rw = QueryRewriter(llm=MockLLMForAgent("Transformer 的计算复杂度"))
    result = rw.rewrite("它的复杂度？", None)
    result2 = rw.rewrite("它的复杂度？", [])
    # 无历史不需要改写
    assert result == "它的复杂度？"
    assert result2 == "它的复杂度？"


# ════════════ Mock ════════════

class MockLLMForAgent:
    """Agent 测试用 Mock LLM"""
    def __init__(self, response="Final Answer: 测试回答"):
        self._response = response
        self.prompts = []
    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self._response
    def generate_stream(self, prompt, **kwargs):
        yield from self._response


class SequentialMockLLM:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return next(self._responses)
