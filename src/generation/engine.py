"""
================================================================================
Layer 5: 生成引擎 —— LLM 推理封装
================================================================================

生成引擎负责将 Prompt 送入 LLM 并产出回答。
Phase 1 提供两种实现：
  - LlamaCppEngine: 基于 llama-cpp-python，加载 GGUF 格式的量化模型
  - MockLLM: 测试用，返回固定回答，不依赖任何模型文件

设计原则：
  - BaseLLM 抽象，后续可替换为 ONNX / vLLM / API
  - 支持流式输出（generate_stream），打字机效果更好
  - GenerationConfig 集中管理所有推理参数
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class GenerationConfig:
    """LLM 生成参数

    这些参数直接影响生成质量、速度和显存占用。

    关键参数解释：
      temperature: 温度，控制随机性。0.0=贪婪（最确定），1.0=随机。
                   RAG 场景建议 0.4-0.7，平衡准确性和流畅度。
      top_p:      核采样阈值。只在累计概率 ≤ top_p 的候选词中采样。
                   0.9 是常用默认值。
      top_k:      只从概率最高的 k 个候选词中采样。40 是常用默认值。
      repeat_penalty: 重复惩罚。>1.0 降低已出现 token 的概率，减少重复生成。
      n_ctx:      上下文窗口（tokens）。决定了能塞入多少检索结果。
                   4096 是本地模型的常见设置。
                   注意：n_ctx 也决定了 KV Cache 的最大大小。
      n_batch:    每批处理的 token 数。影响 Prefill 阶段的速度和内存波动。
                   512 是 llama.cpp 默认值。
    """
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 1024       # 单次生成的最大 token 数
    repeat_penalty: float = 1.1  # 1.0=不惩罚, >1.0=惩罚重复
    # llama.cpp 特有参数
    n_ctx: int = 4096            # 上下文窗口（影响 KV Cache 容量）
    n_threads: int = 8           # CPU 推理线程数（GPU 推理可设为 1）
    n_gpu_layers: int = 0        # GPU offload 层数，0=纯CPU, -1=全GPU
    n_batch: int = 512           # 批处理大小
    seed: int = 42               # 随机种子（固定可复现）


class BaseLLM(ABC):
    """LLM 抽象基类

    所有生成引擎必须实现 generate() 和 generate_stream()。
    流式输出的优势：用户可以看到逐字输出，体验更接近 ChatGPT。
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """完整生成（阻塞直到完成）

        Args:
            prompt: 完整的 prompt 字符串
            **kwargs: 可覆盖 GenerationConfig 中的参数

        Returns:
            LLM 生成的完整回答文本
        """
        ...

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成（逐 token yield）

        Args:
            prompt: 完整的 prompt 字符串
            **kwargs: 可覆盖 GenerationConfig 中的参数

        Yields:
            每个 yield 返回一个新生成的 token 片段
        """
        ...


class LlamaCppEngine(BaseLLM):
    """基于 llama-cpp-python 的推理引擎

    加载 GGUF 格式的量化模型（Q4_K_M / Q8_0 / FP16 等），
    通过 llama.cpp 的 C++ 内核进行推理。

    使用示例:
        config = GenerationConfig(n_ctx=4096, n_threads=8)
        engine = LlamaCppEngine("./models/qwen2.5-3b-Q4_K_M.gguf", config)
        answer = engine.generate("你好，请介绍一下自己")
        for token in engine.generate_stream(prompt):
            print(token, end="")
    """

    def __init__(self, model_path: str, config: GenerationConfig | None = None):
        self._model_path = model_path
        self._config = config or GenerationConfig()

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "请安装 llama-cpp-python: pip install llama-cpp-python\n"
                "GPU 版: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )

        # verbose=False 避免 llama.cpp 打印大量调试信息
        self._llm = Llama(
            model_path=model_path,
            n_ctx=self._config.n_ctx,
            n_threads=self._config.n_threads,
            n_gpu_layers=self._config.n_gpu_layers,
            n_batch=self._config.n_batch,
            seed=self._config.seed,
            verbose=False,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """完整生成

        返回格式: {"choices": [{"text": "生成的文本"}]}
        """
        output = self._llm(
            prompt,
            max_tokens=kwargs.get("max_tokens", self._config.max_tokens),
            temperature=kwargs.get("temperature", self._config.temperature),
            top_p=kwargs.get("top_p", self._config.top_p),
            top_k=kwargs.get("top_k", self._config.top_k),
            repeat_penalty=kwargs.get("repeat_penalty", self._config.repeat_penalty),
            echo=False,  # 不在输出中重复 prompt
        )
        return output["choices"][0]["text"]

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成

        设置 stream=True 后，llama.cpp 每生成一个 token 就返回。
        适合在 CLI 中实现打字机效果。
        """
        output = self._llm(
            prompt,
            max_tokens=kwargs.get("max_tokens", self._config.max_tokens),
            temperature=kwargs.get("temperature", self._config.temperature),
            top_p=kwargs.get("top_p", self._config.top_p),
            top_k=kwargs.get("top_k", self._config.top_k),
            repeat_penalty=kwargs.get("repeat_penalty", self._config.repeat_penalty),
            echo=False,
            stream=True,  # 开启流式
        )
        for item in output:
            yield item["choices"][0]["text"]

    def __repr__(self) -> str:
        return f"LlamaCppEngine(model={self._model_path}, ctx={self._config.n_ctx})"


class MockLLM(BaseLLM):
    """测试用 Mock LLM

    不加载任何模型，直接返回固定回答。
    用于单元测试和 CI 管道（不需要下载 GB 级模型文件）。

    使用示例:
        llm = MockLLM("这是一个测试回答。")
        llm.generate("任意 prompt")  # → "这是一个测试回答。"
    """

    def __init__(self, fixed_response: str = "这是一个模拟回答。"):
        self._response = fixed_response

    def generate(self, prompt: str, **kwargs) -> str:
        return self._response

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        # 逐字符返回，模拟流式效果
        for char in self._response:
            yield char

    def __repr__(self) -> str:
        return "MockLLM()"
