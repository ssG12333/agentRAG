"""生成引擎 —— llama.cpp 封装 + MockLLM"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class GenerationConfig:
    """生成参数"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 1024
    repeat_penalty: float = 1.1
    # llama.cpp 特有
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0
    n_batch: int = 512
    seed: int = 42


class BaseLLM(ABC):
    """LLM 抽象基类"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成完整回答"""
        ...

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成回答"""
        ...


class LlamaCppEngine(BaseLLM):
    """基于 llama-cpp-python 的推理引擎"""

    def __init__(self, model_path: str, config: GenerationConfig | None = None):
        self._model_path = model_path
        self._config = config or GenerationConfig()

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "请安装 llama-cpp-python: pip install llama-cpp-python"
            )

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
        output = self._llm(
            prompt,
            max_tokens=kwargs.get("max_tokens", self._config.max_tokens),
            temperature=kwargs.get("temperature", self._config.temperature),
            top_p=kwargs.get("top_p", self._config.top_p),
            top_k=kwargs.get("top_k", self._config.top_k),
            repeat_penalty=kwargs.get("repeat_penalty", self._config.repeat_penalty),
            echo=False,
        )
        return output["choices"][0]["text"]

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        output = self._llm(
            prompt,
            max_tokens=kwargs.get("max_tokens", self._config.max_tokens),
            temperature=kwargs.get("temperature", self._config.temperature),
            top_p=kwargs.get("top_p", self._config.top_p),
            top_k=kwargs.get("top_k", self._config.top_k),
            repeat_penalty=kwargs.get("repeat_penalty", self._config.repeat_penalty),
            echo=False,
            stream=True,
        )
        for item in output:
            yield item["choices"][0]["text"]

    def __repr__(self) -> str:
        return f"LlamaCppEngine(model={self._model_path}, ctx={self._config.n_ctx})"


class MockLLM(BaseLLM):
    """测试用 Mock LLM —— 返回固定回答，无需模型文件"""

    def __init__(self, fixed_response: str = "这是一个模拟回答。"):
        self._response = fixed_response

    def generate(self, prompt: str, **kwargs) -> str:
        return self._response

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        for char in self._response:
            yield char

    def __repr__(self) -> str:
        return "MockLLM()"
