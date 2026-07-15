"""
================================================================================
Layer 5: KV Cache 监控 —— 上下文窗口使用率追踪
================================================================================

KV Cache 是什么？
  Transformer 推理时，每个 token 的 Key 和 Value 需要与之前所有 token
  交互（自注意力）。KV Cache 缓存了已生成 token 的 K/V，避免每个新 token
  都重新计算整个序列。

为什么 RAG 需要关心 KV Cache？
  - 检索出来的 chunk 越多 → 上下文越长 → KV Cache 越大 → 显存越紧张
  - 当 KV Cache 快满时（接近 n_ctx），必须减少检索数量或截断上下文
  - Prefix Caching (Phase 3) 可以复用系统提示 + 检索结果的 KV Cache

API 限制（Phase 1）：
  llama-cpp-python 不暴露 KV Cache 的内存读写接口。
  当前只能估算，精确监控需 fork llama.cpp（Phase 4 研究）。

设计思路：
  - KVCacheMonitor 是独立模块，不侵入生成引擎
  - 生成前预估上下文占用，生成后追踪实际使用
  - 检索器可根据剩余容量动态调整 top_k
"""

from dataclasses import dataclass


@dataclass
class KVCacheStats:
    """KV Cache 状态快照

    total_slots: 总槽位数 = n_ctx（上下文窗口大小）
    used_slots:  当前已占用的槽位数（= 已生成 tokens 数 + prompt tokens 数）
    dtype:       存储精度（FP16 / INT8），影响内存占用估算

    内存估算公式 (FP16):
      KV_Cache_bytes = 2 * n_layers * n_kv_heads * n_ctx * head_dim * 2
      以 Qwen2.5-3B 为例（2 个 KV heads）:
      2 * 36 * 2 * 4096 * 128 * 2 = 144 MiB（理论 payload）
      实际占用仍受 runtime 布局、对齐和 batch 等因素影响。
    """
    total_slots: int = 0
    used_slots: int = 0
    dtype: str = "FP16"

    @property
    def free_slots(self) -> int:
        """剩余可用槽位"""
        return self.total_slots - self.used_slots

    @property
    def usage_pct(self) -> float:
        """使用率 0.0 ~ 1.0"""
        if self.total_slots == 0:
            return 0.0
        return self.used_slots / self.total_slots

    @property
    def memory_bytes(self) -> int:
        """估算 KV Cache 内存占用（字节）

        Phase 1 返回 0（无法从 llama-cpp-python 获取准确值）。
        Phase 4 接入 llama.cpp C API 后精确计算。
        """
        return 0


class KVCacheMonitor:
    """KV Cache 使用率监控器

    用法示例:
        monitor = KVCacheMonitor(total_slots=4096, warn_threshold=0.9)
        # 索引阶段：预估每个 chunk 占 ~500 tokens
        max_chunks = monitor.suggest_max_chunk_tokens(chunk_size_estimate=500)
        # 生成阶段：追踪使用量
        monitor.update(used_slots=3500)
        if monitor.is_low():
            print("警告：KV Cache 接近容量上限")
    """

    def __init__(self, total_slots: int = 4096, warn_threshold: float = 0.9):
        """初始化监控器

        Args:
            total_slots: 上下文窗口大小 = llama.cpp 的 n_ctx
            warn_threshold: 警告阈值，默认 90%（余量不足 10% 时提示）
        """
        self.total_slots = total_slots
        self.warn_threshold = warn_threshold
        self._used_slots = 0

    def update(self, used_slots: int) -> None:
        """更新当前使用量

        由生成引擎在 prefill 完成后回调。
        used_slots = 系统提示 tokens + 检索结果 tokens + 历史 tokens + 用户问题 tokens
        """
        self._used_slots = used_slots

    def reset(self) -> None:
        """重置使用量（新一轮对话开始时调用）"""
        self._used_slots = 0

    def stats(self) -> KVCacheStats:
        """获取当前状态快照"""
        return KVCacheStats(
            total_slots=self.total_slots,
            used_slots=self._used_slots,
        )

    def is_low(self) -> bool:
        """剩余容量是否不足（超过警告阈值）"""
        return self.stats().usage_pct >= self.warn_threshold

    def suggest_max_chunk_tokens(self, chunk_size_estimate: int = 512) -> int:
        """根据剩余容量建议最大可检索的 chunk 数

        流程：
          1. 计算剩余槽位
          2. 预留 200 tokens 给系统提示和用户问题
          3. 用剩余槽位除以预估的每块 token 数

        Args:
            chunk_size_estimate: 每个 chunk 大约占用的 token 数
                                （中文约 1.5 chars/token，512 chars ≈ 340 tokens）

        Returns:
            建议的最大 chunk 数（至少为 1）
        """
        free = self.stats().free_slots
        if free <= 0:
            return 1
        # 预留空间给系统提示和问答格式 (~200 tokens)
        usable = max(0, free - 200)
        return max(1, usable // chunk_size_estimate)
