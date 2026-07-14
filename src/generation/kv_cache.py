"""KV Cache 监控 —— 上下文窗口使用率追踪"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class KVCacheStats:
    """KV Cache 状态快照"""
    total_slots: int = 0            # 总槽位 (= n_ctx)
    used_slots: int = 0             # 当前已用槽位
    dtype: str = "FP16"             # 存储精度

    @property
    def free_slots(self) -> int:
        return self.total_slots - self.used_slots

    @property
    def usage_pct(self) -> float:
        """使用率 0.0 ~ 1.0"""
        if self.total_slots == 0:
            return 0.0
        return self.used_slots / self.total_slots

    @property
    def memory_bytes(self) -> int:
        """估算 KV Cache 内存占用"""
        return 0  # TODO: 接入 llama.cpp 的实际统计


class KVCacheMonitor:
    """KV Cache 使用率监控器

    集成到检索和生成流程中，当上下文接近容量上限时发出警告或调整行为。
    """

    def __init__(self, total_slots: int = 4096, warn_threshold: float = 0.9):
        self.total_slots = total_slots
        self.warn_threshold = warn_threshold
        self._used_slots = 0

    def update(self, used_slots: int) -> None:
        """更新当前使用量（由生成引擎回调）"""
        self._used_slots = used_slots

    def reset(self) -> None:
        self._used_slots = 0

    def stats(self) -> KVCacheStats:
        return KVCacheStats(
            total_slots=self.total_slots,
            used_slots=self._used_slots,
        )

    def is_low(self) -> bool:
        """剩余容量不足"""
        return self.stats().usage_pct >= self.warn_threshold

    def suggest_max_chunk_tokens(self, chunk_size_estimate: int = 512) -> int:
        """根据剩余容量建议最大检索 chunk 数"""
        free = self.stats().free_slots
        if free <= 0:
            return 1
        # 预留 200 tokens 给系统提示和用户问题
        usable = max(0, free - 200)
        return max(1, usable // chunk_size_estimate)
