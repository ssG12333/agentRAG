# 13｜量化预研

> 状态：**预研** ｜ 当前项目没有 ONNX INT8 嵌入实现；本章实验仅是 NumPy 玩具量化。

## 学习目标与先修知识

- 理解 FP32 到 INT8 的线性映射、scale 与误差来源；
- 区分 dynamic quantization、static quantization 和模型文件压缩；
- 建立“速度/体积收益不能掩盖检索质量下降”的验收门槛；
- 不把规划中的 ONNX 路线写成当前能力。

先修：向量、NumPy 与 [03｜嵌入模型](03_embeddings.md)。

## 对称线性量化

对一组 FP32 值 `x`，最简单的 per-tensor 对称 INT8 量化为：

\[
s=\frac{\max|x|}{127},\quad
q=clip(round(x/s),-127,127),\quad
\hat{x}=s q
\]

`q` 只占 1 byte，而 FP32 占 4 bytes，所以纯数值 payload 理论上约为四分之一；scale、元数据、算子支持和内存对齐会让真实模型不一定正好缩小 4 倍。

量化误差可用 MAE、最大绝对误差或余弦相似度衡量。对 RAG 更重要的是：误差是否改变近邻排序与最终 Recall/MRR。

## Dynamic 与 Static

- **Dynamic quantization**：权重预先量化，激活范围在运行时计算；部署简单，但运行时有额外开销。
- **Static quantization**：权重和激活范围都通过 calibration 数据预估；潜在性能更好，但校准集必须代表真实输入。

ONNX Runtime 的具体算子、执行提供器和硬件支持可能变化，实施时必须查当时的官方兼容矩阵。本仓库目前没有 ONNX 导出、校准或 INT8 推理链路。

## 快速实验

```powershell
python examples/learning/run_lab.py --lab 13
```

实验会对一个小 FP32 向量进行量化和反量化，输出理论 payload 比、MAE 与余弦相似度。它只验证公式，不代表 BGE、ONNX 或实际硬件加速结果。

## 未来实验的最低门槛

先建立 FP32 基线，然后只改变精度：

| 维度 | 必须记录 |
|---|---|
| 质量 | Recall@k、MRR/nDCG、多种子均值与波动 |
| 性能 | warm-up 后延迟、吞吐、峰值内存、模型体积 |
| 公平性 | 相同数据、batch、硬件、线程与执行提供器 |
| 回退 | 质量低于门槛或目标硬件无加速时继续使用 FP32 |

可证伪假设示例：

> INT8 通过减少权重带宽降低 CPU 嵌入延迟，同时 Recall@10 绝对下降不超过 0.01；任一条件不成立就不进入默认路径。

## 常见错误与反例

- 把文件变小直接等同于推理更快；
- 只测向量重建误差，不测检索排序；
- 用测试集做 calibration，造成数据泄漏；
- 混用不同 batch、线程或 warm-up 次数；
- 将本章 NumPy 输出称为“ONNX INT8 已验证”。

## 练习题

1. 为什么 scale 不能简单固定为 1？
2. 一个异常大值会怎样影响 per-tensor 量化？
3. 若模型缩小但延迟没有下降，应检查什么？

<details><summary>参考答案</summary>

1. 不同张量数值范围不同，固定 scale 可能导致大量裁剪或精度浪费。
2. 它会放大 scale，使多数小值挤在较少的整数档位上；可考虑 per-channel 或异常值处理。
3. 检查算子是否真正走 INT8 kernel、量化/反量化开销、执行提供器、线程、输入 batch 和内存瓶颈。

</details>

## 完成检查表

- [ ] 我能从 FP32 数组手算对称 INT8 量化。
- [ ] 我不会把 4 倍 payload 压缩写成必然的 4 倍端到端加速。
- [ ] 我能写出包含质量门槛的量化假设。
- [ ] 我知道当前仓库尚无 ONNX INT8 实现。

## 原始资料

- [ONNX Runtime：Model Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

上一章：[12｜评估与实验](12_evaluation.md) ｜ 下一章：[14｜真实 Prefix KV 预研](14_prefix_kv_preresearch.md)
