# 04｜精确向量检索

> 状态：**已实现** ｜ 路径：快速离线

## 学习目标与先修知识

- 理解精确全量搜索为何是正确性基线。
- 掌握矩阵内积、`argpartition` 和 top-k 内部排序。
- 能验证向量、Chunk 与 metadata 的持久化一致性。

## 当前实现边界

`NumpyVectorStore` 保存所有 float32 向量，查询复杂度为 `O(Nd)`，适合小规模基线。它不会自动去重，也不会替调用方校验 Chunk 数量；课程实验必须保持向量行与 Chunk 一一对应。

## 概念直觉与核心公式

设索引矩阵 `V∈R^(N×d)`，归一化查询 `q∈R^d`：

```text
scores = V q
time = O(Nd)
vector payload = N · d · 4 bytes
```

完整 `argsort` 需要排序所有 N 个值；`argpartition` 先以近似 `O(N)` 找出候选集合，再只对 k 个候选排序。它改变的是选择成本，不改变分数本身。

## 项目调用链

- `add()`：转 float32 → 行归一化 → 拼接向量和 Chunk。
- `search()`：归一化 query → 全量内积 → 选择并排序 top-k。
- `save()`：把向量、Chunk 字段和 JSON metadata 写入压缩 NPZ。
- `load()`：恢复当前格式，并兼容没有 `chunk_metadata` 的旧索引。
- 回归测试：`tests/test_phase1.py` 与 `tests/test_retrieval_pipeline.py`。

## 最小实验

```powershell
python examples/learning/run_lab.py --lab 04
```

实验在临时目录保存 NPZ，再加载到新对象。预期现象：保存前后第一名 Chunk ID 和 metadata 一致，临时文件在实验结束后自动清理。

## 常见错误、边界与反例

- 只保存向量而丢失 Chunk 顺序，会把分数映射到错误文本。
- `top_k` 大于数据量时返回全部结果，这是允许行为。
- 精确搜索提供基准，但数据量增长时延迟和内存线性增长。
- 单次随机查询的第一名正确不能证明整个评估集 Recall 为 1；精确基线是相对于同一向量空间定义的。

## 练习

1. 10 万个 768 维 float32 向量的纯向量载荷约多少？
2. 为什么加载旧索引时 metadata 使用空字典，而不是拒绝加载？

<details><summary>参考答案</summary>

1. `100000×768×4=307,200,000` 字节，约 293 MiB，尚未计算 Python 对象和文件结构开销。2. 这是向后兼容策略：旧数据仍可检索，只是来源等附加信息缺失；调用方应显式处理缺失字段。

</details>

## 完成检查

- [ ] 能解释 `argpartition` 后为何还要局部排序。
- [ ] 能估算 float32 索引载荷。
- [ ] 能验证持久化前后的 ID、内容和 metadata。

上一章：[03｜嵌入模型](03_embeddings.md) ｜ 下一章：[05｜C++ 与 pybind11](05_cpp_pybind_kmeans.md)
