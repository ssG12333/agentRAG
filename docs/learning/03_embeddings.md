# 03｜嵌入模型

> 状态：**已实现** ｜ 路径：快速离线使用 ToyEmbedding；真实路径使用 BGE

## 学习目标与先修知识

- 理解 Bi-Encoder 为什么适合先离线编码文档，再在线编码 query。
- 掌握 L2 归一化、内积和余弦相似度的关系。
- 看懂 `BaseEmbedding` 抽象及 `CachedEmbedding` 的真实淘汰行为。

## 当前实现边界

`SentenceTransformerEmbedding` 支持 BGE 等模型并可归一化输出；首次使用会下载模型。快速实验用确定性 ToyEmbedding，不能替代真实语义质量。当前 `CachedEmbedding` 使用普通有序字典，命中时不会移动条目，因此严格说是**近似 FIFO**，不是完整 LRU。

## 概念直觉与核心公式

对向量 `x`、`y`：

```text
cos(x, y) = (x · y) / (||x||₂ ||y||₂)
```

若二者已归一化为单位向量，则 `cos(x,y)=x·y`。这就是项目在写入向量库时统一归一化、查询时只做矩阵乘法的原因。

Bi-Encoder 独立计算 `e_q=f(query)` 和 `e_d=f(document)`，文档向量可以预计算。Cross-Encoder 则需要每个 `(query, document)` 对共同前向，精度潜力更高但不适合全库扫描。

## 项目调用链

- `BaseEmbedding.embed/embed_query/dim` 定义稳定接口。
- `SentenceTransformerEmbedding` 延迟到实例化时导入第三方库。
- `CachedEmbedding.embed()` 分离命中和未命中，批量计算后恢复输入顺序。
- `Retriever.retrieve()` 只依赖 `embed_query()`，不知道具体模型。

## 最小实验

```powershell
python examples/learning/run_lab.py --lab 03
```

预期现象：共享关键词的文本余弦相似度更高；第二次嵌入同一文本后缓存命中率上升。ToyEmbedding 只用于解释数学关系，不应据此评价中文语义模型。

真实 BGE 路径：

```powershell
python examples/learning/real_model_lab.py --component embedding
```

缺少模型或依赖时脚本必须显示 `SKIPPED`，不会回退到 ToyEmbedding 后冒充真实结果。

## 常见错误、边界与反例

- 索引与查询使用不同模型或维度会使结果无意义或直接报错。
- 零向量无法表达方向；项目为避免除零保留零向量，但它不会与正常文本相似。
- 缓存命中率高只表示输入字符串重复，不表示检索质量高。
- 归一化会丢失向量模长信息，这是使用余弦相似度的有意选择。

## 练习

1. 为什么归一化后可以用内积替代余弦计算？
2. 如何把当前近似 FIFO 缓存改成真正 LRU？

<details><summary>参考答案</summary>

1. 单位向量的两个模长都为 1，余弦公式分母消失。2. 使用 `OrderedDict`，每次命中调用 `move_to_end()`，超容量时从头部淘汰；还要增加命中后顺序变化的测试。

</details>

## 完成检查

- [ ] 能手算两个二维向量的余弦相似度。
- [ ] 能区分 Bi-Encoder 召回与 Cross-Encoder 重排。
- [ ] 不把 ToyEmbedding 当作真实语义模型。

## 原始资料

- Reimers and Gurevych, [Sentence-BERT](https://aclanthology.org/D19-1410/).

上一章：[02｜文档处理](02_document_processing.md) ｜ 下一章：[04｜精确向量检索](04_exact_vector_search.md)
