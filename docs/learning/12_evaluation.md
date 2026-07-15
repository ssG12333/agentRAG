# 12｜评估与实验方法

> 状态：**部分实现** ｜ P1 合成 Recall、延迟与内存基准已完成；真实语义 qrels 评估仍待建设。

## 学习目标与先修知识

学完本章，你应该能够：

- 区分“向量近邻是否找回”和“答案是否相关”两类评价目标；
- 计算 Recall@k、MRR 与 nDCG@k；
- 设计固定变量、可复现、可证伪的检索实验；
- 正确解释当前 `Recall@10=0.23` 的负面结果。

先修：完成 [04｜精确向量检索](04_exact_vector_search.md) 和 [06｜残差 IVF-PQ](06_residual_ivfpq.md)。

## 指标不是越多越好

设查询的相关文档集合为 `Rel`，前 `k` 个结果为 `R_k`：

\[
Recall@k = \frac{|Rel \cap R_k|}{|Rel|}
\]

若第一个相关结果位于排名 `r`，则 `RR=1/r`，多查询平均后得到 `MRR`。当相关性有等级时：

\[
DCG@k=\sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)},
\quad nDCG@k=\frac{DCG@k}{IDCG@k}
\]

`Recall@k` 关心是否找全，`MRR` 强调第一个正确答案的位置，`nDCG` 同时考虑相关等级与排序。指标必须由业务问题决定，不能只挑结果最好看的一个。

## 当前基准到底测了什么

真实调用链：

```text
随机单位向量与查询
  → NumpyVectorStore 精确 top-k（对照）
  → IVFPQIndex 近似 top-k（实验）
  → 集合交集计算 Recall@k
  → CSV 记录延迟、索引体积和配置
```

源码定位：

- `scripts/benchmark_retrieval.py`：固定种子、运行两种后端并写 CSV；
- `tests/test_ivfpq_index.py`：小规模正确性、持久化和边界测试；
- [`docs/benchmarks/retrieval_p1.csv`](../benchmarks/retrieval_p1.csv)：P1 原始结果。

这里的“相关集合”来自精确向量 top-k，而不是人工语义标注。因此它能评估 ANN 近似误差，却不能证明系统回答质量。

## 必须保留的反例：Recall@10 = 0.23

P1 合成基准中，IVF-PQ 在对应配置下得到 `Recall@10=0.23`。这意味着精确结果中的十个近邻，平均只找回约 2.3 个；它不是可以忽略的抖动。

因此当前默认后端仍是 NumPy 精确检索。后续只有在以下证据同时成立时，才应考虑切换默认值：

1. 固定数据、查询、种子和 `top_k`；
2. 单变量扫描 `n_probe`、`n_bits` 等参数；
3. Recall 达到预先设定门槛；
4. 延迟或内存收益足以抵偿召回损失；
5. 在真实 qrels 上也没有明显退化。

## 快速实验

先计算一个手工可核对的排序指标：

```powershell
python examples/learning/run_lab.py --lab 12
```

预期现象：输出 `Recall@3`、`RR` 和 `nDCG@4`；所有值都应位于 `[0, 1]`。

复现 P1 风格的合成基准：

```powershell
python scripts/benchmark_retrieval.py --n-vectors 1000 --dim 64 --n-queries 100 --top-k 10 --seed 42 --output tmp/learning_retrieval.csv
```

这条命令需要可导入 `agentrag_core`。若扩展不可用，应记录“未执行”，不能用另一次 Mock 输出冒充。

## 可证伪实验模板

> 假设：增大 `n_probe` 通过搜索更多倒排桶提高 Recall@10；若固定种子的 Recall 没有提升，或延迟增长超过预设预算，则不支持该参数调整。

至少记录：提交哈希、数据版本、随机种子、参数、硬件、原始 CSV、均值与波动。单次快照只能描述当前环境，不能表述为普遍性能。

## 常见错误与反例

- 用精确 ANN 邻居充当真实语义标注，却声称“问答准确率”；
- 调高多个参数后，把提升全部归因于其中一个；
- 只报告最好的一次运行，不报告失败种子；
- 用不同训练预算或不同候选池比较两个检索器；
- 因低 Recall 不好看而删除负面结果。

## 练习题

1. 相关文档为 `{d1,d3}`，排序为 `[d0,d1,d4,d3]`，计算 `Recall@3` 和 RR。
2. 为什么当前 P1 基准不能替代真实 qrels？
3. 设计一个只改变 `n_probe` 的停止条件。

<details><summary>参考答案</summary>

1. `Recall@3=1/2=0.5`；首个相关文档在第 2 位，所以 `RR=0.5`。
2. P1 只比较近似向量邻居与精确向量邻居，没有人工相关性、答案正确性或领域覆盖证据。
3. 例如固定其他参数与三个种子；若 Recall@10 均值提升不足 0.05，或 P95 延迟增长超过 50%，停止增加 `n_probe`。

</details>

## 完成检查表

- [ ] 我能手算 Recall、RR 和一个简单的 DCG。
- [ ] 我能解释 `Recall@10=0.23` 为什么阻止 IVF-PQ 成为默认后端。
- [ ] 我能区分合成 ANN 基准与真实语义评估。
- [ ] 我会在结论中记录环境、原始数据和未执行项。

## 原始资料

- [Stanford IR Book：Evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)
- 项目基准脚本：`scripts/benchmark_retrieval.py`

上一章：[11｜CLI 集成](11_cli_integration.md) ｜ 下一章：[13｜量化预研](13_quantization_preresearch.md)
