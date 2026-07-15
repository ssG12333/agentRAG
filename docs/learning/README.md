# agentRAG 系统学习课程

这是一套以当前 agentRAG 源码为实验对象的中文课程，面向已经会 Python、希望系统理解 RAG 的学习者。课程不是对早期规划的复述，而是从可运行代码、测试和原始实验结果出发，解释每个组件为什么存在、怎样连接、如何被证伪。

## 两条学习路径

- **快速离线路径**：Mock LLM、小型确定性向量、CPU，不下载 GGUF 或 Cross-Encoder。
- **真实模型路径**：BGE、Cross-Encoder、Qwen GGUF；章节会先做依赖和文件检查，缺少资源时明确显示 `SKIPPED`。

## 状态图例

| 状态 | 含义 |
|---|---|
| 已实现 | 当前源码和测试已经覆盖，可直接运行实验 |
| 部分实现 | 接口或逻辑存在，但真实模型、性能收益或底层能力尚未验证 |
| 预研 | 只讲原理、风险和最小玩具实验，不代表项目已有该功能 |

## 课程目录

| 章节 | 状态 | 主题 |
|---|---|---|
| [00](00_course_guide.md) | 已实现 | 课程指南、环境和实验规范 |
| [01](01_rag_architecture.md) | 已实现 | RAG 六层架构与端到端数据流 |
| [02](02_document_processing.md) | 已实现 | Document、Chunk、解析与分块 |
| [03](03_embeddings.md) | 已实现 | 嵌入、归一化、余弦相似度与缓存 |
| [04](04_exact_vector_search.md) | 已实现 | NumPy 精确检索与持久化 |
| [05](05_cpp_pybind_kmeans.md) | 已实现 | C++、pybind11 与 K-Means |
| [06](06_residual_ivfpq.md) | 已实现 | 残差 IVF-PQ 与近似检索权衡 |
| [07](07_bm25.md) | 已实现 | BM25 稀疏检索 |
| [08](08_hybrid_rerank.md) | 部分实现 | RRF、混合检索与可选重排 |
| [09](09_prompt_generation_kv.md) | 部分实现 | Prompt、llama.cpp、流式生成与 KV 监控 |
| [10](10_agent.md) | 部分实现 | Tool、Memory、ReAct 与逻辑 Prefix Cache |
| [11](11_cli_integration.md) | 已实现 | index/ask/chat 集成与调试 |
| [12](12_evaluation.md) | 部分实现 | Recall、MRR、nDCG 与实验方法 |
| [13](13_quantization_preresearch.md) | 预研 | FP32/INT8 与 ONNX 量化路线 |
| [14](14_prefix_kv_preresearch.md) | 预研 | 真实 Prefix KV 复用的机制与限制 |
| [15](15_production_preresearch.md) | 预研 | 配置、API、日志、Docker 与 CI |

补充阅读：[术语表](glossary.md) · [原始资料](references.md) · [章节模板](_chapter_template.md)

## 事实来源顺序

1. 当前源码与测试。
2. `LOG.md`、`TODO.md` 和原始 CSV。
3. README。
4. `PLAN.md` 只作历史背景，不能作为完成状态证据。

## 完成标准

读完后应能解释一条查询如何穿过六层架构，能运行对应实验，能指出精确检索与近似检索、稠密与稀疏、Mock 与真实模型、逻辑缓存与真实 KV 复用之间的边界。
