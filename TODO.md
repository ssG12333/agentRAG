# agentRAG TODO

> 当前状态：Phase 2/3 收尾中；完整回归 29/29，通过主链路集成和基线验收后再进入 Phase 4

---

## Phase 0: 项目脚手架 ✅

- [x] 0.1 创建目录结构
- [x] 0.2 pyproject.toml
- [x] 0.3 CMakeLists.txt (C++ 构建)
- [x] 0.4 CLI 入口骨架 (agentrag --help)
- [x] 0.5 pybind C++ 骨架 (hello.cpp)
- [x] 0.6 README.md
- [x] 0.7 configs/default.yaml

## Phase 1: MVP 基础 RAG 管道 ✅

- [x] 1.1 数据模型 (Document / Chunk)
- [x] 1.2 文档解析器 (MD / TXT)
- [x] 1.3 文本分块器 (递归 / 固定窗口)
- [x] 1.4 嵌入模型 (BGE + 缓存)
- [x] 1.5 向量存储 (numpy + 持久化)
- [x] 1.6 检索器
- [x] 1.7 提示词模板
- [x] 1.8 生成引擎 (llama.cpp CPU / Mock)
- [x] 1.8b KV Cache 内存监控
- [x] 1.9 CLI index 命令
- [x] 1.10 CLI ask 命令
- [x] 1.11 端到端示例
- [x] 1.12 单元测试 (11/11)

## Phase 2: 混合检索 + C++ 加速 🔄

- [x] 2.1 C++ 向量数据结构 (vector_types.h)
- [x] 2.2 C++ K-Means 聚类
- [x] 2.3 C++ IVF 倒排索引
- [x] 2.4 C++ PQ 乘积量化
- [ ] 2.5 IVF-PQ 组合索引与 Python VectorStore（当前 IVF/PQ 仍为独立组件）
- [x] 2.6 中文分词器（Python jieba 可用）
- [x] 2.7 BM25 倒排索引 (C++)
- [x] 2.8 pybind11 绑定源码
- [x] 2.9 C++ 首次编译通过 (Ninja + MSVC 2026 + conda pybind11)
- [x] 2.9b 重编最新 SearchResult 绑定并验证扩展导出
- [x] 2.10 Python 稀疏存储层 (SparseRetriever)
- [x] 2.11 混合检索融合 (RRF)
- [x] 2.12 Cross-encoder 重排序组件
- [ ] 2.13 HybridRetriever + reranker 接入 index/ask/chat 主链路
- [x] 2.14 Python/C++ 集成回归（完整测试 29/29）
- [ ] 2.15 C++ 正确性测试与 numpy/C++ 基准

## Phase 3: Agent + Prefix Caching 🔄

- [x] 3.1 工具抽象 (Tool / ToolRegistry / XML格式)
- [x] 3.2 内置工具 (search_knowledge_base / list_documents)
- [x] 3.3 对话记忆 (Conversation / Working)
- [x] 3.4a Prefix Cache 管理器 (LRU + hit rate)
- [ ] 3.4b PrefixAwareEngine 接入 Agent；`/cache` 产生真实命中统计
- [x] 3.5 ReAct 循环 (Thought→Action→Observation)
- [x] 3.6 查询改写 (指代消解, LLM fallback)
- [x] 3.7 Agent CLI (chat 交互式 / /clear / /docs)
- [x] 3.8 Phase 3 组件测试 (13/13)
- [ ] 3.9 Agent 集成测试（工具调用 → Observation → Final Answer）
- [ ] 3.10 真实 GGUF 最小对话验证（无模型时保留 Mock 回归）

## 近期执行计划：Phase 2/3 收尾与基线建设

### P0：恢复可信全绿状态

- [x] 使用当前源码重新编译 `agentrag_core`
- [x] 验证 `hasattr(agentrag_core, "SearchResult") is True`
- [x] 执行完整测试并达到 29/29 passed
- [x] 同步 README、TODO、LOG 中的测试数字和阶段状态
- [x] 审查当前未提交改动，只提交本轮相关文件

### P1：完成主链路集成

- [ ] 明确 Phase 2 范围：实现真正 IVF-PQ，或将对外描述调整为 IVF/PQ 教学原型
- [ ] 将 HybridRetriever 和 reranker 接入 CLI 检索流程
- [ ] 将 PrefixAwareEngine 接入 ReActAgent，或明确降级为缓存管理实验
- [ ] 增加检索、Agent 和 CLI 端到端测试
- [ ] 完成 C++ 正确性测试和性能基准

### P2：建立 Phase 4 基线

- [ ] 建立固定查询—相关 chunk 评估集
- [ ] 实现 Recall@k、MRR、nDCG 和延迟统计
- [ ] 固定模型、数据、线程数、batch、随机种子和运行命令
- [ ] 保存 FP32 嵌入的精度、延迟、内存、模型大小原始结果
- [ ] 记录量化假设、支持/推翻条件和资源上限

### Phase 4 启动门槛

- [x] P0 全部完成，完整回归全绿
- [ ] P1 的范围决策和关键主链路集成完成
- [ ] P2 基线可以重复运行并输出原始数据

## Phase 4: 量化优化

- [ ] 4.1 嵌入模型 ONNX FP32 导出与一致性验证
- [ ] 4.2 ONNX Runtime 推理
- [ ] 4.3 INT8 量化与可回退配置
- [ ] 4.4 嵌入量化基准（精度/延迟/内存/模型大小）
- [ ] 4.5 生成模型量化格式对比 (Q4/Q8/FP16)
- [ ] 4.6 KV Cache INT8 可行性实验（研究性，不阻塞核心交付）
- [ ] 4.7 量化报告

## Phase 5: 生产化

- [ ] 5.1 FastAPI REST 服务
- [ ] 5.2 CLI serve 命令
- [ ] 5.3 配置系统
- [ ] 5.4 结构化日志
- [ ] 5.5 Dockerfile
- [ ] 5.6 文档完善
- [ ] 5.7 CI 配置

---

## 待定 / 未来

- [ ] llama.cpp GPU 版（需升级 CUDA 11.8 → 13.x）
- [ ] 下载真实 GGUF 模型（Qwen2.5-3B Q4_K_M）
- [ ] PDF 解析器
