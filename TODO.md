# agentRAG TODO

> 当前状态：P1 主链路集成完成；完整回归 46/46，进入 P2 评估基线建设

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

## Phase 2: 混合检索 + C++ 加速 ✅

- [x] 2.1 C++ 向量数据结构 (vector_types.h)
- [x] 2.2 C++ K-Means 聚类
- [x] 2.3 C++ IVF 倒排索引
- [x] 2.4 C++ PQ 乘积量化
- [x] 2.5 残差 IVF-PQ 组合索引、持久化与 Python VectorStore
- [x] 2.6 中文分词器（Python jieba 可用）
- [x] 2.7 BM25 倒排索引 (C++)
- [x] 2.8 pybind11 绑定源码
- [x] 2.9 C++ 首次编译通过 (Ninja + MSVC 2026 + conda pybind11)
- [x] 2.9b 重编最新 SearchResult 绑定并验证扩展导出
- [x] 2.10 Python 稀疏存储层 (SparseRetriever)
- [x] 2.11 混合检索融合 (RRF)
- [x] 2.12 Cross-encoder 重排序组件
- [x] 2.13 可配置 dense/IVF-PQ + BM25/RRF + reranker 接入 index/ask/chat 主链路
- [x] 2.14 Python/C++ 集成回归（完整测试 46/46）
- [x] 2.15a IVF-PQ C++/Python 正确性与持久化测试
- [x] 2.15b numpy/C++ 合成性能、召回率与内存基准

## Phase 3: Agent + Prefix Caching 🔄

- [x] 3.1 工具抽象 (Tool / ToolRegistry / XML格式)
- [x] 3.2 内置工具 (search_knowledge_base / list_documents)
- [x] 3.3 对话记忆 (Conversation / Working)
- [x] 3.4a Prefix Cache 管理器 (LRU + hit rate)
- [x] 3.4b PrefixAwareEngine 接入 Agent；`/cache` 输出逻辑命中统计
- [ ] 3.4c 真实 KV Cache prefill 复用（研究性，不属于 P1）
- [x] 3.5 ReAct 循环 (Thought→Action→Observation)
- [x] 3.6 查询改写 (指代消解, LLM fallback)
- [x] 3.7 Agent CLI (chat 交互式 / /clear / /docs)
- [x] 3.8 Phase 3 组件测试 (15/15)
- [x] 3.9 Agent 集成测试（工具调用 → Observation → Final Answer）
- [ ] 3.10 真实 GGUF 最小对话验证（无模型时保留 Mock 回归）

## 近期执行计划：Phase 2/3 收尾与基线建设

### P0：恢复可信全绿状态

- [x] 使用当前源码重新编译 `agentrag_core`
- [x] 验证 `hasattr(agentrag_core, "SearchResult") is True`
- [x] 执行完整测试并达到 29/29 passed
- [x] 同步 README、TODO、LOG 中的测试数字和阶段状态
- [x] 审查当前未提交改动，只提交本轮相关文件

### P1：完成主链路集成

- [x] 明确 Phase 2 范围：实现真正的残差 IVF-PQ
- [x] 将 HybridRetriever 和 reranker 接入 CLI 检索流程
- [x] 将 PrefixAwareEngine 作为逻辑缓存管理实验接入 ReActAgent
- [x] 增加检索、Agent 和 CLI 端到端测试
- [x] 完成 C++ 正确性测试和合成性能基准

### P2：建立 Phase 4 基线

- [ ] 建立固定查询—相关 chunk 评估集
- [ ] 实现 Recall@k、MRR、nDCG 和延迟统计
- [ ] 固定模型、数据、线程数、batch、随机种子和运行命令
- [ ] 保存 FP32 嵌入的精度、延迟、内存、模型大小原始结果
- [ ] 记录量化假设、支持/推翻条件和资源上限

### Phase 4 启动门槛

- [x] P0 全部完成，完整回归全绿
- [x] P1 的范围决策和关键主链路集成完成
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

## 系统学习课程

- [x] 建立 `docs/learning/` 课程索引、章节模板、术语表和参考资料
- [ ] 完成 01-04 基础 RAG 章节与实验
- [ ] 完成 05-08 C++ 与检索算法章节与实验
- [ ] 完成 09-11 生成、Agent、CLI 章节与实验
- [ ] 完成 12-15 评估和预研章节与实验
- [ ] 增加学习实验 smoke test 和 Markdown 链接校验
