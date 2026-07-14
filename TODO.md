# agentRAG TODO

> 当前状态：Phase 1 已完成，进入 Phase 2

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

## Phase 2: 混合检索 + C++ 加速

- [ ] 2.1 C++ 向量数据结构 (vector_types.h ✅ 骨架已建)
- [ ] 2.2 C++ K-Means 聚类
- [ ] 2.3 C++ IVF 倒排索引
- [ ] 2.4 C++ PQ 乘积量化
- [ ] 2.5 IVF-PQ 组合索引
- [ ] 2.6 中文分词器 (cppjieba)
- [ ] 2.7 BM25 倒排索引 (C++)
- [ ] 2.8 pybind11 绑定
- [ ] 2.9 Python 向量存储层 (调 C++ 索引)
- [ ] 2.10 Python 稀疏存储层
- [ ] 2.11 混合检索融合 (RRF)
- [ ] 2.12 Cross-encoder 重排序
- [ ] 2.13 检索管道整合
- [ ] 2.14 Phase 2 测试 + 基准

## Phase 3: Agent + Prefix Caching

- [ ] 3.1 工具抽象 (Tool / ToolRegistry)
- [ ] 3.2 内置工具 (search / read / list / calc)
- [ ] 3.3 对话记忆 (Conversation / Working)
- [ ] 3.4 Prefix Caching 管理器
- [ ] 3.5 ReAct 循环
- [ ] 3.6 查询改写
- [ ] 3.7 Agent CLI (chat 命令)
- [ ] 3.8 Phase 3 测试
- [ ] 3.9 Agent 示例

## Phase 4: 量化优化

- [ ] 4.1 嵌入模型 ONNX 导出 + INT8 量化
- [ ] 4.2 ONNX Runtime 推理
- [ ] 4.3 嵌入量化基准
- [ ] 4.4 KV Cache INT8 量化（研究性）
- [ ] 4.5 生成模型量化格式对比 (Q4/Q8/FP16)
- [ ] 4.6 量化报告

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
- [ ] 评估体系（RAGAS 或自定义指标）
