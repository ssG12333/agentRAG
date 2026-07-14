# agentRAG

> 🚧 开发中 | 本地轻量 RAG Agent — 从零搭建，理解全链路

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 这是什么？

一套**从零搭建**的本地 RAG（检索增强生成）系统，支持 Agent 智能体能力。

**不依赖** LangChain / LlamaIndex / FAISS / Milvus 等高层框架，每一层都自己实现：

| 层 | 自研内容 |
|----|---------|
| 文档层 | MD/TXT 解析器、递归分块器 |
| 嵌入层 | ONNX INT8 量化推理、BGE 模型封装 |
| 索引层 | IVF-PQ 稠密索引 (C++)、BM25 稀疏索引 (C++) |
| 检索层 | 混合检索 (RRF)、Cross-encoder 重排序、查询改写 |
| 生成层 | llama.cpp 推理、KV Cache 管理、Prefix Caching |
| Agent 层 | ReAct 循环、工具系统、对话记忆 |

## 快速开始

```bash
# 安装
pip install -e .

# 索引文档
agentrag index --path ./docs/

# 问答
agentrag ask "Transformer 中的自注意力计算复杂度是多少？"

# 交互对话
agentrag chat

# API 服务
agentrag serve
```

## 技术栈

- **嵌入模型**: BGE-small-zh / BGE-base-zh (ONNX INT8)
- **生成模型**: Qwen2.5-3B-Instruct (GGUF Q4_K_M)
- **向量索引**: 自研 IVF-PQ (C++ + pybind11)
- **稀疏检索**: 自研 BM25 (C++ 倒排索引)
- **推理引擎**: llama.cpp (`llama-cpp-python`)
- **Agent**: ReAct + 工具调用

## 项目结构

```
agentRAG/
├── src/
│   ├── document/       # Layer 1: 文档解析/分块
│   ├── embedding/      # Layer 2: 嵌入模型
│   ├── index/          # Layer 3: 向量/稀疏索引
│   ├── retrieval/      # Layer 4: 检索管道
│   ├── generation/     # Layer 5: LLM 生成 + KV Cache
│   ├── agent/          # Layer 6: Agent 智能体
│   ├── cli/            # CLI 入口
│   ├── core/           # C++ 加速模块
│   ├── api/            # REST API
│   └── utils/          # 工具函数
├── configs/            # 配置文件
├── tests/              # 测试
├── examples/           # 示例
└── docs/               # 文档
```

## 开发状态

- [x] Phase 0: 项目脚手架
- [ ] Phase 1: MVP 基础 RAG 管道
- [ ] Phase 2: 混合检索 + C++ 加速
- [ ] Phase 3: Agent 能力 + Prefix Caching
- [ ] Phase 4: 量化优化 + KV Cache INT8
- [ ] Phase 5: 生产化

## License

MIT © 2026 ssG12333
