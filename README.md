<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/C++-17-00599C?style=flat&logo=c%2B%2B&logoColor=white" alt="C++" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="Status" />
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs" />
</p>

<h1 align="center">agentRAG</h1>

<p align="center">
  <b>从零搭建 · 本地轻量 · 量化优先 · 全链路自研</b>
</p>

<p align="center">
  <i>不依赖 LangChain / LlamaIndex / FAISS / Milvus<br/>
  每个组件手写实现，理解 RAG 的每一层</i>
</p>

---

## 理念

市面上的 RAG 教程大多教你调用 LangChain 的一行 API。这能跑通，但你不会理解：

- 向量检索为什么快？IVF 的倒排列表和 PQ 的码本到底怎么工作的？
- BM25 的 TF-IDF 和词饱和度参数对中文检索有什么影响？
- KV Cache 在长上下文 RAG 中如何管理？Prefix Caching 能省多少时间？
- 本地模型如何量化？INT8 嵌入和 Q4 权重的精度损失到底多大？

**agentRAG 的目标是把这些全拆开给你看。**

## 架构

```
CLI / API  ──→  Agent React Loop  ──→  llama.cpp Generator
                    │                         │
                    │                         ├─ KV Cache Monitor
                    │                         ├─ Prefix Caching
                    │                         └─ INT8 KV Cache Quant
                    │
                    ├── Hybrid Retriever ──→ Cross-Encoder Reranker
                    │       │
                    │       ├── IVF-PQ Dense Index (C++)
                    │       └── BM25 Sparse Index (C++)
                    │
                    ├── ONNX INT8 Embedding (BGE)
                    └── Document Parser + Chunker
```

## 快速开始

```bash
# 安装
git clone https://github.com/ssG12333/agentRAG.git
cd agentRAG
pip install -e .

# 索引你的文档
agentrag index --path ./your-docs/

# 提问
agentrag ask "Transformer 的计算复杂度是多少？" --show-sources

# 交互对话 (开发中)
agentrag chat
```

## 模块清单

| 层 | 模块 | 自研内容 | 状态 |
|:--:|------|---------|:--:|
| 1 | `document/` | MD/TXT 解析器、递归分块器（分隔符优先） | ✅ |
| 2 | `embedding/` | BGE 模型封装、LRU 嵌入缓存、ONNX INT8 量化准备 | ✅ |
| 3 | `index/` | Numpy 向量存储 → Phase 2 升级为 IVF-PQ (C++) + BM25 | 🔄 |
| 4 | `retrieval/` | 检索器、RAGPromptBuilder → Phase 2 加入 RRF 混合检索 + 重排序 | 🔄 |
| 5 | `generation/` | llama.cpp 引擎、KV Cache 监控 | ✅ |
| 6 | `agent/` | Tool 系统、ReAct 循环、Prefix Caching | ⏳ |

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 嵌入模型 | BGE-small-zh / BGE-base-zh | 中文友好，轻量，支持 ONNX 导出 |
| 生成模型 | Qwen2.5-3B-Instruct (GGUF Q4) | 6GB 显存全 GPU，Agent 推理够用 |
| 推理引擎 | llama.cpp (`llama-cpp-python`) | 纯 C++ 内核，KV Cache API 完备 |
| 向量索引 | 自研 IVF-PQ (C++ + pybind11) | 不用 FAISS，从零理解索引原理 |
| 稀疏检索 | 自研 BM25 (C++) | 中文分词 + 倒排索引 |
| 分词 | cppjieba | 成熟 C++ jieba 实现 |

## 为什么不用 LangChain？

| LangChain 组件 | agentRAG 替代 | 你学到什么 |
|---------------|-------------|----------|
| `TextLoader` | `document/parser.py` | 文档解析的工程细节 |
| `TextSplitter` | `document/chunker.py` | 递归分割算法 + 重叠策略 |
| `OpenAIEmbeddings` | `embedding/model.py` | BGE 模型加载 + 批量嵌入 |
| `FAISS` | `index/vector_store.py` → IVF-PQ C++ | K-Means 聚类 + 乘积量化 + 倒排索引 |
| `RetrievalQA` | `retrieval/retriever.py` | 混合检索（RRF）+ Cross-Encoder 重排 |
| `AgentExecutor` | `agent/loop.py` | ReAct 循环 + 工具调度 |

## 路线图

- [x] **Phase 0** — 项目脚手架 + CLI 骨架
- [x] **Phase 1** — MVP 基础 RAG 管道（文档→嵌入→检索→生成）
- [ ] **Phase 2** — C++ IVF-PQ 索引 + BM25 稀疏检索 + 混合检索 + 重排序
- [ ] **Phase 3** — Agent 智能体（ReAct 循环 + 工具调用 + Prefix Caching）
- [ ] **Phase 4** — 量化优化（嵌入 INT8 + 权重 Q4 + KV Cache INT8）
- [ ] **Phase 5** — 生产化（FastAPI + Docker + 文档）

## 测试

```bash
pytest tests/ -v

# Phase 1: 11/11 passed
```

## 许可证

MIT © 2026 [ssG12333](https://github.com/ssG12333)
