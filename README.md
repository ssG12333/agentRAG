<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/C++-17-00599C?style=flat&logo=c%2B%2B&logoColor=white" alt="C++" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="Status" />
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs" />
</p>

<h1 align="center">agentRAG</h1>

<p align="center">
  <b>Built from Scratch · Local-First · Quantization-Native · Full-Stack Self-Built</b><br/>
  <b>从零搭建 · 本地轻量 · 量化优先 · 全链路自研</b>
</p>

<p align="center">
  <i>A RAG agent that doesn't wrap LangChain. Every component is hand-written.<br/>
  每个组件手写实现，理解 RAG 的每一层。</i>
</p>

---

<p align="center"><a href="#english">English</a> | <a href="#chinese">中文</a></p>

---

<h2 id="english">English</h2>

### Philosophy

Most RAG tutorials teach you to call a one-liner from LangChain. It works, but you never understand what's inside:

- How does IVF-PQ actually accelerate vector search?
- What makes BM25 tick for non-English languages?
- How does KV Cache behave under long-context RAG?
- What exactly do you lose when quantizing an embedding model to INT8?

**agentRAG answers every one of these.**

### Architecture

```
CLI / API  -->  Agent React Loop  -->  llama.cpp Generator
                     |                       |
                     |                       +-- KV Cache Monitor
                     |                       +-- Prefix Caching
                     |                       +-- INT8 KV Cache Quant (planned)
                     |
                     +-- Hybrid Retriever --> Reranker
                     |       |
                     |       +-- IVF-PQ Dense Index (C++)
                     |       +-- BM25 Sparse Index (C++)
                     |
                     +-- ONNX INT8 Embedding (BGE)
                     +-- Document Parser + Chunker
```

### Quick Start

```bash
git clone https://github.com/ssG12333/agentRAG.git
cd agentRAG
pip install -e .

# Index your documents
agentrag index --path ./your-docs/

# Ask a question
agentrag ask "What is the complexity of self-attention?" --show-sources

# Interactive chat (in development)
agentrag chat
```

### Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Embedding | BGE-small-zh / BGE-base-zh | Chinese-friendly, lightweight, ONNX exportable |
| Generation | Qwen2.5-3B-Instruct (GGUF Q4) | Fits 6GB VRAM, strong enough for Agent reasoning |
| Inference | llama.cpp (`llama-cpp-python`) | Pure C++ kernel, KV Cache API |
| Dense Index | Self-built IVF-PQ (C++ + pybind11) | Understand every line, no FAISS black box |
| Sparse Index | Self-built BM25 (C++) | Inverted index + jieba tokenizer |
| Reranker | BGE-reranker-base (Cross-Encoder) | Phase 2 onward |

### Why Not LangChain?

| LangChain Component | Our Replacement | What You Learn |
|--------------------|----------------|----------------|
| `TextLoader` | `document/parser.py` | File I/O, encoding, metadata extraction |
| `TextSplitter` | `document/chunker.py` | Recursive split with overlap strategy |
| `OpenAIEmbeddings` | `embedding/model.py` | BGE model loading, batch inference |
| `FAISS` | `index/` → C++ IVF-PQ | K-Means, Inverted File, Product Quantization |
| `RetrievalQA` | `retrieval/` | Hybrid retrieval (RRF), Cross-Encoder rerank |
| `AgentExecutor` | `agent/loop.py` (planned) | ReAct loop, tool dispatch |

### Roadmap

- [x] **Phase 0** — Project scaffold + CLI skeleton
- [x] **Phase 1** — MVP RAG pipeline (doc -> embed -> retrieve -> generate)
- [x] **Phase 2** — Hybrid retrieval: C++ IVF-PQ + BM25 + RRF + reranker (Python tests: 16/16)
- [ ] **Phase 3** — Agent (ReAct loop, tool calling, Prefix Caching)
- [ ] **Phase 4** — Quantization (embedding INT8, weight Q4, KV Cache INT8)
- [ ] **Phase 5** — Production (FastAPI, Docker, docs)

### Tests

```bash
pytest tests/ -v
# Phase 1+2: 16/16 passed
```

---

<h2 id="chinese">中文</h2>

### 理念

市面上的 RAG 教程大多教你调用 LangChain 的一行 API。这能跑通，但你不会理解：

- 向量检索为什么快？IVF 倒排列表和 PQ 码本到底怎么工作的？
- BM25 的 TF-IDF 和词饱和度参数对中文检索有什么影响？
- KV Cache 在长上下文 RAG 中如何管理？Prefix Caching 能省多少时间？
- 本地模型如何量化？INT8 嵌入和 Q4 权重的精度损失到底多大？

**agentRAG 的目标是把这些全拆开给你看。**

### 架构

```
CLI / API  -->  Agent React 循环  -->  llama.cpp 生成器
                     |                       |
                     |                       +-- KV Cache 监控
                     |                       +-- Prefix Caching
                     |                       +-- INT8 KV Cache 量化（计划中）
                     |
                     +-- 混合检索器 --> 重排序器
                     |       |
                     |       +-- IVF-PQ 稠密索引 (C++)
                     |       +-- BM25 稀疏索引 (C++)
                     |
                     +-- ONNX INT8 嵌入 (BGE)
                     +-- 文档解析 + 分块
```

### 快速开始

```bash
git clone https://github.com/ssG12333/agentRAG.git
cd agentRAG
pip install -e .

# 索引文档
agentrag index --path ./your-docs/

# 提问
agentrag ask "Transformer 的计算复杂度是多少？" --show-sources

# 交互对话（开发中）
agentrag chat
```

### 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| 嵌入模型 | BGE-small-zh / BGE-base-zh | 中文友好，轻量，支持 ONNX 导出 |
| 生成模型 | Qwen2.5-3B-Instruct (GGUF Q4) | 6GB 显存全 GPU，Agent 推理够用 |
| 推理引擎 | llama.cpp (`llama-cpp-python`) | 纯 C++ 内核，KV Cache API 完备 |
| 稠密索引 | 自研 IVF-PQ (C++ + pybind11) | 不用 FAISS，从零理解索引原理 |
| 稀疏索引 | 自研 BM25 (C++) | 倒排索引 + jieba 分词 |
| 重排序 | BGE-reranker-base (Cross-Encoder) | Phase 2 起使用 |

### 为什么不用 LangChain？

| LangChain 组件 | agentRAG 替代 | 你学到什么 |
|---------------|-------------|----------|
| `TextLoader` | `document/parser.py` | 文档解析的工程细节 |
| `TextSplitter` | `document/chunker.py` | 递归分割算法 + 重叠策略 |
| `OpenAIEmbeddings` | `embedding/model.py` | BGE 模型加载 + 批量嵌入 |
| `FAISS` | `index/` → C++ IVF-PQ | K-Means 聚类 + 乘积量化 + 倒排索引 |
| `RetrievalQA` | `retrieval/` | 混合检索（RRF）+ Cross-Encoder 重排 |
| `AgentExecutor` | `agent/loop.py`（计划中） | ReAct 循环 + 工具调度 |

### 路线图

- [x] **Phase 0** — 项目脚手架 + CLI 骨架
- [x] **Phase 1** — MVP 基础 RAG 管道（文档→嵌入→检索→生成）
- [x] **Phase 2** — 混合检索：C++ IVF-PQ + BM25 + RRF + 重排序（Python 16/16 测试通过）
- [ ] **Phase 3** — Agent 智能体（ReAct 循环 + 工具调用 + Prefix Caching）
- [ ] **Phase 4** — 量化优化（嵌入 INT8 + 权重 Q4 + KV Cache INT8）
- [ ] **Phase 5** — 生产化（FastAPI + Docker + 文档）

### 测试

```bash
pytest tests/ -v
# Phase 1+2: 16/16 passed
```

---

### License

MIT © 2026 [ssG12333](https://github.com/ssG12333)
