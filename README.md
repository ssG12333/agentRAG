<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/C++-17-00599C?style=flat&logo=c%2B%2B&logoColor=white" alt="C++" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="Status" />
</p>

<h1 align="center">agentRAG</h1>

<p align="center">
  <b>Built from Scratch · Local-First · Quantization-Native</b>
</p>

<p align="center">
  <sub><a href="README-cn.md">中文</a></sub>
</p>

<p align="center">
  <i>A RAG agent that doesn't wrap LangChain. Every component is hand-written<br/>
  so you understand every layer of retrieval-augmented generation.</i>
</p>

---

### Philosophy

Most RAG tutorials teach you to call a one-liner from LangChain. It works, but you never understand what's inside. **agentRAG answers every one of these.**

### Architecture

```
CLI / API  -->  Agent React Loop  -->  llama.cpp Generator
                     |                       |
                     |                       +-- KV Cache Monitor
                     |                       +-- Prefix Caching
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

agentrag index --path ./your-docs/
agentrag ask "What is the complexity of self-attention?" --show-sources
```

### Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Embedding | BGE-small-zh / BGE-base-zh | Chinese-friendly, ONNX exportable |
| Generation | Qwen2.5-3B-Instruct (GGUF Q4) | Fits 6GB VRAM, Agent-capable |
| Inference | llama.cpp | Pure C++ kernel, full KV Cache API |
| Dense Index | Self-built IVF-PQ (C++) | No FAISS black box |
| Sparse Index | Self-built BM25 (C++) | Inverted index + jieba |
| Reranker | BGE-reranker-base (Cross-Encoder) | Phase 2+ |

### Why Not LangChain?

| LangChain | agentRAG | What You Learn |
|-----------|----------|----------------|
| `TextLoader` | `document/parser.py` | File I/O, encoding, metadata |
| `TextSplitter` | `document/chunker.py` | Recursive split, overlap strategy |
| `OpenAIEmbeddings` | `embedding/model.py` | Model loading, batch inference |
| `FAISS` | C++ residual IVF-PQ (`src/core/`) | K-Means, Inverted File, residual PQ |
| `RetrievalQA` | `retrieval/` | RRF hybrid, Cross-Encoder rerank |
| `AgentExecutor` | `agent/loop.py` | ReAct loop, tool dispatch |

### Roadmap

- [x] **Phase 0** — Scaffold + CLI
- [x] **Phase 1** — MVP RAG (parse → embed → retrieve → generate)
- [ ] **Phase 2** — Residual IVF-PQ/BM25 core complete; RRF/reranker main-path integration in progress
- [ ] **Phase 3** — Agent core complete; Prefix Cache and end-to-end integration in progress
- [ ] **Phase 4** — Quantization (INT8 embed, Q4 weight, KV Cache INT8)
- [ ] **Phase 5** — Production (FastAPI, Docker, docs)

### Tests

```bash
pytest tests/ -v   # 34/34 passed
```

### License

MIT © 2026 [ssG12333](https://github.com/ssG12333)
