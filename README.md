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
CLI (API planned) --> Agent ReAct Loop --> llama.cpp Generator
                     |                       |
                     |                       +-- KV Cache Monitor
                     |                       +-- Logical Prefix Stats
                     |
                     +-- Hybrid Retriever --> Reranker
                     |       |
                     |       +-- IVF-PQ Dense Index (C++)
                     |       +-- BM25 Sparse Index (C++)
                     |
                     +-- BGE Embedding (ONNX INT8 planned)
                     +-- Document Parser + Chunker
```

### Quick Start

```bash
git clone https://github.com/ssG12333/agentRAG.git
cd agentRAG
pip install -e ".[embedding]"

agentrag index --path ./your-docs/
agentrag ask "What is the complexity of self-attention?" --show-sources

# Residual IVF-PQ + BM25/RRF hybrid retrieval
# Windows: install pybind11/CMake and build the optional C++ extension first
python -m pip install pybind11 cmake ninja
scripts\build_cpp.bat
# For a non-standard MSVC install, set AGENTRAG_VCVARS to vcvars64.bat
agentrag index --path ./your-docs/ --backend ivfpq --save ./data/kb.ivfpq
agentrag ask "Exact model X100" --backend ivfpq --index-path ./data/kb.ivfpq --hybrid

# Cross-Encoder reranking is opt-in and may download the specified model
agentrag ask "your question" --hybrid --reranker-model BAAI/bge-reranker-base
```

### Systematic Learning Course

The repository includes a Chinese, code-driven RAG course with 16 chapters, offline labs, exercises, and explicit boundaries between implemented and planned features. Start at [docs/learning/README.md](docs/learning/README.md); the quick path does not download GGUF or Cross-Encoder models.

### Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Embedding | BGE-small-zh / BGE-base-zh | Chinese-friendly, ONNX exportable |
| Generation | llama.cpp-compatible GGUF (Qwen2.5-3B target) | Local generation; real GGUF validation pending |
| Inference | llama.cpp Python binding | KV monitoring exists; real prefix KV reuse is not integrated |
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
- [x] **Phase 2** — Residual IVF-PQ + BM25/RRF + optional reranker
- [ ] **Phase 3** — Mock end-to-end Agent and logical Prefix Cache complete; real GGUF/KV reuse pending
- [ ] **Phase 4** — Quantization (INT8 embed, Q4 weight, KV Cache INT8)
- [ ] **Phase 5** — Production (FastAPI, Docker, docs)

### Tests

```bash
pytest tests/ -v   # 49/49 passed
```

P1 synthetic retrieval snapshot (5,000 random vectors, not a real RAG quality baseline):

| Backend | Avg query | Recall@10 | Index data |
|---------|-----------|-----------|------------|
| NumPy exact | 0.0725 ms | 1.00 | 1,280,000 B |
| IVF-PQ | 0.0518 ms | 0.23 | 80,480 B |

The low IVF-PQ recall means it remains opt-in until P2 evaluation and tuning. [Raw CSV](docs/benchmarks/retrieval_p1.csv).

### License

MIT © 2026 [ssG12333](https://github.com/ssG12333)
