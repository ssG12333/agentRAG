<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/C++-17-00599C?style=flat&logo=c%2B%2B&logoColor=white" alt="C++" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="Status" />
</p>

<h1 align="center">agentRAG</h1>

<p align="center">
  <b>从零搭建 · 本地轻量 · 量化优先 · 全链路自研</b>
</p>

<p align="center">
  <sub><a href="README.md">English</a></sub>
</p>

<p align="center">
  <i>不依赖 LangChain 的 RAG Agent。每个组件手写实现，<br/>
  让你理解检索增强生成的每一层。</i>
</p>

---

### 理念

市面上的 RAG 教程大多教你调用 LangChain 的一行 API。能跑通，但你不理解。**agentRAG 把每一层拆开给你看。**

### 架构

```
CLI（API 规划中）--> Agent ReAct 循环 --> llama.cpp 生成器
                     |                       |
                     |                       +-- KV Cache 监控
                     |                       +-- 逻辑 Prefix 统计
                     |
                     +-- 混合检索器 --> 重排序器
                     |       |
                     |       +-- IVF-PQ 稠密索引 (C++)
                     |       +-- BM25 稀疏索引 (C++)
                     |
                     +-- BGE 嵌入（ONNX INT8 规划中）
                     +-- 文档解析 + 分块
```

### 快速开始

```bash
git clone https://github.com/ssG12333/agentRAG.git
cd agentRAG
pip install -e ".[embedding]"

agentrag index --path ./your-docs/
agentrag ask "Transformer 的计算复杂度是多少？" --show-sources

# 残差 IVF-PQ + BM25/RRF 混合检索
# Windows：先安装 pybind11/CMake 并构建可选 C++ 扩展
python -m pip install pybind11 cmake ninja
scripts\build_cpp.bat
# MSVC 位于非标准目录时，将 AGENTRAG_VCVARS 指向 vcvars64.bat
agentrag index --path ./your-docs/ --backend ivfpq --save ./data/kb.ivfpq
agentrag ask "精确型号 X100" --backend ivfpq --index-path ./data/kb.ivfpq --hybrid

# Cross-Encoder 重排序默认关闭；显式启用时可能下载指定模型
agentrag ask "你的问题" --hybrid --reranker-model BAAI/bge-reranker-base
```

### 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| 嵌入模型 | BGE-small-zh / BGE-base-zh | 中文友好，支持 ONNX 导出 |
| 生成模型 | llama.cpp 兼容 GGUF（目标 Qwen2.5-3B） | 本地生成；真实 GGUF 验证待完成 |
| 推理引擎 | llama.cpp Python 绑定 | 已有 KV 监控；尚未接入真实 Prefix KV 复用 |
| 稠密索引 | 自研 IVF-PQ (C++) | 不用 FAISS，理解每行代码 |
| 稀疏索引 | 自研 BM25 (C++) | 倒排索引 + jieba 分词 |
| 重排序 | BGE-reranker-base (Cross-Encoder) | Phase 2 起 |

### 为什么不用 LangChain？

| LangChain | agentRAG | 你学到什么 |
|-----------|----------|----------|
| `TextLoader` | `document/parser.py` | 文件 I/O、编码、元数据 |
| `TextSplitter` | `document/chunker.py` | 递归分割 + 重叠策略 |
| `OpenAIEmbeddings` | `embedding/model.py` | 模型加载 + 批量嵌入 |
| `FAISS` | C++ 残差 IVF-PQ (`src/core/`) | K-Means + 倒排索引 + 残差乘积量化 |
| `RetrievalQA` | `retrieval/` | RRF 混合检索 + Cross-Encoder 重排 |
| `AgentExecutor` | `agent/loop.py` | ReAct 循环 + 工具调度 |

### 路线图

- [x] **Phase 0** — 项目脚手架 + CLI
- [x] **Phase 1** — MVP RAG 管道（解析→嵌入→检索→生成）
- [x] **Phase 2** — 残差 IVF-PQ + BM25/RRF + 可选重排序
- [ ] **Phase 3** — Mock Agent 端到端与逻辑 Prefix Cache 完成；真实 GGUF/KV 复用待验证
- [ ] **Phase 4** — 量化（嵌入 INT8、权重 Q4、KV Cache INT8）
- [ ] **Phase 5** — 生产化（FastAPI、Docker、文档）

### 测试

```bash
pytest tests/ -v   # 46/46 passed
```

P1 合成检索快照（5,000 个随机向量，不代表真实 RAG 质量基线）：

| 后端 | 平均查询 | Recall@10 | 索引数据 |
|------|----------|-----------|----------|
| NumPy 精确检索 | 0.0725 ms | 1.00 | 1,280,000 B |
| IVF-PQ | 0.0518 ms | 0.23 | 80,480 B |

IVF-PQ 召回率仍然较低，因此在 P2 评估和调参完成前保持显式启用。[原始 CSV](docs/benchmarks/retrieval_p1.csv)。

### 许可证

MIT © 2026 [ssG12333](https://github.com/ssG12333)
