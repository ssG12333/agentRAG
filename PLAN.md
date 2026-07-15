# agentRAG — 从零搭建本地轻量 RAG Agent 实施计划

> [!WARNING]
> 本文件是 2026-07-14 的历史实施设想，不代表当前已实现能力。请以
> [`TODO.md`](TODO.md)、[`LOG.md`](LOG.md)、当前源码和测试为准。尤其是 ONNX INT8、
> 真实 Prefix KV Cache 复用、FastAPI 和生产化能力仍属于后续研究或规划。

> 最后更新：2026-07-14
> 状态：规划中

---

## 决策回顾

| 维度 | 选择 |
|------|------|
| 项目方向 | 本地轻量 RAG Agent + 从零搭建理解全链路 |
| 项目定位 | 学习简历项目 + 长期迭代开源 |
| 开发语言 | Python（应用编排）+ C++（热路径加速） |
| 核心约束 | 本地运行、无云API依赖、量化优先、模块可替换 |

## 硬件与环境

| 组件 | 配置 |
|------|------|
| CPU | Intel Core i7-10870H (8核16线程, 2.20GHz) |
| GPU | NVIDIA GeForce RTX 2060 (6GB VRAM, CUDA 11.8) |
| 内存 | 32GB DDR4 |
| Python | 3.10.17 (Conda: `LR1`) |
| PyTorch | 2.6.0+cu118 |
| ONNX Runtime | 1.23.2 |

## 模型选型（GPU 部署）

### 生成模型（GGUF, llama.cpp + CUDA）

| 模型 | 大小 (Q4_K_M) | VRAM (全GPU) | Tokens/s | Agent 能力 |
|------|:--:|:--:|:--:|:--:|
| Qwen2.5-1.5B-Instruct | 1.0 GB | ~1.3 GB | 40-60 | 弱 |
| **Qwen2.5-3B-Instruct** ⭐ | **2.0 GB** | **~2.8 GB** | **30-40** | **可用** |
| Qwen2.5-7B-Instruct 部分 | 4.5 GB | ~5.8 GB (20/28层) | 15-25 | 良好 |

> **选择 Qwen2.5-3B Q4_K_M**：全量加载到 GPU，6GB VRAM 绰绰有余，留 3GB 给嵌入模型和 KV Cache。

### RTX 2060 显存分配（主力方案）

```
嵌入 BGE-base-zh INT8 (ONNX CUDA)    ~150 MB
生成 Qwen2.5-3B Q4_K_M (28/28 GPU)  ~2000 MB
KV Cache (8K ctx, FP16)              ~800 MB
CUDA 运行时 + 缓冲                    ~400 MB
─────────────────────────────────────────────
已用                                ~3350 MB
剩余                                ~2800 MB  (可扩上下文或批处理)
```

### 嵌入模型

| 模型 | ONNX INT8 大小 | VRAM | 使用阶段 |
|------|:--:|:--:|------|
| BGE-small-zh-v1.5 | 25 MB | ~80 MB | 快速调试 |
| **BGE-base-zh-v1.5** ⭐ | 100 MB | ~150 MB | 标准使用 |

### 双配置文件

```yaml
# configs/light.yaml   — 快速调试 / 低配用户
generation.model: Qwen2.5-1.5B-Instruct Q4_K_M
embedding.model: BAAI/bge-small-zh-v1.5
generation.n_gpu_layers: -1  # 全 GPU

# configs/standard.yaml — 实际使用
generation.model: Qwen2.5-3B-Instruct Q4_K_M
embedding.model: BAAI/bge-base-zh-v1.5
generation.n_gpu_layers: -1  # 全 GPU
```

---

## 总体架构

```
CLI / API  ──→  Agent 循环  ──→  生成(llama.cpp)
                     │                │
                     │                ├── KV Cache 管理
                     │                │   ├── Prefix Caching（复用系统提示）
                     │                │   ├── 驱逐策略（长上下文窗口）
                     │                │   └── INT8 KV Cache 量化
                     │                │
                     ├── 检索(混合) ──→ 重排序
                     │       │
                     │       ├── 稠密索引(IVF-PQ, C++)
                     │       └── 稀疏索引(BM25, C++)
                     │
                     ├── 嵌入(ONNX INT8)
                     └── 文档处理(解析/分块)
```

### KV Cache 在本项目中的角色

KV Cache 不只是生成引擎的内部细节，而是贯穿多层的关键优化：

| 层次 | KV Cache 应用 | 收益 |
|------|-------------|------|
| **生成层** | 配置 `n_ctx`、`batch_size`，监控 KV Cache 内存 | 可控的长文本推理 |
| **检索层** | 多 chunk 拼接为长上下文时的 Cache 内存预算 | 避免 OOM |
| **对话层** | Prefix Caching：系统提示+检索结果不变部分复用 | 多轮对话首 token 延迟降低 50%+ |
| **量化层** | KV Cache INT8/FP8 量化 | Cache 内存减半，支持更长上下文 |
| **Agent 层** | 工具调用间复用 Prefix Cache | 多步推理加速 |

---

# Phase 0: 项目脚手架

**目标**：空项目跑通，目录就绪，构建系统可工作

## 步骤 0.1：创建目录结构

```
agentRAG项目/
├── src/
│   ├── __init__.py
│   ├── document/
│   ├── embedding/
│   ├── index/
│   ├── retrieval/
│   ├── generation/
│   ├── agent/
│   ├── cli/
│   └── core/          ← C++ 代码
│       ├── include/
│       ├── src/
│       └── pybind/
├── tests/
├── examples/
├── configs/
├── scripts/
└── docs/
```

## 步骤 0.2：创建 `pyproject.toml`

- 包名 `agentrag`
- 依赖：`numpy`, `pyyaml`, `rich`, `click`
- 开发依赖：`pytest`, `pytest-cov`
- 入口点：`agentrag` → `src.cli.main:main`

## 步骤 0.3：创建 `CMakeLists.txt`

- 根 `CMakeLists.txt`：项目名、C++17 标准
- `src/core/CMakeLists.txt`：pybind11 模块编译
- 使用 FetchContent 获取 pybind11

## 步骤 0.4：创建 CLI 入口骨架 `src/cli/main.py`

- `click` 命令组
- `agentrag --help` 输出版本和子命令列表
- 预留子命令：`ask`, `chat`, `index`, `serve`

## 步骤 0.5：创建 `src/core/pybind/module.cpp`

- 最小 pybind11 模块：`agentrag_core`
- 一个 `add(int, int) -> int` 测试函数
- 验证 Python 能 `import agentrag_core`

## 步骤 0.6：创建 `README.md`

- 项目名称和一句话简介
- 技术栈标签
- "🚧 开发中" 状态标识

## 步骤 0.7：创建 `configs/default.yaml`

- 空配置骨架，注释说明各字段含义

**Phase 0 验证**：
```bash
pip install -e .
agentrag --help
python -c "import agentrag_core; print(agentrag_core.add(1, 2))"
```

---

# Phase 1: MVP 基础 RAG 管道（纯 Python）

**目标**：端到端跑通——给文档建索引 → 提问 → 检索 → 生成回答

## 步骤 1.1：文档数据模型 `src/document/__init__.py`

- `Document` dataclass：
  - `id: str`（文档唯一ID）
  - `content: str`（原始文本）
  - `metadata: dict`（来源文件、页码、标题等）
- `Chunk` dataclass：
  - `id: str`
  - `document_id: str`
  - `content: str`
  - `metadata: dict`
  - `embedding: Optional[np.ndarray]`

## 步骤 1.2：Markdown 解析器 `src/document/parser.py`

- `BaseParser` 抽象类：`parse(file_path) -> Document`
- `MarkdownParser`：读取 .md 文件，保留标题层级
- `TextParser`：读取 .txt 文件
- `get_parser(file_path) -> BaseParser`：工厂函数，按扩展名选择解析器
- 先不做 PDF（依赖重，Phase 5 再加）

## 步骤 1.3：文本分块器 `src/document/chunker.py`

- `BaseChunker` 抽象类：`chunk(document) -> List[Chunk]`
- `RecursiveChunker`：
  - 分隔符优先级：`["\n\n", "\n", "。", ".", " ", ""]`
  - `chunk_size` 默认 512，`chunk_overlap` 默认 64
  - 对每个 chunk 保留原始文档的 metadata
- `FixedWindowChunker`：固定窗口滑动（最简单，用于验证）
- 两个分块器都实现，CLI 可通过参数切换

## 步骤 1.4：嵌入模型封装 `src/embedding/model.py`

- `BaseEmbedding` 抽象类：`embed(texts: List[str]) -> np.ndarray`
- `SentenceTransformerEmbedding`：
  - 加载 `BAAI/bge-small-zh-v1.5`（中文）或 `all-MiniLM-L6-v2`（英文）
  - `embed()` 返回 `(n, dim)` 的 numpy 数组
  - 单条文本也接受，内部统一转为列表
- `CachedEmbedding`：带 LRU 缓存的装饰器（减少重复嵌入）

## 步骤 1.5：简单向量存储 `src/index/vector_store.py`

- `BaseVectorStore` 抽象类：
  - `add(embeddings, chunks)`：存储向量+块
  - `search(query_vector, top_k) -> List[Tuple[Chunk, float]]`
- `NumpyVectorStore`（Phase 1 实现）：
  - 内部用 `np.ndarray` 存向量
  - 余弦相似度检索（np.dot + 归一化）
  - 全量遍历（不做索引加速，Phase 2 替换）
- 不实现持久化（先跑通流程）

## 步骤 1.6：检索器 `src/retrieval/retriever.py`

- `Retriever` 类：
  - `__init__(embedding_model, vector_store)`
  - `retrieve(query, top_k) -> List[Tuple[Chunk, float]]`
  - 流程：embed query → vector_store.search → 返回 chunk+score

## 步骤 1.7：提示词模板 `src/generation/prompt.py`

- `PromptTemplate` dataclass：
  - `system: str`
  - `user: str`
- `RAGPromptBuilder`：
  - `build(query, chunks, chat_history) -> str`
  - 系统提示：`"你是一个知识助手，只根据提供的文档内容回答问题..."`
  - 用户提示模板：
    ```
    ## 参考文档
    {chunks}
    
    ## 对话历史
    {history}
    
    ## 用户问题
    {query}
    ```

## 步骤 1.8：生成引擎 `src/generation/engine.py`

- `BaseLLM` 抽象类：
  - `generate(prompt, **kwargs) -> str`
  - `generate_stream(prompt, **kwargs) -> Iterator[str]`
- `LlamaCppEngine`：
  - 封装 `llama-cpp-python`（编译时需开 `CMAKE_ARGS="-DGGML_CUDA=ON"`）
  - 配置：`n_ctx`, `n_threads`, `temperature`, `top_p`, `n_gpu_layers`
  - `n_gpu_layers = -1`：所有层加载到 GPU（3B 模型 28 层全 GPU 仅占 2GB VRAM）
  - KV Cache 相关配置：
    - `n_batch`：每批处理 token 数（影响 KV Cache 写入效率）
    - `n_ctx`：上下文窗口 → KV Cache 容量上限
  - 流式输出 `yield` token
- `MockLLM`（测试用）：
  - 返回固定回答，不依赖模型文件

## 步骤 1.8b：KV Cache 内存监控 `src/generation/kv_cache.py`

- `KVCacheStats` dataclass：
  - `total_slots: int`：总槽位数 = `n_ctx`
  - `used_slots: int`：当前已用槽位
  - `free_slots: int`：剩余槽位
  - `memory_bytes: int`：KV Cache 占用内存（估算）
  - `dtype: str`：存储精度（FP16 / INT8）
- `KVCacheMonitor`：
  - `stats() -> KVCacheStats`：查询当前状态
  - `warn_if_low(threshold_pct=0.9)`：接近上限时警告
  - `suggest_truncation(max_tokens) -> int`：建议截断点
- 集成到 `Retriever`：检索时根据 KV Cache 剩余容量动态调整 `top_k`

## 步骤 1.9：CLI `index` 子命令 `src/cli/main.py`

- `agentrag index --path ./docs/ --chunk-size 512`
- 流程：遍历目录 → 解析文档 → 分块 → 嵌入 → 存入向量库
- 用 `rich.progress` 显示进度条
- 完成后打印统计：文档数、块数、向量维度

## 步骤 1.10：CLI `ask` 子命令

- `agentrag ask "问题文本" --top-k 5`
- 流程：embed query → 检索 → 构建 prompt → 生成 → 打印回答
- 可选参数：`--temperature`, `--show-sources`（打印引用来源）

## 步骤 1.11：端到端示例 `examples/01_basic_rag.py`

- 下载一个中文模型（如 Qwen2.5-1.5B-Instruct GGUF）
- 创建测试文档（关于 Transformer 的一段说明）
- 索引 → 提问 → 打印回答 + 来源

## 步骤 1.12：Phase 1 测试 `tests/`

| 测试文件 | 测试内容 |
|---------|---------|
| `tests/test_parser.py` | MarkdownParser 解析 .md → Document |
| `tests/test_chunker.py` | RecursiveChunker 分块数、重叠正确 |
| `tests/test_embedding.py` | 嵌入输出 shape 正确、缓存命中 |
| `tests/test_vector_store.py` | add+search 返回正确 chunk |
| `tests/test_retriever.py` | 检索端到端返回正确数量 |
| `tests/test_prompt.py` | Prompt 格式包含所有必要字段 |
| `tests/test_engine.py` | MockLLM 返回预期回答 |

**Phase 1 验证**：
```bash
pytest tests/ -v
python examples/01_basic_rag.py
```

---

# Phase 2: 混合检索 + C++ 加速

**目标**：用 C++ 实现 IVF-PQ 和 BM25，替换 numpy 版本；加入重排序

## 步骤 2.1：C++ 向量数据结构 `src/core/include/vector_types.h`

- `using float32 = float;`
- `struct Vector { float32* data; size_t dim; };`
- `struct VectorBatch { Vector* vectors; size_t n; size_t dim; };`
- `struct SearchResult { int id; float score; };`

## 步骤 2.2：C++ K-Means 聚类 `src/core/src/kmeans.cpp`

- `kmeans(vectors, k, max_iters) -> (centroids, assignments)`
- 初始化：随机选 k 个点作为初始中心
- 迭代：分配 → 更新中心 → 检查收敛
- 纯 C++，不依赖任何库

## 步骤 2.3：C++ IVF 索引 `src/core/src/vector_index.cpp`

- 数据结构：
  - `centroids`: k 个聚类中心的向量
  - `inverted_lists`: k 个列表，每个列表存 `(vector_id, residual)`
  - `vector_ids`: 全局 ID 映射
- `build(vectors, k, n_probe)`：
  - 对所有向量做 K-Means
  - 计算每个向量到最近中心的残差
  - 存入对应的倒排列表
- `search(query, top_k)`：
  - 计算 query 到 k 个中心的距离
  - 选最近的 `n_probe` 个倒排列表
  - 在这些列表中暴力搜索 top_k

## 步骤 2.4：C++ PQ 编码 `src/core/src/product_quantizer.cpp`

- `ProductQuantizer` 类：
  - `train(vectors, n_subvectors, n_bits)`：
    - 将向量均分为 `n_subvectors` 段
    - 每段独立 K-Means → 码本
  - `encode(vector) -> codes`：
    - 每段找最近码字 → 返回码本索引数组
  - `decode(codes) -> vector`：
    - 查码本 → 拼接还原
  - `distances(query, codes)`：
    - 预计算查询向量每段到各码字的距离表
    - 查表累加得近似距离（快于解码后算距离）

## 步骤 2.5：IVF-PQ 组合 `src/core/src/ivf_pq_index.cpp`

- 将步骤 2.3 的倒排列表中存残差的 PQ 编码（而非原始残差）
- `build(vectors)`：聚类 → 算残差 → PQ 编码残差
- `search(query, top_k)`：粗量化 → 表距离搜索 → top_k
- 内存：原始 `n * d * 4` → PQ `n * n_subvectors * n_bits/8`

## 步骤 2.6：中文分词器 (C++) `src/core/src/tokenizer.cpp`

- `class Tokenizer`：
  - `tokenize(text) -> vector<string>`
  - 使用前缀词典 + 动态规划（jieba 算法简化版）
  - 内置常见中文词典（约 2-3 万词）
  - 支持用户词典
- 接口足够简单，后续可替换为 jieba 的 C 接口

## 步骤 2.7：BM25 倒排索引 (C++) `src/core/src/sparse_index.cpp`

- `class BM25Index`：
  - `add_document(doc_id, tokens)`：
    - 统计词频 tf
    - 更新文档长度
  - `search(query_tokens, top_k) -> vector<SearchResult>`：
    - BM25 公式：`IDF * (tf * (k1+1)) / (tf + k1 * (1 - b + b * len/avg_len))`
    - 参数 k1=1.5, b=0.75
  - `remove_document(doc_id)`
  - 内部数据结构：
    - `posting_lists`: `map<string, vector<pair<int, int>>>`（词 → doc_id+tf 列表）
    - `doc_lengths`: `vector<int>`
    - `avg_doc_length`: `float`

## 步骤 2.8：pybind11 绑定 `src/core/pybind/module.cpp`

- 暴露以下类到 Python：
  - `IVFPQIndex`：`build(numpy_array)`, `search(numpy_array, top_k) -> List[Tuple[int, float]]`
  - `BM25Index`：`add_doc(id, tokens)`, `search(tokens, top_k) -> List[Tuple[int, float]]`
  - `Tokenizer`：`tokenize(text) -> List[str]`
- numpy ↔ C++ 数组转换（零拷贝）
- 编译脚本：`python setup.py build_ext` 或 `pip install -e .`

## 步骤 2.9：Python 向量存储层 `src/index/vector_store.py`

- 将 Phase 1 的 `NumpyVectorStore` 改为内部使用 `IVFPQIndex`
- `CppVectorStore`：
  - `add(embeddings, chunks)`：调用 C++ `build()`
  - `search(query_vector, top_k)`：调用 C++ `search()`
  - 保持接口不变，Phase 1 代码无需改动

## 步骤 2.10：Python 稀疏存储层 `src/index/sparse_store.py`

- `SparseVectorStore`：
  - 内部持有 `BM25Index` 实例
  - `add(chunks)`：对每个 chunk 分词 → 调用 C++ `add_doc()`
  - `search(query, top_k)`：分词 → 调用 C++ `search()`
  - `remove(doc_id)`

## 步骤 2.11：混合检索融合 `src/index/hybrid.py`

- `HybridRetriever`：
  - `__init__(dense_store, sparse_store, alpha=0.5)`
  - `search(query, top_k)`：
    - 并行调用 dense + sparse
    - RRF 融合：`score = alpha * dense_score + (1-alpha) * sparse_score`
    - 去重（同 chunk 取最高分）
    - 重排返回 top_k

## 步骤 2.12：Cross-encoder 重排序 `src/retrieval/reranker.py`

- `BaseReranker` 抽象类：`rerank(query, chunks) -> List[Tuple[Chunk, float]]`
- `CrossEncoderReranker`：
  - 加载 `BAAI/bge-reranker-base`（~278M 参数）
  - 输入：`(query, chunk_content)` 对
  - 输出：相关性分数
  - 对 top-N 候选做精排（N 默认 50 → top_k 5）
- `NoOpReranker`：直接透传（Phase 1 使用）

## 步骤 2.13：检索管道整合 `src/retrieval/retriever.py`

- 更新 `Retriever`：集成 HybridRetriever + Reranker
- 流程：embed → hybrid search → rerank → return
- 保持对 Phase 1 的 CLI 兼容

## 步骤 2.14：Phase 2 测试

| 测试文件 | 测试内容 |
|---------|---------|
| `tests/test_cpp_tokenizer.py` | 分词结果正确（中文/英文/混合） |
| `tests/test_cpp_ivf.py` | 索引构建、搜索召回率 |
| `tests/test_cpp_bm25.py` | BM25 分数计算正确 |
| `tests/test_hybrid.py` | 混合检索 > 单一检索 |
| `tests/test_reranker.py` | 重排序提升 MRR |
| `tests/benchmark/` | Phase 1 numpy vs Phase 2 C++ 速度对比 |

**Phase 2 验证**：
```bash
# C++ 单元测试
cd build && cmake .. && make && ctest

# Python 集成测试
pytest tests/ -v

# 性能对比脚本
python scripts/benchmark_retrieval.py
```

---

# Phase 3: Agent 能力

**目标**：从"被动问答"升级为"主动规划+调用工具"的智能体

## 步骤 3.1：工具抽象 `src/agent/tools.py`

- `Tool` dataclass：
  - `name: str`
  - `description: str`（给 LLM 看的）
  - `parameters: dict`（JSON Schema 格式）
  - `func: Callable[..., str]`
- `ToolRegistry`：
  - `register(tool)` / `unregister(name)`
  - `get_tool(name) -> Tool`
  - `list_tools() -> List[Tool]`
  - `get_tools_prompt() -> str`（生成给 LLM 的工具描述）
- 工具调用格式：
  ```
  <tool_call>
  {"name": "search_knowledge_base", "arguments": {"query": "xxx", "top_k": 5}}
  </tool_call>
  ```

## 步骤 3.2：内置工具

- `search_knowledge_base(query, top_k) -> str`：
  - 调用 Retriever → 格式化 chunk 返回
- `read_file(path) -> str`：
  - 读取指定文件内容
  - 路径安全检查（防止目录遍历）
- `list_documents() -> str`：
  - 列出已索引的文档列表
- `calculator(expression) -> str`：
  - 安全的数学表达式求值（仅数字和运算符）

## 步骤 3.3：对话记忆 `src/agent/memory.py`

- `ConversationMemory`：
  - `messages: List[Dict[str, str]]`（`role` + `content`）
  - `add_user(message)`, `add_assistant(message)`, `add_system(message)`
  - `get_history(n_last=10) -> str`
  - `clear()`
- `WorkingMemory`（短期）：
  - 当前任务的目标、中间结果
  - `set(key, value)`, `get(key) -> value`

## 步骤 3.4：Prefix Caching 管理器 `src/generation/prefix_cache.py`

- `PrefixCache`：
  - 核心思想：系统提示 + 检索上下文在多轮中重复出现时，复用其 KV Cache
  - `cache_key(prompt_prefix) -> str`：对前缀内容 hash
  - `has(key) -> bool`：是否已缓存
  - `store(key, kv_cache_slice)`：存储前缀的 KV Cache 切片
  - `load(key) -> kv_cache_slice`：恢复
  - `evict_lru()`：LRU 驱逐
- `PrefixAwareEngine`：
  - 包装 `LlamaCppEngine`
  - `generate_with_cache(system_prompt, context, query)`：
    1. 将 system_prompt + context 作为前缀
    2. 若命中 Prefix Cache → 跳过前缀的 prefill，直接从 query 开始 decode
    3. 若未命中 → 完整 prefill，存入 Prefix Cache
  - 收益：多轮对话中首 token 延迟降低 50-70%（当检索结果不变时）
- 为什么 RAG 场景特别适合 Prefix Caching：
  - 系统提示固定
  - 同一知识库的多轮对话中检索结果高度重叠
  - 同一批 chunks 在多次追问中重复使用

## 步骤 3.5：ReAct 循环 `src/agent/loop.py`

- `ReActAgent`：
  - `run(user_query, max_steps=5) -> str`
  - 每步：
    1. 构建 prompt（系统提示 + 工具列表 + 历史 + 当前查询 + 观察）
    2. 调用 LLM 生成（通过 `PrefixAwareEngine`，复用系统提示前缀）
    3. 解析输出：Thought / Action / Observation
    4. 如果 Action → 执行工具 → 得到 Observation → 回到步骤 1
    5. 如果 Final Answer → 返回
  - `max_steps` 防止无限循环
  - Agent 循环内 Prefix Caching 收益：
    - 多步推理中系统提示 + 工具描述始终不变 → 前缀始终命中
    - 每步只需 prefill 新增的 Observation 部分
- prompt 格式（ReAct 经典格式）：
  ```
  Thought: 我需要先搜索知识库...
  Action: search_knowledge_base
  Action Input: {"query": "Transformer 自注意力", "top_k": 5}
  
  Observation: [文档1] Transformer中的自注意力...
  
  Thought: 我已经找到相关文档...
  Final Answer: Transformer中的自注意力计算复杂度是O(n²·d)...
  ```

## 步骤 3.6：查询改写 `src/retrieval/rewriter.py`

- `QueryRewriter`：
  - `rewrite(query, chat_history) -> str`
  - 将指代消解：`"它有什么优点？"` → `"Transformer 有什么优点？"`
  - 使用 LLM 做改写（成本低，本地模型可承受）
  - 带缓存避免重复改写

## 步骤 3.7：Agent CLI

- 新增 `agentrag chat` 命令
- 交互式对话：`>` 提示符
- 命令：
  - `/clear`：清除对话历史和 Prefix Cache
  - `/sources`：显示上一轮引用的来源
  - `/docs`：列出已索引文档
  - `/cache`：显示 Prefix Cache 命中率统计
  - `/exit`：退出
- 每轮显示：`🤔 思考中...` → `🔧 调用工具: search_knowledge_base` → `📝 回答`
- 流式输出回答

## 步骤 3.8：Phase 3 测试

| 测试文件 | 测试内容 |
|---------|---------|
| `tests/test_tools.py` | 工具注册/调用/格式正确 |
| `tests/test_memory.py` | 记忆增删查 |
| `tests/test_prefix_cache.py` | Prefix Cache 存储/命中/驱逐 |
| `tests/test_agent_loop.py` | ReAct 解析、步数限制、Final Answer 检测 |
| `tests/test_rewriter.py` | 指代消解正确 |
| `tests/test_agent_e2e.py` | Agent 多轮对话端到端 |

## 步骤 3.9：Agent 示例 `examples/03_agent_demo.py`

- 加载知识库（AI Infra 相关文档）
- 多轮对话演示：提问 → Agent 自主检索 → 追问 → 指代消解 → 回答
- 显示 Prefix Cache 命中率

**Phase 3 验证**：
```bash
pytest tests/ -v
python examples/03_agent_demo.py
```

---

# Phase 4: 量化优化

**目标**：发挥量化专长，用 INT8/INT4 全面压缩模型

## 步骤 4.1：嵌入模型导出 ONNX `src/embedding/quantize.py`

- `export_to_onnx(model_name, output_path)`：
  - 加载 sentence-transformers 模型
  - 导出 transformer 部分为 ONNX
  - 保留 tokenizer 配置
- `quantize_onnx(input_path, output_path, precision="int8")`：
  - 使用 ONNX Runtime 的静态量化
  - 需要校准数据集（从分块文本中采样 100 条）

## 步骤 4.2：ONNX Runtime 推理 `src/embedding/onnx_runner.py`

- `ONNXEmbedding`：
  - 加载量化后的 ONNX 模型
  - 使用 ONNX Runtime 推理
  - 与 `SentenceTransformerEmbedding` 共享 `BaseEmbedding` 接口
  - 替换后上层代码无感知

## 步骤 4.3：量化前后对比基准 `scripts/benchmark_embedding.py`

- 指标：
  - 模型大小（MB）
  - 推理时间（ms/条，batch=1, 8, 32）
  - 内存占用（MB）
  - 余弦相似度与 FP32 基线的相关性（衡量精度损失）
- 输出表格到终端 + CSV

## 步骤 4.4：KV Cache INT8 量化 `src/generation/kv_cache_quant.py`

- `KVCacheQuantizer`：
  - 对 KV Cache 做 per-channel / per-token INT8 量化
  - 量化粒度选择：
    - `per_channel`：每个 head 每个 channel 独立 scale（精度高）
    - `per_token`：每个 token 独立 scale（更激进）
  - `quantize(kv_cache_fp16) -> kv_cache_int8`：
    - K: `(n_layers, n_kv_heads, seq_len, head_dim)` → INT8
    - V: 同上
    - scale 和 zero_point 单独存储
  - `dequantize(kv_cache_int8) -> kv_cache_fp16`：
    - 需要时反量化回 FP16 用于 attention 计算
- 内存收益估算：
  - FP16 KV Cache: `2 * n_layers * n_kv_heads * seq_len * head_dim * 2` bytes
  - INT8 KV Cache: `2 * n_layers * n_kv_heads * seq_len * head_dim * 1` bytes + scales
  - 约 **45-50% 内存节省**
- 精度验证：
  - 反量化后与原始 FP16 的余弦相似度
  - 下游生成质量对比（困惑度变化 < 1%）

## 步骤 4.5：生成模型量化格式对比 `scripts/benchmark_generation.py`

- 同一模型不同量化格式对比：
  - 权重：FP16, Q8_0, Q4_K_M, Q4_0, IQ4_XS, Q2_K
  - KV Cache：FP16, INT8
  - 共 5×2 = 10 组组合
  - 指标：模型大小、加载时间、tokens/s、困惑度、最大上下文长度
- 用标准测试集（如 wiki 片段）评估
- 输出对比矩阵

## 步骤 4.6：内存与延迟报告 `docs/quantization_report.md`

- 汇总 4.3 + 4.4 + 4.5 的结果
- 推荐配置：嵌入 INT8 + 权重 Q4_K_M + KV Cache INT8
- 目标：总内存 < 3.5GB（含索引），支持 8K 以上上下文
- 包含 KV Cache 量化前后的长文本场景对比

**Phase 4 验证**：
```bash
python scripts/benchmark_embedding.py
python scripts/benchmark_generation.py       # 含 KV Cache 量化对比
python scripts/benchmark_kv_cache_quant.py   # KV Cache 量化精度+内存对比
# 权重量化：精度损失 < 1%，速度提升 > 2x
# KV Cache 量化：内存节省 ~50%，生成质量下降 < 1%
```

---

# Phase 5: 生产化

**目标**：可 pip install，可 Docker 部署，文档齐全

## 步骤 5.1：FastAPI 服务 `src/api/`

- `src/api/__init__.py`
- `src/api/server.py`：
  - `POST /ask`：单次问答
  - `POST /chat`：多轮对话（带 session_id）
  - `POST /index`：索引文档
  - `GET /health`：健康检查
  - `GET /docs`：列出已索引文档
- `src/api/middleware.py`：请求日志、CORS、错误处理

## 步骤 5.2：CLI `serve` 子命令

- `agentrag serve --host 0.0.0.0 --port 8000`
- 启动 FastAPI + uvicorn

## 步骤 5.3：配置系统 `src/config.py`

- `Config` dataclass，从 YAML 加载
- 字段：
  - `embedding.model_name`, `embedding.device`, `embedding.quantized`
  - `index.ivf_k`, `index.ivf_n_probe`, `index.pq_n_subvectors`, `index.pq_n_bits`
  - `retrieval.top_k`, `retrieval.rerank_top_n`
  - `generation.model_path`, `generation.n_ctx`, `generation.temperature`
  - `chunking.chunk_size`, `chunking.chunk_overlap`

## 步骤 5.4：结构化日志 `src/utils/logger.py`

- 使用 Python `logging` 模块
- 格式：`时间 | 级别 | 模块 | 消息`
- INFO：正常流程
- WARNING：回退行为（如量化模型不可用回退 FP32）
- ERROR：需要关注的失败

## 步骤 5.5：Dockerfile

- 基础镜像：`python:3.11-slim`
- 安装依赖 + 编译 C++ 模块
- 暴露端口 8000
- 默认命令：`agentrag serve`

## 步骤 5.6：文档完善

- `README.md`：完整安装、快速开始、示例
- `docs/architecture.md`：架构图 + 各组件说明
- `docs/getting_started.md`：5 分钟快速开始
- `docs/api_reference.md`：CLI + REST API 文档

## 步骤 5.7：CI 配置 `.github/workflows/`

- `ci.yml`：pytest + C++ 编译 + lint
- `release.yml`：发布到 PyPI

**Phase 5 验证**：
```bash
pip install agentrag
agentrag serve
curl -X POST http://localhost:8000/ask -d '{"query": "test"}'
```

---

## 依赖清单

### Python 依赖

| 包名 | 用途 | Phase | 安装方式 |
|------|------|-------|---------|
| `numpy` | 向量计算 | 0 | pip (LR1 已有 ✅) |
| `pyyaml` | 配置解析 | 0 | pip (LR1 已有 ✅) |
| `click` | CLI | 0 | pip (LR1 已有 ✅) |
| `rich` | 终端美化 | 0 | pip (LR1 已有 ✅) |
| `torch` | PyTorch CUDA 后端 | 1 | pip (LR1 已有 2.6.0+cu118 ✅) |
| `transformers` | HuggingFace 模型加载 | 1 | pip (LR1 已有 4.52.3 ✅) |
| `onnxruntime-gpu` | ONNX CUDA 推理 | 1 | pip (LR1 已有 1.23.2 ✅) |
| `sentence-transformers` | 嵌入模型高层 API | 1 | `pip install sentence-transformers` |
| `llama-cpp-python` | GGUF + CUDA 推理 | 1 | `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python` |
| `jieba` | 中文分词（C++ 版本就绪前的回退） | 1 | `pip install jieba` |
| `pybind11` | C++ 绑定（编译头文件） | 2 | `pip install pybind11`（或 CMake FetchContent） |
| `pytest` + `pytest-cov` | 测试框架 | 1 | `pip install pytest pytest-cov` |
| `fastapi` + `uvicorn` | REST API 服务 | 5 | `pip install fastapi uvicorn` |

### 源码参考库（克隆到工作区）

| 仓库 | 用途 | 路径 |
|------|------|------|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | 推理引擎核心、KV Cache API 参考 | `D:\010\git工作区\llama.cpp` |

### 编译依赖（C++）

| 依赖 | 用途 | Phase | 方式 |
|------|------|-------|------|
| `pybind11` | Python ↔ C++ 绑定 | 2 | CMake FetchContent |
| `CMake >= 3.18` | 构建系统 | 0 | conda install / 系统已有 |
| C++17 编译器 | MSVC（Windows） | 0 | Visual Studio Build Tools |
| CUDA Toolkit >= 11.8 | GPU 加速 K-Means（可选进阶） | 2+ | 系统已有 (11.8) |

### 编译依赖（C++）

| 依赖 | 用途 | Phase |
|------|------|-------|
| `pybind11` | Python ↔ C++ 绑定 | 2 |
| `CMake >= 3.18` | 构建系统 | 0 |
| C++17 编译器 | MSVC / GCC / Clang | 0 |

### 模型文件（用户自行下载）

| 模型 | 用途 | 大小 |
|------|------|------|
| BAAI/bge-small-zh-v1.5 | 中文嵌入 | ~95MB |
| BAAI/bge-reranker-base | 重排序 | ~1.1GB |
| Qwen2.5-1.5B-Instruct GGUF | 生成 | ~1-2GB (Q4_K_M) |

---

## 风险清单

| 风险 | 影响 | Phase | 缓解 |
|------|------|-------|------|
| llama-cpp-python 安装失败 | Phase 1 阻塞 | 1 | 提供 MockLLM；提供预编译 wheel 链接 |
| C++ IVF 索引开发周期长 | Phase 2 延期 | 2 | Phase 1 numpy 版本已可用，C++ 为性能优化 |
| ONNX 量化精度损失过大 | Phase 4 回退 | 4 | 保留 FP32 选项，量化是可选的 |
| KV Cache INT8 量化精度不足 | 长文本生成质量下降 | 4 | per-channel 先验证；保留 FP16 回退；per-token 作为可选 |
| Prefix Cache 命中率低于预期 | 优化收益不明显 | 3 | 仅实时监控，不做硬依赖；命中率 < 30% 时自动禁用 |
| 中文分词 C++ 自研效果差 | BM25 精度低 | 2 | 先用 jieba C 接口，后优化 |
| 个人开发时间有限 | 整体延期 | 全 | 每个 Phase 独立可交付，不阻塞后续 |
