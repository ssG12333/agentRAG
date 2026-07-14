# agentRAG 项目进展日志

## 2026-07-14 21:00 | 项目启动与方向决策

### 当前目标
- 确定 AI Agent + RAG 项目的技术方向和定位

### 完成内容
- 审查工作区已有项目：AI INFRA（推理系统）、深度量化（GPTQ/AWQ）、NCNN、量化
- 确定方向：本地轻量 RAG Agent + 从零搭建理解全链路
- 确定定位：学习简历项目 + 长期迭代开源
- 确定语言：Python（应用编排）+ C++（热路径加速）
- 决策：不依赖 LangChain/LlamaIndex/FAISS 等高层框架

### 关键结论
- 核心约束：本地运行、无云API依赖、量化优先、模块可替换
- 6 层架构：文档 → 嵌入 → 索引 → 检索 → 生成(+KV Cache) → Agent
- 5 个 Phase 分阶段实施

### 下一步
- Phase 0: 项目脚手架

---

## 2026-07-14 21:30 | 计划细化与环境搭建

### 当前目标
- 细化 PLAN.md 到文件级别，搭建 conda 环境

### 完成内容
- PLAN.md 写入 `agentRAG项目/`（非 `.claude/plans/`），细分 53 步
- KV Cache 内容融入各 Phase：监控(Phase 1) / Prefix Caching(Phase 3) / INT8量化(Phase 4)
- 链路合理性审查：
  - 修正混合检索融合公式（线性加权 → RRF）
  - 修正分词方案（C++从零 → cppjieba）
  - 调整 Phase 顺序（量化前置）
  - 添加持久化和评估体系
- GPU 方案确定：RTX 2060 6GB → Qwen2.5-3B 全GPU + BGE-base-zh
- 创建 conda 环境 `agentrag` (Python 3.10)
- 清理旧 conda 环境 5 个
- llama-cpp-python CPU 版已安装（GPU 版需升级 CUDA 11.8→13.x）

### 关键决策
- CUDA 11.8 不兼容 VS 2026，暂用 CPU 版 llama.cpp，Phase 4 再切 GPU
- 暂时放弃 KV Cache INT8 自研（llama.cpp API 不暴露 KV Cache 数值）

### 下一步
- Phase 0

---

## 2026-07-14 22:00 | Phase 0: 项目脚手架

### 当前目标
- 创建目录结构、构建系统、CLI 入口

### 完成内容
- 目录结构：src/{document,embedding,index,retrieval,generation,agent,cli,core,api,utils}
- `pyproject.toml`：包名 agentrag，入口点 agentrag → src.cli.main
- `CMakeLists.txt`：C++ 构建入口（pybind11 via FetchContent）
- `src/core/`：C++ 骨架（hello.cpp + pybind/module.cpp + vector_types.h）
- `src/cli/main.py`：index / ask / chat / serve 四个子命令
- `src/document/__init__.py`：Document + Chunk 数据模型
- `README.md`：项目介绍 + 开发状态
- `configs/default.yaml`：全参数配置
- `.gitignore` / `LICENSE`

### 验证结果
- `pip install -e .` 成功
- `agentrag --help` 输出帮助信息
- `agentrag --version` → agentrag, version 0.1.0

### Git
- 提交：`3420098` feat: Phase 0 + Phase 1 项目脚手架与MVP基础RAG管道
- 提交：`e4f9ebf` docs: 美化 README
- 提交：`4fe715c` docs: 全部源码添加详细中文注释
- 提交：`95af3fe` feat: Phase 2 C++核心 + Python封装
- 远程：git@github.com:ssG12333/agentRAG.git
- 分支：master

---

## 2026-07-14 23:00 | Phase 2: 混合检索 + C++ 加速

### 当前目标
- C++ 实现 IVF/PQ/BM25 索引 + pybind11 绑定 + Python 封装

### 完成内容
- `src/core/include/kmeans.h` + `src/kmeans.cpp`：K-Means 聚类（Lloyd 迭代）
- `src/core/include/ivf_index.h` + `src/ivf_index.cpp`：IVF 倒排索引
- `src/core/include/product_quantizer.h` + `src/product_quantizer.cpp`：PQ 乘积量化（ADC 距离查询）
- `src/core/include/bm25_index.h` + `src/bm25_index.cpp`：BM25 倒排索引
- `src/core/pybind/module.cpp`：pybind11 绑定（KMeans/IVFIndex/ProductQuantizer/BM25Index）
- `src/index/sparse_store.py`：SparseRetriever（C++ 后端 + Python 回退）
- `src/index/hybrid.py`：HybridRetriever（RRF 稠密+稀疏融合）
- `src/retrieval/reranker.py`：CrossEncoderReranker + NoOpReranker
- `scripts/build_cpp.bat`：编译脚本
- `tests/test_phase2_python.py`：5 个测试

### 验证结果
- 执行命令：`pytest tests/ -v`
- 结果：**16/16 passed**（11 Phase 1 + 5 Phase 2）
- C++ 编译：**未完成**（VS 2026 linker 报 Unknown system error -1 + CMake FetchContent 下载 pybind11 被墙）

### 关键结论
- RRF 融合公式验证正确：RRF = sum(1/(60+rank_i))
- Python BM25 回退可用，jieba 分词正常
- C++ 编译问题与代码无关，是环境问题（VS 2026 bug / 网络限制）

### 问题与风险
- VS 2026 (MSVC 19.51) linker 报 Unknown system error -1，怀疑杀毒软件拦截
- CMake FetchContent 下载 pybind11 被墙，需改用 conda install pybind11
- CUDA 11.8 不兼容 VS 2026，GPU 版 llama.cpp 仍需升级 CUDA

### 下一步
- 换 Ninja 生成器或 conda pybind11 解决编译问题
- 编译成功后跑 C++ 单元测试和基准
- Phase 3: Agent + Prefix Caching

### 文件清单（Phase 2 新增）
- `src/core/include/kmeans.h`：K-Means 头文件
- `src/core/include/ivf_index.h`：IVF 头文件
- `src/core/include/product_quantizer.h`：PQ 头文件
- `src/core/include/bm25_index.h`：BM25 头文件
- `src/core/src/kmeans.cpp`：K-Means 实现
- `src/core/src/ivf_index.cpp`：IVF 实现
- `src/core/src/product_quantizer.cpp`：PQ 实现
- `src/core/src/bm25_index.cpp`：BM25 实现
- `src/core/pybind/module.cpp`：pybind11 绑定（重写）
- `src/core/CMakeLists.txt`：编译配置（更新）
- `src/index/sparse_store.py`：稀疏检索封装
- `src/index/hybrid.py`：RRF 混合检索
- `src/retrieval/reranker.py`：重排序器
- `tests/test_phase2_python.py`：Phase 2 测试
- `scripts/build_cpp.bat`：编译脚本

### Git
- 提交：`95af3fe` feat: Phase 2 C++核心 + Python封装（16/16测试通过）

---

## 2026-07-14 22:30 | Phase 1: MVP 基础 RAG 管道

### 当前目标
- 端到端跑通：文档解析 → 分块 → 嵌入 → 检索 → 提示 → 生成

### 完成内容
- `document/parser.py`：MarkdownParser + TextParser + get_parser 工厂
- `document/chunker.py`：RecursiveChunker（递归字符分割）+ FixedWindowChunker（滑动窗口）
- `embedding/model.py`：SentenceTransformerEmbedding + CachedEmbedding(LRU)
- `index/vector_store.py`：NumpyVectorStore（余弦相似度 + np.savez 持久化）
- `retrieval/retriever.py`：Retriever（embed → search → return）
- `generation/prompt.py`：RAGPromptBuilder（系统提示 + 文档段落 + 历史 + 问题）
- `generation/engine.py`：LlamaCppEngine + MockLLM
- `generation/kv_cache.py`：KVCacheMonitor（使用率追踪 + 容量预警）
- `cli/main.py`：index/ask 命令接入真实逻辑（延迟导入 + 进度条 + 流式输出）
- `tests/test_phase1.py`：11 个测试（解析器/分块器/向量存储/检索器/提示/引擎）
- `examples/01_basic_rag.py`：端到端示例

### 验证结果
- 执行命令：`pytest tests/test_phase1.py -v`
- 结果：**11 passed in 1.11s**
- 端到端示例：BGE-small-zh 成功加载，3 chunks 索引完成
- 检索验证：复杂度分析段落排第一 (score=0.562) ✓
- SSL 证书问题已修复（certifi cacert.pem）

### 关键结论
- BGE-small-zh-v1.5 可正常下载和使用（512 维）
- NumpyVectorStore 全量遍历在小规模（<10K chunks）下够用
- MockLLM 可支撑无模型时的测试，真实模型需后续接入

### 问题与风险
- 终端 GBK 编码导致 emoji 和中文显示乱码（不影响功能）
- llama.cpp 目前是 CPU 版，GPU 版需升级 CUDA Toolkit
- 尚未下载真实 GGUF 模型文件

### 下一步
- Phase 2: 混合检索 + C++ 加速（IVF-PQ + BM25）

### 文件清单
- `src/document/__init__.py`：Document / Chunk 数据模型
- `src/document/parser.py`：Markdown / Text 解析器
- `src/document/chunker.py`：递归 / 固定窗口分块器
- `src/embedding/model.py`：SentenceTransformer + 缓存
- `src/index/vector_store.py`：Numpy 向量存储 + 持久化
- `src/retrieval/retriever.py`：检索器
- `src/generation/prompt.py`：RAG 提示构建
- `src/generation/engine.py`：llama.cpp / Mock 引擎
- `src/generation/kv_cache.py`：KV Cache 监控
- `src/cli/main.py`：CLI 入口
- `tests/test_phase1.py`：11 个单元测试
- `examples/01_basic_rag.py`：端到端示例
- `test_data/transformer_intro.md`：测试文档

### Git
- 尚未创建提交
