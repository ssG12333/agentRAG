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
- 提交：`95af3fe` feat: Phase 2 C++核心 + Python封装

---

## 2026-07-15 15:30 | C++ 编译成功

### 完成内容
- PyBind11 查找逻辑优化(优先 conda 本地 → FetchContent 兜底)
- Ninja 编译通过, 自动生成 `agentrag_core.cp310-win_amd64.pyd`
- K-Means / IVF / PQ / BM25 四大 C++ 模块全部可 import

### 验证结果
- `agentrag_core.add(1,2)=3` ✅
- K-Means / IVF / PQ / BM25 功能验证 ✅

---

## 2026-07-15 16:00 | Phase 3: Agent + Prefix Caching

### 当前目标
- Agent 智能体: ReAct 循环 + 工具系统 + 对话记忆 + 查询改写

### 完成内容
- `src/agent/tools.py`: Tool 数据类 + ToolRegistry(XML格式工具描述)
- `src/agent/memory.py`: ConversationMemory + WorkingMemory
- `src/agent/loop.py`: ReActAgent(Thought→Action→Observation循环)
- `src/generation/prefix_cache.py`: PrefixCache(LRU) + PrefixAwareEngine
- `src/retrieval/rewriter.py`: QueryRewriter(指代消解)
- `src/cli/main.py`: `agentrag chat` 交互式 Agent 对话(工具调用+多轮记忆)
- `tests/test_phase3.py`: 13 个 Agent 层测试
- CMakeList 新增 POST_BUILD 自动复制 .pyd 到项目根

### 验证结果
- 执行命令: `pytest tests/ -v`
- 结果: **27/29 passed**(2 个失败依赖 C++ SearchResult 绑定重编, 重编后 29/29)

### 关键结论
- ReAct 循环解析鲁棒(支持 Final Answer / tool_call 两种格式)
- QueryRewriter 在无 LLM 或无历史时优雅回退
- PrefixCache LRU 淘汰策略正确

### 问题与风险
- PrefixAwareEngine 实际 prefill 跳过需 llama.cpp C API 支持(Phase 4)
- llama.cpp GPU 版仍未编译(CUDA 11.8 不兼容 VS 2026)

### 下一步
- 重编 C++ 验证 SearchResult 绑定 → 29/29 全绿
- Phase 4: 量化优化

### 文件清单(Phase 3 新增)
- `src/agent/tools.py`: 工具系统
- `src/agent/memory.py`: 对话/工作记忆
- `src/agent/loop.py`: ReAct Agent 循环
- `src/generation/prefix_cache.py`: 前缀缓存
- `src/retrieval/rewriter.py`: 查询改写
- `tests/test_phase3.py`: 13 个 Agent 测试

### Git
- 待提交

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

---

## 2026-07-15 17:23 | 进度审计与下一阶段规划

### 当前目标
- 核对 Phase 2/3 的实际完成度，建立进入 Phase 4 前的收尾和基线计划

### 完成内容
- 对照 `LOG.md`、`TODO.md`、代码调用关系和当前 C++ 扩展检查项目状态
- 将下一阶段调整为“Phase 2/3 收尾与基线建设”，暂不直接进入 Phase 4
- 更新 `TODO.md`，拆分 P0 全绿回归、P1 主链路集成、P2 量化基线三组任务
- 为 Phase 4 增加启动门槛，并将 KV Cache INT8 调整为不阻塞核心交付的研究性实验

### 修改文件
- `TODO.md`：校正 Phase 2/3 状态，补充优先级、验收标准和 Phase 4 启动门槛
- `LOG.md`：追加本次进度审计、验证证据和下一阶段计划

### 验证结果
- 执行命令：`E:\anaconda3\.conda\envs\agentrag\python.exe -m pytest tests -q`
- 结果：**28 passed, 1 failed**
- 唯一失败：`tests/test_phase2_python.py::test_sparse_retriever_add_search`
- 失败证据：当前 `.pyd` 未导出 `agentrag::core::SearchResult`；`hasattr(agentrag_core, "SearchResult")` 为 `False`
- 时间证据：项目根目录 `.pyd` 生成于 15:06，`src/core/pybind/module.cpp` 修改于 15:16，当前二进制早于绑定源码
- 文档检查：`git diff --check` 未发现空白错误，仅有 Windows LF/CRLF 提示

### 关键结论
- Phase 3 的 13 个组件测试已通过，但完整仓库当前可信状态是 28/29，不能记录为 29/29
- C++ 源码已添加 `SearchResult` 绑定，下一步应先重编并重新验证，不能把“预计通过”写成已通过
- 当前 IVF 与 PQ 是独立组件，尚无真正的 IVF-PQ 组合索引及 Python VectorStore
- `HybridRetriever` 和 reranker 尚未接入 CLI 主链路
- `PrefixCache` 管理逻辑已实现，但 ReActAgent 未调用 `PrefixAwareEngine`，`/cache` 尚不能反映真实推理缓存命中
- 量化实验缺少固定评估集和 FP32 基线，Phase 4 应在基线可复现后启动

### 问题与风险
- `conda run` 在 Windows GBK 控制台回显 pytest 输出时触发 `UnicodeEncodeError`，后续验证应直接调用环境 Python 或显式使用 UTF-8
- llama.cpp Python 接口未提供当前设计所需的原始 KV Cache 操作，KV Cache INT8 只能先做可行性研究
- 工作区已有多项未提交代码和文档修改，后续提交必须只纳入已核对的本轮内容

### 下一步
1. 重新编译 C++ 扩展，确认 `SearchResult` 已导出并取得 29/29 全绿结果
2. 决定 IVF-PQ 的交付范围，完成混合检索、reranker 和 Prefix Cache 的主链路集成
3. 增加 C++、检索和 Agent 端到端测试及基准
4. 建立 FP32 嵌入评估基线，再启动 ONNX/INT8 量化实验

### 终止与回退条件
- 完整回归未全绿时，不进入 Phase 4
- IVF-PQ 组合实现成本超过当前阶段预算时，回退为明确标注的 IVF/PQ 教学原型
- Prefix Cache 无可用底层 API 或实测收益不足时，保留监控/管理器，不把它作为性能提升结论
- 量化导致核心检索指标明显退化时，保留 FP32 路径并停止扩大实验规模

### Git
- 提交：未创建；当前工作区包含本次任务开始前已有的未提交修改

---

## 2026-07-15 18:00 | P0：恢复可信全绿状态

### 当前目标
- 重编最新 C++ 绑定，恢复完整回归全绿，并同步项目状态文档

### 完成内容
- 使用 agentrag 环境内已安装的 pybind11 重新配置现有 Ninja 构建目录
- 设置 `PYBIND11_FINDPYTHON=ON`，避免 CMake 4.4 调用 pybind11 旧版 Python 查找逻辑
- 增量编译并自动复制最新 `agentrag_core.cp310-win_amd64.pyd` 到项目根目录
- 更新根 CMake 配置，默认使用现代 FindPython
- 重写 `scripts/build_cpp.bat` 为可重复的增量 Ninja 构建，不再删除 build 或依赖 `conda activate`
- 校正中英文 README：Phase 2/3 标记为集成收尾中，不再把独立 IVF/PQ 组件表述为完整 IVF-PQ 主链路
- 更新 `TODO.md` 中的 SearchResult 导出与 29/29 回归状态

### 修改文件
- `README.md`：校正 Phase 2/3 路线图和组件描述
- `README-cn.md`：同步中文路线图和组件描述
- `TODO.md`：完成 P0 编译、导出检查、完整测试和文档同步事项
- `LOG.md`：记录 P0 构建过程和验证结果
- `CMakeLists.txt`：启用现代 FindPython
- `scripts/build_cpp.bat`：使用固定 agentrag 环境路径执行增量 Ninja 构建

### 验证结果
- CMake 配置：成功；使用 Ninja、Release、本地 pybind11 和现代 FindPython
- C++ 构建：成功；7 个编译/链接步骤全部完成
- 构建脚本复验：成功；CMake 配置完成，Ninja 报告 `no work to do`
- 导出检查：`hasattr(agentrag_core, "SearchResult")` 为 `True`
- 完整测试：`python -m pytest tests -q -p no:cacheprovider` → `29 passed in 0.88s`

### 关键结论
- SearchResult 失败由旧 `.pyd` 引起，源码绑定本身有效
- P0 的编译和回归目标已经完成，下一阶段进入 P1 主链路集成
- 构建仍会提示 `vswhere.exe` 不在 PATH，但在已初始化的 VS x64 环境中不阻塞配置和编译

### 问题与风险
- 构建脚本仍包含当前机器的 Conda 环境和 VS 工具链路径；跨机器使用前需要参数化
- `.pytest_cache` 权限异常仍存在，已通过 `-p no:cacheprovider` 获得无警告回归结果

### 下一步
1. 审查并提交本轮 Phase 3/P0 相关修改
2. 决定真正 IVF-PQ 与教学原型之间的交付范围
3. 接入 HybridRetriever、reranker 和 PrefixAwareEngine
4. 增加 Agent 与检索主链路端到端测试

### Git
- 提交：待创建
