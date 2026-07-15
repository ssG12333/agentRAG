# 术语表

| 术语 | 含义 |
|---|---|
| RAG | Retrieval-Augmented Generation，先检索外部知识再生成 |
| Document | 原始文档及其 metadata |
| Chunk | 从文档切出的最小检索单元 |
| Embedding | 将文本映射到稠密向量的过程或结果 |
| Bi-Encoder | query 和文档独立编码，适合大规模召回 |
| Cross-Encoder | query 与文档共同编码，适合少量候选精排 |
| Cosine similarity | 用夹角衡量向量方向相似度 |
| ANN | Approximate Nearest Neighbor，近似最近邻 |
| IVF | Inverted File，用粗聚类缩小搜索范围 |
| PQ | Product Quantization，把向量分段后分别量化 |
| Residual | 向量减去粗聚类中心后的差值 |
| ADC | Asymmetric Distance Computation，原查询到量化编码的近似距离 |
| BM25 | 基于词频、逆文档频率和长度归一化的稀疏排序函数 |
| RRF | Reciprocal Rank Fusion，用排名倒数融合多个结果列表 |
| Recall@k | 前 k 个结果覆盖相关项的比例 |
| MRR | 第一个相关结果排名倒数的查询平均值 |
| nDCG | 考虑位置折损和分级相关性的归一化指标 |
| Prompt | 送给生成模型的完整输入文本 |
| Prefill | 一次处理输入 prompt 并写入 KV Cache 的阶段 |
| Decode | 基于已有上下文逐 token 生成的阶段 |
| KV Cache | 自回归推理中缓存历史 token 的 Key/Value |
| Prefix Cache | 对相同前缀复用计算；本项目当前仅有逻辑命中统计 |
| ReAct | 交替执行推理、动作和观察的 Agent 范式 |
| GGUF | llama.cpp 生态常用的模型文件格式 |
| Quantization | 用较低位宽表示权重或激活以降低资源成本 |
