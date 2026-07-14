# Transformer 模型简介

## 1. 什么是 Transformer

Transformer 是 Google 在 2017 年提出的序列到序列模型，论文标题为 "Attention Is All You Need"。
它完全基于自注意力机制（Self-Attention），摒弃了传统的循环神经网络（RNN）结构。

## 2. 自注意力机制

自注意力机制的核心思想是：对于序列中的每一个位置，计算它与其他所有位置的相似度，
并用这些相似度对所有位置的值进行加权求和。

具体计算过程：
- 输入 X 经过三个线性变换得到 Q（Query）、K（Key）、V（Value）
- 计算注意力分数：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

其中 d_k 是 Key 的维度，除以 sqrt(d_k) 是为了防止点积过大导致 softmax 梯度消失。

## 3. 计算复杂度分析

对于长度为 n 的序列：
- 自注意力的计算复杂度为 O(n² * d)
- 其中 n² 来自 QK^T 的结果矩阵大小
- d 是每个位置的向量维度

这意味着当序列很长时，计算量会急剧增长。这也是为什么后来出现了许多高效注意力变体。

## 4. 多头注意力

Transformer 使用多头注意力机制，将 Q、K、V 分成多个头，每个头独立计算注意力：
- 每个头可以关注不同的特征子空间
- 例如 8 个头，每个头维度 d_k = d_model / 8 = 64（假设 d_model = 512）

## 5. 位置编码

由于 Transformer 没有循环结构，无法感知序列顺序，因此需要位置编码。
原始论文使用正弦/余弦位置编码：
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

## 6. 训练与推理

Transformer 训练时可以并行计算所有位置（teacher forcing），
但推理时需要逐 token 生成（自回归），每一步都需要计算 KV Cache 以复用之前的计算结果。

KV Cache 可以避免重复计算，将推理复杂度从 O(n³) 降到 O(n²)。
