/**
 * ============================================================================
 * BM25 倒排索引 —— 稀疏检索（关键词匹配）
 * ============================================================================
 *
 * BM25 是经典的文本检索算法，基于 TF-IDF 改进。
 *
 * 公式：
 *   score(D, Q) = sum IDF(qi) * (tf * (k1+1)) / (tf + k1*(1-b+b*len/avg_len))
 *   其中：
 *     qi    : 查询中的第 i 个词
 *     tf    : 词在当前文档中的出现次数
 *     len   : 当前文档长度
 *     avg_len: 平均文档长度
 *     k1    : 词频饱和度参数（默认 1.5，tf 贡献上限约 k1+1）
 *     b     : 文档长度归一化参数（默认 0.75，b=0 不归一化，b=1 完全归一化）
 *     IDF   : 逆文档频率 = log((N - df + 0.5) / (df + 0.5) + 1)
 *
 * 与向量检索互补：
 *   - 稠密检索擅长"语义相似"（"汽车" ≈ "轿车"）
 *   - BM25 擅长"精确匹配"（人名、术语、编号等）
 *   - 混合检索（RRF 融合）通常比单一方式 Recall 高 5-15%
 */

#pragma once

#include "vector_types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace agentrag {
namespace core {

/**
 * BM25 倒排索引
 *
 * 存储倒排列表：{词 → [(文档ID, 词频), ...]}
 * 检索时对每个查询词取出倒排列表，累加 BM25 分数。
 */
class BM25Index {
public:
    BM25Index() = default;

    /**
     * 配置 BM25 参数
     *
     * @param k1  词频饱和度（默认 1.5）
     * @param b   长度归一化（默认 0.75）
     */
    void set_params(float k1, float b) { k1_ = k1; b_ = b; }

    /**
     * 添加文档
     *
     * @param doc_id  文档 ID（整数）
     * @param tokens  分词结果（词列表）
     */
    void add_document(int32_t doc_id, const std::vector<std::string>& tokens);

    /**
     * 删除文档
     *
     * @param doc_id  文档 ID
     */
    void remove_document(int32_t doc_id);

    /**
     * 检索
     *
     * @param query_tokens  查询分词结果
     * @param top_k          返回数量
     * @return               [(doc_id, bm25_score), ...] 按分数降序
     */
    std::vector<SearchResult> search(
        const std::vector<std::string>& query_tokens,
        size_t top_k = 10
    ) const;

    // ── 属性 ──
    size_t num_docs() const { return doc_lengths_.size(); }
    float avg_doc_length() const { return avg_doc_length_; }

private:
    float k1_ = 1.5f;
    float b_ = 0.75f;

    // 倒排列表：{词 → [(doc_id, 词频), ...]}
    std::unordered_map<std::string, std::vector<std::pair<int32_t, int32_t>>> posting_lists_;

    // 文档长度
    std::vector<int32_t> doc_lengths_;
    float avg_doc_length_ = 0.0f;

    // 已删除的文档 ID（惰性删除，标记后不参与检索）
    std::unordered_set<int32_t> deleted_;
};

}  // namespace core
}  // namespace agentrag
