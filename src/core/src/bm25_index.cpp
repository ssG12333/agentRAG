/**
 * BM25 倒排索引实现
 */

#include "bm25_index.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace agentrag {
namespace core {

void BM25Index::add_document(int32_t doc_id, const std::vector<std::string>& tokens) {
    // 统计每个词在此文档中的频率
    std::unordered_map<std::string, int32_t> tf;
    for (const auto& token : tokens) {
        tf[token]++;
    }

    // 更新倒排列表
    for (const auto& [term, freq] : tf) {
        posting_lists_[term].emplace_back(doc_id, freq);
    }

    // 记录文档长度
    int32_t len = static_cast<int32_t>(tokens.size());
    doc_lengths_.push_back(len);

    // 更新平均文档长度
    avg_doc_length_ = 0.0f;
    for (int32_t l : doc_lengths_) {
        avg_doc_length_ += static_cast<float>(l);
    }
    avg_doc_length_ /= static_cast<float>(doc_lengths_.size());
}

void BM25Index::remove_document(int32_t doc_id) {
    deleted_.insert(doc_id);
}

std::vector<SearchResult> BM25Index::search(
    const std::vector<std::string>& query_tokens,
    size_t top_k) const
{
    if (query_tokens.empty() || doc_lengths_.empty()) return {};

    // 文档总数（排除已删除的）
    int32_t N = static_cast<int32_t>(doc_lengths_.size()) - static_cast<int32_t>(deleted_.size());

    // 累计每个文档的 BM25 分数
    std::unordered_map<int32_t, float> scores;

    for (const auto& term : query_tokens) {
        auto it = posting_lists_.find(term);
        if (it == posting_lists_.end()) continue;

        const auto& postings = it->second;
        int32_t df = static_cast<int32_t>(postings.size());  // 文档频率

        // IDF: 逆文档频率
        float idf = std::log((N - df + 0.5f) / (df + 0.5f) + 1.0f);

        for (const auto& [doc_id, tf] : postings) {
            if (deleted_.count(doc_id)) continue;

            int32_t len = doc_lengths_[doc_id];
            float norm = 1.0f - b_ + b_ * (static_cast<float>(len) / avg_doc_length_);
            float numerator = static_cast<float>(tf) * (k1_ + 1.0f);
            float denominator = static_cast<float>(tf) + k1_ * norm;
            scores[doc_id] += idf * numerator / denominator;
        }
    }

    // 转换为 SearchResult 并排序
    std::vector<SearchResult> results;
    results.reserve(scores.size());
    for (const auto& [doc_id, score] : scores) {
        results.push_back({doc_id, score});
    }

    // 降序排列（分数越高越相关）
    std::sort(results.begin(), results.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.score > b.score;
        });

    if (results.size() > top_k) {
        results.resize(top_k);
    }

    return results;
}

}  // namespace core
}  // namespace agentrag
