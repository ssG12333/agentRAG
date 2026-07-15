/**
 * 残差 IVF-PQ 索引：粗量化分桶 + PQ 压缩残差 + ADC 检索。
 */

#pragma once

#include "product_quantizer.h"
#include "vector_types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace agentrag {
namespace core {

class IVFPQIndex {
public:
    IVFPQIndex() = default;

    void build(
        const float* vectors,
        size_t n,
        size_t dim,
        size_t n_clusters = 256,
        size_t n_probe = 8,
        size_t n_subvectors = 64,
        size_t n_bits = 8,
        int32_t n_iters = 25
    );

    std::vector<SearchResult> search(const float* query, size_t top_k) const;

    void save(const std::string& path) const;
    void load(const std::string& path);

    size_t size() const { return n_vectors_; }
    size_t dim() const { return dim_; }
    size_t num_clusters() const { return n_clusters_; }
    size_t n_probe() const { return n_probe_; }
    size_t n_subvectors() const { return pq_.n_subvectors(); }
    size_t n_bits() const { return pq_.n_bits(); }
    size_t codes_bytes() const { return n_vectors_ * pq_.n_subvectors(); }
    size_t estimated_memory_bytes() const;

private:
    struct InvertedList {
        std::vector<int32_t> ids;
        std::vector<uint8_t> codes;  // (ids.size(), n_subvectors) 平坦存储
    };

    size_t n_vectors_ = 0;
    size_t dim_ = 0;
    size_t n_clusters_ = 0;
    size_t n_probe_ = 0;

    std::vector<float> centroids_;
    ProductQuantizer pq_;
    std::vector<InvertedList> lists_;
};

}  // namespace core
}  // namespace agentrag
