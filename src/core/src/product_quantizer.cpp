/**
 * 乘积量化器实现
 */

#include "product_quantizer.h"
#include "kmeans.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace agentrag {
namespace core {

static float squared_l2(const float* a, const float* b, size_t dim) {
    float dist = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

void ProductQuantizer::train(
    const float* vectors,
    size_t n,
    size_t dim,
    size_t n_subvectors,
    size_t n_bits,
    int32_t n_iters)
{
    if (dim % n_subvectors != 0) {
        throw std::invalid_argument("dim must be divisible by n_subvectors");
    }

    dim_ = dim;
    n_subvectors_ = n_subvectors;
    n_bits_ = n_bits;

    size_t d_sub = dim / n_subvectors;          // 每段维度
    size_t k = 1u << n_bits;                     // 码本大小 = 2^n_bits
    size_t codebook_stride = k * d_sub;

    codebooks_.resize(n_subvectors * codebook_stride, 0.0f);

    // 对每段子向量独立训练 K-Means
    for (size_t m = 0; m < n_subvectors; ++m) {
        // 提取第 m 段的所有子向量
        std::vector<float> sub_vectors(n * d_sub);
        for (size_t i = 0; i < n; ++i) {
            std::memcpy(
                sub_vectors.data() + i * d_sub,
                vectors + i * dim + m * d_sub,
                d_sub * sizeof(float)
            );
        }

        // K-Means 训练码本
        size_t actual_k = std::min(k, n);  // 码本大小不能超过样本数
        KMeansResult km = kmeans(sub_vectors.data(), n, d_sub, actual_k, n_iters);

        // 存入全局码本数组
        float* codebook_ptr = codebooks_.data() + m * codebook_stride;
        std::memcpy(codebook_ptr, km.centroids.data(), actual_k * d_sub * sizeof(float));
        // 如果 actual_k < k（样本太少），剩余码字保持 0
    }
}

std::vector<uint8_t> ProductQuantizer::encode(const float* vec) const {
    std::vector<uint8_t> codes(n_subvectors_);
    size_t d_sub = dim_ / n_subvectors_;
    size_t k = 1u << n_bits_;
    size_t codebook_stride = k * d_sub;

    for (size_t m = 0; m < n_subvectors_; ++m) {
        const float* sub_vec = vec + m * d_sub;
        const float* codebook = codebooks_.data() + m * codebook_stride;

        // 在此段码本中找最近码字
        uint8_t best = 0;
        float best_dist = squared_l2(sub_vec, codebook, d_sub);
        for (size_t c = 1; c < k; ++c) {
            float dist = squared_l2(sub_vec, codebook + c * d_sub, d_sub);
            if (dist < best_dist) {
                best_dist = dist;
                best = static_cast<uint8_t>(c);
            }
        }
        codes[m] = best;
    }

    return codes;
}

std::vector<std::vector<uint8_t>> ProductQuantizer::encode_batch(
    const float* vectors, size_t n) const
{
    std::vector<std::vector<uint8_t>> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = encode(vectors + i * dim_);
    }
    return result;
}

std::vector<float> ProductQuantizer::compute_distances(
    const float* query,
    const std::vector<std::vector<uint8_t>>& codes_batch) const
{
    size_t n = codes_batch.size();
    size_t d_sub = dim_ / n_subvectors_;
    size_t k = 1u << n_bits_;
    size_t codebook_stride = k * d_sub;

    // 1. 预计算距离表：query 每段到每个码字的距离
    //    dist_table[m * k + c] = distance(query_sub, codebook[m][c])
    std::vector<float> dist_table(n_subvectors_ * k);
    for (size_t m = 0; m < n_subvectors_; ++m) {
        const float* q_sub = query + m * d_sub;
        const float* codebook = codebooks_.data() + m * codebook_stride;
        float* table_row = dist_table.data() + m * k;

        for (size_t c = 0; c < k; ++c) {
            table_row[c] = squared_l2(q_sub, codebook + c * d_sub, d_sub);
        }
    }

    // 2. 查表累加距离（ADC）
    std::vector<float> distances(n);
    for (size_t i = 0; i < n; ++i) {
        float dist = 0.0f;
        for (size_t m = 0; m < n_subvectors_; ++m) {
            uint8_t code = codes_batch[i][m];
            dist += dist_table[m * k + code];
        }
        distances[i] = dist;
    }

    return distances;
}

}  // namespace core
}  // namespace agentrag
