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
    if (vectors == nullptr || n == 0 || dim == 0) {
        throw std::invalid_argument("vectors, n and dim must be non-empty");
    }
    if (n_subvectors == 0) {
        throw std::invalid_argument("n_subvectors must be greater than zero");
    }
    if (n_bits == 0 || n_bits > 8) {
        throw std::invalid_argument("n_bits must be in [1, 8] for uint8 codes");
    }
    if (dim % n_subvectors != 0) {
        throw std::invalid_argument("dim must be divisible by n_subvectors");
    }

    dim_ = dim;
    n_subvectors_ = n_subvectors;
    n_bits_ = n_bits;
    trained_codebook_size_ = std::min<size_t>(1u << n_bits, n);

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
        KMeansResult km = kmeans(
            sub_vectors.data(), n, d_sub, trained_codebook_size_, n_iters);

        // 存入全局码本数组
        float* codebook_ptr = codebooks_.data() + m * codebook_stride;
        std::memcpy(
            codebook_ptr,
            km.centroids.data(),
            trained_codebook_size_ * d_sub * sizeof(float));
    }
}

std::vector<uint8_t> ProductQuantizer::encode(const float* vec) const {
    if (vec == nullptr || trained_codebook_size_ == 0) {
        throw std::runtime_error("ProductQuantizer is not trained");
    }
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
        for (size_t c = 1; c < trained_codebook_size_; ++c) {
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
    for (const auto& codes : codes_batch) {
        if (codes.size() != n_subvectors_) {
            throw std::invalid_argument("each code must have n_subvectors entries");
        }
    }

    std::vector<uint8_t> codes_flat;
    codes_flat.reserve(codes_batch.size() * n_subvectors_);
    for (const auto& codes : codes_batch) {
        codes_flat.insert(codes_flat.end(), codes.begin(), codes.end());
    }
    return compute_distances_flat(query, codes_flat, codes_batch.size());
}

std::vector<float> ProductQuantizer::compute_distances_flat(
    const float* query,
    const std::vector<uint8_t>& codes_flat,
    size_t n) const
{
    if (query == nullptr || trained_codebook_size_ == 0) {
        throw std::runtime_error("ProductQuantizer is not trained");
    }
    if (codes_flat.size() != n * n_subvectors_) {
        throw std::invalid_argument("codes_flat size does not match n * n_subvectors");
    }

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
            uint8_t code = codes_flat[i * n_subvectors_ + m];
            if (code >= trained_codebook_size_) {
                throw std::invalid_argument("PQ code references an untrained codeword");
            }
            dist += dist_table[m * k + code];
        }
        distances[i] = dist;
    }

    return distances;
}

}  // namespace core
}  // namespace agentrag
