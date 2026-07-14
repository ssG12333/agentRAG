/**
 * ============================================================================
 * Product Quantization (PQ) 乘积量化 —— 向量压缩
 * ============================================================================
 *
 * 核心思想：把高维向量切分成多段，每段独立做 K-Means 建码本。
 * 一个 d 维向量变成 m 个 uint8 码本索引，存储从 d*4 字节 → m 字节。
 *
 * 算法：
 *   训练：
 *     1. 将每个 d 维向量均分为 m 段（每段 d_sub = d / m）
 *     2. 对每段的所有子向量做 K-Means（码本大小 = 2^n_bits）
 *     3. 得到 m 个码本，每个码本有 2^n_bits 个码字
 *
 *   编码：
 *     1. 对向量的每段，在对应码本中找最近码字
 *     2. 返回 m 个码本索引（codes）
 *
 *   距离查询（不对称距离 Asymmetric Distance Computation）：
 *     1. 预计算 query 每段到码本中每个码字的距离（距离表，m * 256 大小）
 *     2. 对每个被编码的向量：查表累加 m 个 dist → 近似距离
 *     3. 比解码后再算距离快 100x+（避免重复向量运算）
 *
 * 内存收益：
 *   d=768, m=64, n_bits=8: 768*4=3072 → 64 bytes，压缩比 48:1
 *   d=512, m=64, n_bits=8: 512*4=2048 → 64 bytes，压缩比 32:1
 *
 * 参考：
 *   Jegou et al. "Product Quantization for Nearest Neighbor Search" (2011)
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace agentrag {
namespace core {

/**
 * 乘积量化器
 *
 * 训练后得到 m 个码本，每个码本 256 个码字（n_bits=8 时）。
 */
class ProductQuantizer {
public:
    ProductQuantizer() = default;

    /**
     * 训练码本
     *
     * @param vectors        (n, dim) 训练向量
     * @param n             向量数量
     * @param dim           向量维度
     * @param n_subvectors  切分段数 m（dim 必须能被 m 整除）
     * @param n_bits        每段编码位数（8 = uint8, 每个码本 256 个码字）
     * @param n_iters       K-Means 迭代次数
     */
    void train(
        const float* vectors,
        size_t n,
        size_t dim,
        size_t n_subvectors = 64,
        size_t n_bits = 8,
        int32_t n_iters = 25
    );

    /**
     * 编码一个向量
     *
     * @param vec   (dim,) 输入向量
     * @return      (n_subvectors,) 每段的码本索引
     */
    std::vector<uint8_t> encode(const float* vec) const;

    /**
     * 编码一批向量
     *
     * @param vectors  (n, dim) 平坦存储
     * @param n        数量
     * @return          (n, n_subvectors) codes[i][j] = 第 i 个向量第 j 段
     */
    std::vector<std::vector<uint8_t>> encode_batch(const float* vectors, size_t n) const;

    /**
     * 计算查询向量到一批编码的距离（查表法 ADC）
     *
     * @param query             (dim,) 查询向量
     * @param codes_batch       (n, n_subvectors) 编码
     * @return                  (n,) 近似距离
     */
    std::vector<float> compute_distances(
        const float* query,
        const std::vector<std::vector<uint8_t>>& codes_batch
    ) const;

    // ── 属性 ──
    size_t dim() const { return dim_; }
    size_t n_subvectors() const { return n_subvectors_; }
    size_t n_bits() const { return n_bits_; }
    size_t codebook_size() const { return 1u << n_bits_; }  // 256 for n_bits=8
    size_t d_sub() const { return dim_ / n_subvectors_; }    // 每段维度

private:
    size_t dim_ = 0;
    size_t n_subvectors_ = 0;
    size_t n_bits_ = 8;

    // 码本：n_subvectors * codebook_size * d_sub 平坦存储
    // codebooks_[m * codebook_size * d_sub + c * d_sub + d] = 第 m 段第 c 个码字第 d 维
    std::vector<float> codebooks_;
};

}  // namespace core
}  // namespace agentrag
