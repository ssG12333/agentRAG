/**
 * ============================================================================
 * IVF (Inverted File) 倒排索引 —— 粗量化 + 倒排列表
 * ============================================================================
 *
 * 核心思想（类比图书馆）：
 *   先找书的类别（粗量化），再在该类别内逐本翻阅（细查）。
 *
 * 算法：
 *   构建：
 *     1. K-Means 将向量空间分为 k 个区域（Voronoi 划分）
 *     2. 每个向量分配到最近的聚类中心
 *     3. 存入对应中心的倒排列表（存残差 = 向量 - 中心）
 *
 *   检索：
 *     1. 计算 query 到 k 个中心的距离
 *     2. 选最近的 n_probe 个倒排列表
 *     3. 在这些列表中暴力搜索 top_k
 *
 * 复杂度：
 *   构建: O(n * k * d * n_iters) → K-Means 主导
 *   检索: O(k * d + n_probe * n/k * d) → 在选中的列表中遍历
 *
 * 参数建议：
 *   k:      sqrt(n) 到 4*sqrt(n)，通常取 256~1024
 *   n_probe: 检索精度 vs 速度权衡，通常 1~32，推荐 8
 *
 * 参考：
 *   Jegou et al. "Product Quantization for Nearest Neighbor Search" (2011)
 *   FAISS IndexIVF 实现
 */

#pragma once

#include "vector_types.h"

#include <cstddef>
#include <vector>

namespace agentrag {
namespace core {

/**
 * IVF 倒排索引（不含 PQ 压缩，存原始残差）
 *
 * Phase 2.3 先实现纯 IVF + 残差。
 * Phase 2.5 升级为 IVF + PQ（残差压缩存储）。
 */
class IVFIndex {
public:
    IVFIndex() = default;

    /**
     * 构建索引
     *
     * @param vectors   (n, dim) 平坦存储的训练向量
     * @param n         向量数量
     * @param dim       向量维度
     * @param k         聚类数（粗量化中心个数）
     * @param n_probe   检索时探测的倒排列表数
     * @param n_iters   K-Means 最大迭代次数
     */
    void build(
        const float* vectors,
        size_t n,
        size_t dim,
        size_t k = 256,
        size_t n_probe = 8,
        int32_t n_iters = 25
    );

    /**
     * 检索 top_k 最近邻
     *
     * @param query   (dim,) 查询向量
     * @param top_k   返回结果数
     * @return        [(vector_id, distance), ...] 按距离升序（最近在前）
     */
    std::vector<SearchResult> search(const float* query, size_t top_k) const;

    // ── 属性 ──
    size_t size() const { return n_vectors_; }
    size_t dim() const { return dim_; }
    size_t num_clusters() const { return k_; }
    size_t n_probe() const { return n_probe_; }

private:
    size_t n_vectors_ = 0;
    size_t dim_ = 0;
    size_t k_ = 0;
    size_t n_probe_ = 8;

    // 聚类中心：k * dim
    std::vector<float> centroids_;

    // 倒排列表：k 个列表，每个存 (原始向量ID, 残差向量)
    // 残差 = 原始向量 - 聚类中心，维度为 dim
    struct InvertedList {
        std::vector<int32_t> ids;         // 原始向量 ID
        std::vector<float> residuals;     // ids.size() * dim 平坦存储
    };
    std::vector<InvertedList> lists_;
};

}  // namespace core
}  // namespace agentrag
