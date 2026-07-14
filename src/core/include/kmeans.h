/**
 * ============================================================================
 * K-Means 聚类 —— 向量量化的基础算法
 * ============================================================================
 *
 * 用途：将 n 个 d 维向量聚为 k 个类，得到 k 个聚类中心（centroids）。
 * 这是 IVF 和 PQ 的共同前置依赖。
 *
 * 算法（Lloyd 迭代）：
 *   1. 随机选 k 个点作为初始中心
 *   2. 分配：每个点归入最近的中心
 *   3. 更新：每个类的中心 = 类内所有点的均值
 *   4. 重复 2-3 直到收敛或达到最大迭代次数
 *
 * 复杂度：
 *   - 每次迭代 O(n * k * d)
 *   - n_iters 通常在 10-30 次收敛
 *   - 总体 O(n * k * d * n_iters)
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace agentrag {
namespace core {

/**
 * K-Means 聚类结果
 *
 * centroids:     (k, dim) 平坦存储，centroids[i*dim : (i+1)*dim] 为第 i 个中心
 * assignments:   (n,) 每个向量所属的聚类编号
 * n_iters:       实际迭代次数
 */
struct KMeansResult {
    std::vector<float> centroids;   // k * dim
    std::vector<int32_t> assignments; // n
    int32_t n_iters;
};

/**
 * K-Means 聚类
 *
 * @param vectors      (n, dim) 平坦存储的向量数据
 * @param n            向量数量
 * @param dim          向量维度
 * @param k            聚类数（k << n 通常取 sqrt(n) 或 256~1024）
 * @param max_iters    最大迭代次数（默认 25）
 * @param seed         随机种子（0 表示用系统时钟）
 * @return KMeansResult
 */
KMeansResult kmeans(
    const float* vectors,
    size_t n,
    size_t dim,
    size_t k,
    int32_t max_iters = 25,
    uint32_t seed = 42
);

}  // namespace core
}  // namespace agentrag
