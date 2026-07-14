/**
 * K-Means 聚类实现
 *
 * 参考：
 *   - Lloyd, S. (1982). "Least squares quantization in PCM"
 *   - scikit-learn KMeans 实现
 */

#include "kmeans.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

namespace agentrag {
namespace core {

// ── 辅助函数 ──────────────────────────────────────────

/** 欧氏距离平方（避免 sqrt，比较时等价） */
static float squared_l2(const float* a, const float* b, size_t dim) {
    float dist = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

/** 找最近中心 */
static int32_t nearest_centroid(const float* vec, const float* centroids, size_t k, size_t dim) {
    int32_t best = 0;
    float best_dist = squared_l2(vec, centroids, dim);
    for (size_t c = 1; c < k; ++c) {
        float d = squared_l2(vec, centroids + c * dim, dim);
        if (d < best_dist) {
            best_dist = d;
            best = static_cast<int32_t>(c);
        }
    }
    return best;
}

// ── 主函数 ────────────────────────────────────────────

KMeansResult kmeans(
    const float* vectors,
    size_t n,
    size_t dim,
    size_t k,
    int32_t max_iters,
    uint32_t seed)
{
    KMeansResult result;
    result.centroids.resize(k * dim, 0.0f);
    result.assignments.resize(n, 0);
    result.n_iters = 0;

    if (n == 0 || k == 0 || k > n) {
        return result;
    }

    // ── 1. 初始化：随机选 k 个点作为初始中心 ──
    std::mt19937 rng(seed ? seed : 42);
    std::uniform_int_distribution<size_t> dist(0, n - 1);

    // 用 Fisher-Yates 的部分 shuffle 选 k 个不重复的初始中心
    // 简单实现：随机选 k 个索引（可能有重复，但概率极小）
    for (size_t c = 0; c < k; ++c) {
        size_t idx = dist(rng);
        std::memcpy(
            result.centroids.data() + c * dim,
            vectors + idx * dim,
            dim * sizeof(float)
        );
    }

    // ── 2. 迭代 ──
    // 分配缓冲区: 每个类的向量和计数
    std::vector<float> new_centroids(k * dim, 0.0f);
    std::vector<int32_t> counts(k, 0);

    for (int32_t iter = 0; iter < max_iters; ++iter) {
        result.n_iters = iter + 1;

        // 2a. 分配步骤：每个点找最近中心
        bool changed = false;
        for (size_t i = 0; i < n; ++i) {
            int32_t c_new = nearest_centroid(
                vectors + i * dim,
                result.centroids.data(),
                k, dim
            );
            if (c_new != result.assignments[i]) {
                changed = true;
                result.assignments[i] = c_new;
            }
        }

        // 收敛检查：分配不再变化
        if (!changed && iter > 0) {
            break;
        }

        // 2b. 更新步骤：重新计算每个类的中心
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < n; ++i) {
            int32_t c = result.assignments[i];
            float* acc = new_centroids.data() + c * dim;
            const float* v = vectors + i * dim;
            for (size_t d = 0; d < dim; ++d) {
                acc[d] += v[d];
            }
            counts[c]++;
        }

        // 求均值 + 处理空类（保持旧中心）
        for (size_t c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                float inv = 1.0f / static_cast<float>(counts[c]);
                float* c_new = new_centroids.data() + c * dim;
                float* c_old = result.centroids.data() + c * dim;
                for (size_t d = 0; d < dim; ++d) {
                    c_old[d] = c_new[d] * inv;
                }
            }
            // 空类：中心保持不变（可能重新初始化为随机点更好，但这里简化）
        }
    }

    return result;
}

}  // namespace core
}  // namespace agentrag
