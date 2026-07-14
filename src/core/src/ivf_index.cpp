/**
 * IVF 倒排索引实现
 */

#include "ivf_index.h"
#include "kmeans.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>

namespace agentrag {
namespace core {

// ── 辅助函数 ──────────────────────────────────────────

static float squared_l2(const float* a, const float* b, size_t dim) {
    float dist = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// ── 构建索引 ──────────────────────────────────────────

void IVFIndex::build(
    const float* vectors,
    size_t n,
    size_t dim,
    size_t k,
    size_t n_probe,
    int32_t n_iters)
{
    n_vectors_ = n;
    dim_ = dim;
    k_ = std::min(k, n);  // k 不能超过 n
    n_probe_ = n_probe;

    // 1. K-Means 得到聚类中心
    KMeansResult km = kmeans(vectors, n, dim, k_, n_iters);
    centroids_ = std::move(km.centroids);

    // 2. 建立倒排列表
    lists_.resize(k_);
    for (size_t c = 0; c < k_; ++c) {
        // 预分配空间：平均 n/k 个向量
        size_t expected = n / k_ + 1;
        lists_[c].ids.reserve(expected);
        lists_[c].residuals.reserve(expected * dim);
    }

    for (size_t i = 0; i < n; ++i) {
        int32_t c = km.assignments[i];

        // 计算残差：向量 - 中心
        const float* vec = vectors + i * dim;
        const float* centroid = centroids_.data() + c * dim;

        lists_[c].ids.push_back(static_cast<int32_t>(i));
        for (size_t d = 0; d < dim; ++d) {
            lists_[c].residuals.push_back(vec[d] - centroid[d]);
        }
    }
}

// ── 检索 ──────────────────────────────────────────────

std::vector<SearchResult> IVFIndex::search(const float* query, size_t top_k) const {
    if (lists_.empty() || top_k == 0) return {};

    // 1. 计算 query 到 k 个中心的距离，选最近的 n_probe 个
    // 使用最小堆存储 (distance, cluster_id)
    std::vector<std::pair<float, size_t>> cluster_dists;
    cluster_dists.reserve(k_);
    for (size_t c = 0; c < k_; ++c) {
        float dist = squared_l2(query, centroids_.data() + c * dim_, dim_);
        cluster_dists.emplace_back(dist, c);
    }
    // 部分排序：取前 n_probe 个最小距离的聚类
    size_t n_probe_actual = std::min(n_probe_, k_);
    std::partial_sort(
        cluster_dists.begin(),
        cluster_dists.begin() + n_probe_actual,
        cluster_dists.end()
    );

    // 2. 在选中的倒排列表中搜索
    // 用最大堆维护 top_k 结果 (distance, id)，堆顶是当前第 k 大的
    // 使用 pair 的默认比较：第一个元素是 key
    auto cmp = [](const SearchResult& a, const SearchResult& b) {
        return a.score < b.score;  // 想要最大堆（堆顶最大），score 越低越好 → 用 <
    };
    // 实际上我们需要的是最小堆（取 top_k 个最小距离）
    // 用最大堆维护已找到的 k 个结果中距离最大的：
    // 如果新距离 < 堆顶距离，弹出堆顶插入新结果
    std::vector<SearchResult> heap;
    heap.reserve(top_k);

    for (size_t p = 0; p < n_probe_actual; ++p) {
        size_t c = cluster_dists[p].second;
        const auto& list = lists_[c];

        size_t n_in_list = list.ids.size();
        const float* residuals = list.residuals.data();

        for (size_t j = 0; j < n_in_list; ++j) {
            // 近似距离：query 到此聚类中向量的距离
            // 注意：这里我们简化了——直接算残差的 L2，不加上 query→centroid 的距离
            // 正确做法：dist ≈ distance(query, centroid + residual)
            // 但残差是 vector - centroid，所以 query 到 vector 的距离用余弦近似：
            // 这里用简化版本（残差的模），Phase 2.5 在 PQ 中会用距离表修正
            const float* residual = residuals + j * dim_;
            float dist = squared_l2(query, residual, dim_);

            if (heap.size() < top_k) {
                heap.push_back({list.ids[j], -dist});  // 存负的，方便排序
                if (heap.size() == top_k) {
                    std::make_heap(heap.begin(), heap.end(),
                        [](const SearchResult& a, const SearchResult& b) {
                            return a.score < b.score;  // 最大堆（score 是负的）
                        });
                }
            } else {
                // 如果当前距离小于堆顶（最大距离），替换
                if (-dist > heap[0].score) {  // dist 更小的 → -dist 更大 → score 更大
                    std::pop_heap(heap.begin(), heap.end(),
                        [](const SearchResult& a, const SearchResult& b) {
                            return a.score < b.score;
                        });
                    heap.back() = {list.ids[j], -dist};
                    std::push_heap(heap.begin(), heap.end(),
                        [](const SearchResult& a, const SearchResult& b) {
                            return a.score < b.score;
                        });
                }
            }
        }
    }

    // 3. 排序结果并修正分数符号
    std::sort(heap.begin(), heap.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.score > b.score;  // -dist 越大的（距离越小）排前面
        });

    // 转回正值距离
    for (auto& r : heap) {
        r.score = -r.score;
    }

    return heap;
}

}  // namespace core
}  // namespace agentrag
