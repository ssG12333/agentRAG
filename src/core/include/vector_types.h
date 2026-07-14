#pragma once

#include <cstddef>
#include <cstdint>

namespace agentrag {
namespace core {

// 向量数据类型
using float32 = float;

// 单条向量
struct Vector {
    const float32* data;
    size_t dim;
};

// 向量批次
struct VectorBatch {
    const float32* data;  // 扁平存储: n * dim
    size_t n;
    size_t dim;
};

// 搜索结果
struct SearchResult {
    int32_t id;
    float score;
};

// PQ 码本条目
struct PQCode {
    uint8_t* codes;       // 每段一个 uint8
    size_t n_subvectors;
};

}  // namespace core
}  // namespace agentrag
