/**
 * pybind11 Python 绑定 —— 将 C++ 类暴露给 Python
 *
 * 暴露的类：
 *   - KMeans:       kmeans() 函数
 *   - IVFIndex:     倒排索引构建/检索
 *   - ProductQuantizer: PQ 训练/编码/距离计算
 *   - BM25Index:    稀疏检索索引
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kmeans.h"
#include "ivf_index.h"
#include "ivf_pq_index.h"
#include "product_quantizer.h"
#include "bm25_index.h"

// hello.cpp 中的测试函数（无头文件，手动声明）
namespace agentrag {
namespace core {
    int32_t add(int32_t a, int32_t b);
}
}

namespace py = pybind11;
using namespace agentrag::core;

// ── 辅助：numpy 数组 → C++ 指针 ──────────────────────

// 检查 numpy 数组是否为 C-contiguous float32
static const float* as_float_ptr(const py::array_t<float>& arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2 && buf.ndim != 1) {
        throw std::runtime_error("Expected 1D or 2D numpy array");
    }
    return static_cast<const float*>(buf.ptr);
}

PYBIND11_MODULE(agentrag_core, m) {
    m.doc() = "agentRAG C++ 加速模块 —— K-Means / IVF / PQ / BM25";

    // ── 基础类型 ──
    py::class_<SearchResult>(m, "SearchResult")
        .def(py::init<>())
        .def_readwrite("id", &SearchResult::id)
        .def_readwrite("score", &SearchResult::score);

    // ── 测试函数 ──
    m.def("add", &agentrag::core::add,
          py::arg("a"), py::arg("b"),
          "测试函数: 两整数相加");

    // ── K-Means ──
    py::class_<KMeansResult>(m, "KMeansResult")
        .def(py::init<>())
        .def_readonly("centroids", &KMeansResult::centroids)
        .def_readonly("assignments", &KMeansResult::assignments)
        .def_readonly("n_iters", &KMeansResult::n_iters);

    m.def("kmeans", [](py::array_t<float> vectors,
                         size_t k, int32_t max_iters, uint32_t seed) {
            py::buffer_info buf = vectors.request();
            if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");
            return kmeans(
                static_cast<const float*>(buf.ptr),
                buf.shape[0], buf.shape[1],
                k, max_iters, seed);
        },
        py::arg("vectors"),
        py::arg("k"),
        py::arg("max_iters") = 25,
        py::arg("seed") = 42,
        "K-Means 聚类");

    // ── IVF 索引 ──
    py::class_<IVFIndex>(m, "IVFIndex")
        .def(py::init<>())
        .def("build", [](IVFIndex& self, py::array_t<float> vectors,
                          size_t k, size_t n_probe, int32_t n_iters) {
                py::buffer_info buf = vectors.request();
                if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");
                self.build(
                    static_cast<const float*>(buf.ptr),
                    buf.shape[0], buf.shape[1],
                    k, n_probe, n_iters);
            },
            py::arg("vectors"),
            py::arg("k") = 256,
            py::arg("n_probe") = 8,
            py::arg("n_iters") = 25,
            "构建 IVF 索引")
        .def("search", [](const IVFIndex& self, py::array_t<float> query, size_t top_k) {
                py::buffer_info buf = query.request();
                return self.search(static_cast<const float*>(buf.ptr), top_k);
            },
            py::arg("query"),
            py::arg("top_k") = 5,
            "检索 top_k 最近邻")
        .def("__len__", &IVFIndex::size)
        .def_property_readonly("dim", &IVFIndex::dim);

    // ── 残差 IVF-PQ 索引 ──
    py::class_<IVFPQIndex>(m, "IVFPQIndex")
        .def(py::init<>())
        .def("build", [](IVFPQIndex& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> vectors,
                          size_t n_clusters, size_t n_probe,
                          size_t n_subvectors, size_t n_bits, int32_t n_iters) {
                py::buffer_info buf = vectors.request();
                if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");
                self.build(
                    static_cast<const float*>(buf.ptr),
                    static_cast<size_t>(buf.shape[0]),
                    static_cast<size_t>(buf.shape[1]),
                    n_clusters, n_probe, n_subvectors, n_bits, n_iters);
            },
            py::arg("vectors"),
            py::arg("n_clusters") = 256,
            py::arg("n_probe") = 8,
            py::arg("n_subvectors") = 64,
            py::arg("n_bits") = 8,
            py::arg("n_iters") = 25,
            "构建残差 IVF-PQ 索引")
        .def("search", [](const IVFPQIndex& self,
                           py::array_t<float, py::array::c_style | py::array::forcecast> query,
                           size_t top_k) {
                py::buffer_info buf = query.request();
                if (buf.ndim != 1) throw std::runtime_error("Expected 1D query array");
                if (static_cast<size_t>(buf.shape[0]) != self.dim()) {
                    throw std::runtime_error("Query dimension does not match index");
                }
                return self.search(static_cast<const float*>(buf.ptr), top_k);
            },
            py::arg("query"), py::arg("top_k") = 5,
            "使用 ADC 检索 top_k 近邻")
        .def("save", &IVFPQIndex::save, py::arg("path"))
        .def("load", &IVFPQIndex::load, py::arg("path"))
        .def("__len__", &IVFPQIndex::size)
        .def_property_readonly("dim", &IVFPQIndex::dim)
        .def_property_readonly("num_clusters", &IVFPQIndex::num_clusters)
        .def_property_readonly("n_probe", &IVFPQIndex::n_probe)
        .def_property_readonly("n_subvectors", &IVFPQIndex::n_subvectors)
        .def_property_readonly("n_bits", &IVFPQIndex::n_bits)
        .def_property_readonly("codes_bytes", &IVFPQIndex::codes_bytes)
        .def_property_readonly("estimated_memory_bytes", &IVFPQIndex::estimated_memory_bytes);

    // ── 乘积量化 ──
    py::class_<ProductQuantizer>(m, "ProductQuantizer")
        .def(py::init<>())
        .def("train", [](ProductQuantizer& self, py::array_t<float> vectors,
                          size_t n_subvectors, size_t n_bits, int32_t n_iters) {
                py::buffer_info buf = vectors.request();
                if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");
                self.train(
                    static_cast<const float*>(buf.ptr),
                    buf.shape[0], buf.shape[1],
                    n_subvectors, n_bits, n_iters);
            },
            py::arg("vectors"),
            py::arg("n_subvectors") = 64,
            py::arg("n_bits") = 8,
            py::arg("n_iters") = 25,
            "训练乘积量化码本")
        .def("encode", [](const ProductQuantizer& self, py::array_t<float> vec) {
                py::buffer_info buf = vec.request();
                return self.encode(static_cast<const float*>(buf.ptr));
            },
            py::arg("vec"),
            "编码单个向量")
        .def("encode_batch", [](const ProductQuantizer& self, py::array_t<float> vectors) {
                py::buffer_info buf = vectors.request();
                if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");
                return self.encode_batch(
                    static_cast<const float*>(buf.ptr), buf.shape[0]);
            },
            py::arg("vectors"),
            "编码一批向量")
        .def("compute_distances", &ProductQuantizer::compute_distances,
            py::arg("query"),
            py::arg("codes_batch"),
            "查表计算近似距离（ADC）")
        .def_property_readonly("n_subvectors", &ProductQuantizer::n_subvectors)
        .def_property_readonly("n_bits", &ProductQuantizer::n_bits);

    // ── BM25 索引 ──
    py::class_<BM25Index>(m, "BM25Index")
        .def(py::init<>())
        .def("set_params", &BM25Index::set_params,
            py::arg("k1") = 1.5f, py::arg("b") = 0.75f,
            "设置 BM25 参数")
        .def("add_document", &BM25Index::add_document,
            py::arg("doc_id"), py::arg("tokens"),
            "添加文档到索引")
        .def("remove_document", &BM25Index::remove_document,
            py::arg("doc_id"),
            "从索引中删除文档")
        .def("search", &BM25Index::search,
            py::arg("query_tokens"), py::arg("top_k") = 10,
            "BM25 检索")
        .def_property_readonly("num_docs", &BM25Index::num_docs)
        .def_property_readonly("avg_doc_length", &BM25Index::avg_doc_length);
}
