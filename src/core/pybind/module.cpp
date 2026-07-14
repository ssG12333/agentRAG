#include <pybind11/pybind11.h>

namespace py = pybind11;

// 声明 C++ 函数
namespace agentrag {
namespace core {
    int32_t add(int32_t a, int32_t b);
}
}

PYBIND11_MODULE(agentrag_core, m) {
    m.doc() = "agentRAG C++ 加速模块 —— 向量索引 / BM25 / 分词器";

    m.def("add", &agentrag::core::add,
          py::arg("a"), py::arg("b"),
          "测试函数: 两整数相加");

    // Phase 2 逐步添加:
    //   py::class_<IVFPQIndex>
    //   py::class_<BM25Index>
    //   py::class_<Tokenizer>
}
