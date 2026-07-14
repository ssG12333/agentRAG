# C++ 加速模块构建配置
# 源文件列表
set(CORE_SOURCES
    src/hello.cpp
    src/kmeans.cpp
    src/ivf_index.cpp
    src/product_quantizer.cpp
    src/bm25_index.cpp
)

# 头文件（仅用于 IDE 代码提示）
set(CORE_HEADERS
    include/vector_types.h
    include/kmeans.h
    include/ivf_index.h
    include/product_quantizer.h
    include/bm25_index.h
)

# pybind11 模块
pybind11_add_module(agentrag_core
    pybind/module.cpp
    ${CORE_SOURCES}
)

target_include_directories(agentrag_core PRIVATE include)
target_compile_features(agentrag_core PRIVATE cxx_std_17)

# Windows: 避免 windows.h 的 min/max 宏污染
if(MSVC)
    target_compile_definitions(agentrag_core PRIVATE NOMINMAX)
    # 优化选项：/O2 最大优化（Release 模式）
    target_compile_options(agentrag_core PRIVATE $<$<CONFIG:Release>:/O2>)
endif()
