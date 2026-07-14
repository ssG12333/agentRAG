# 头文件
set(CORE_HEADERS
    include/vector_types.h
)

# 源文件 (Phase 2 逐步添加)
set(CORE_SOURCES
    src/hello.cpp
)

# pybind 模块
pybind11_add_module(agentrag_core
    pybind/module.cpp
    ${CORE_SOURCES}
)

target_include_directories(agentrag_core PRIVATE include)
target_compile_features(agentrag_core PRIVATE cxx_std_17)

# Windows: 避免 min/max 宏冲突
if(MSVC)
    target_compile_definitions(agentrag_core PRIVATE NOMINMAX)
endif()
