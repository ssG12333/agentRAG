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

# Windows: UTF-8 源文件编码 + 避免 min/max 宏污染
if(MSVC)
    target_compile_definitions(agentrag_core PRIVATE NOMINMAX)
    target_compile_options(agentrag_core PRIVATE /utf-8 $<$<CONFIG:Release>:/O2>)
endif()

# 编译后自动复制 .pyd 到项目根目录（无需手动 copy）
add_custom_command(TARGET agentrag_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:agentrag_core>"
        "${CMAKE_SOURCE_DIR}/$<TARGET_FILE_NAME:agentrag_core>"
    COMMENT "Copying agentrag_core.pyd to project root"
)
