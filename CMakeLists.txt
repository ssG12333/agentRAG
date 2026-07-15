cmake_minimum_required(VERSION 3.18)
project(agentrag_core VERSION 0.1.0 LANGUAGES CXX)

# C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# pybind11 3.x / CMake 3.27+ 推荐使用现代 FindPython，
# 避免新版 CMake 调用已移除的 FindPythonInterp/FindPythonLibs。
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11：优先用 conda/pip 安装的，找不到再 FetchContent 下载
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found locally, downloading via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# 子目录
add_subdirectory(src/core)
