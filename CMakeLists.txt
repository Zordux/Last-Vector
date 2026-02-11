cmake_minimum_required(VERSION 3.16)
project(last_vector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LASTVECTOR_WITH_RAYLIB "Build rendered client with raylib" ON)
option(LASTVECTOR_BUILD_PYTHON "Build pybind11 Python extension" ON)

add_library(lastvector_core
    cpp/src/sim.cpp
    cpp/src/observation.cpp
    cpp/src/collision.cpp
    cpp/src/upgrades.cpp
)
target_include_directories(lastvector_core PUBLIC cpp/include)
target_compile_features(lastvector_core PUBLIC cxx_std_20)
target_compile_options(lastvector_core PRIVATE -Wall -Wextra -Wpedantic)

if(LASTVECTOR_WITH_RAYLIB)
    find_package(raylib REQUIRED)
endif()

add_executable(last_vector cpp/src/main.cpp)
target_link_libraries(last_vector PRIVATE lastvector_core)
target_compile_options(last_vector PRIVATE -Wall -Wextra -Wpedantic)

if(LASTVECTOR_WITH_RAYLIB)
    target_compile_definitions(last_vector PRIVATE LASTVECTOR_WITH_RAYLIB=1)
    target_link_libraries(last_vector PRIVATE raylib)
endif()

if(LASTVECTOR_BUILD_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.6
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(last_vector_core_module cpp/src/python_bindings.cpp)
    target_link_libraries(last_vector_core_module PRIVATE lastvector_core)
    target_include_directories(last_vector_core_module PRIVATE cpp/include)
    target_compile_options(last_vector_core_module PRIVATE -Wall -Wextra -Wpedantic)

    set_target_properties(last_vector_core_module PROPERTIES
        OUTPUT_NAME "last_vector_core"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/last_vector_env/native"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/last_vector_env/native"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/last_vector_env/native"
    )
endif()
