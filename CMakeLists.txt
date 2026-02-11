cmake_minimum_required(VERSION 3.16)
project(last_vector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LASTVECTOR_WITH_RAYLIB "Build rendered client with raylib" ON)
option(LASTVECTOR_BUILD_PYTHON_MODULE "Build pybind11 python extension" OFF)

add_library(lastvector_core
    cpp/src/sim.cpp
    cpp/src/observation.cpp
    cpp/src/upgrades.cpp
)

target_include_directories(lastvector_core PUBLIC cpp/include)

if(LASTVECTOR_WITH_RAYLIB)
    find_package(raylib REQUIRED)
endif()

add_executable(last_vector cpp/src/main.cpp)
target_link_libraries(last_vector PRIVATE lastvector_core)

if(LASTVECTOR_WITH_RAYLIB)
    target_compile_definitions(last_vector PRIVATE LASTVECTOR_WITH_RAYLIB=1)
    target_link_libraries(last_vector PRIVATE raylib)
endif()

if(LASTVECTOR_BUILD_PYTHON_MODULE)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(last_vector_core_module MODULE cpp/src/python_bindings.cpp)
    target_link_libraries(last_vector_core_module PRIVATE lastvector_core)
endif()
