cmake_minimum_required(VERSION 3.24)

project(mini-kernel-lib VERSION 0.1.0 LANGUAGES CXX)

include(CheckLanguage)

option(MKLIB_BUILD_TESTS "Build test targets" ON)
option(MKLIB_BUILD_BENCHMARKS "Build benchmark targets" ON)
option(MKLIB_ENABLE_CUDA "Look for the CUDA toolkit" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MKLIB_HAS_CUDA_BACKEND 0)

if(MKLIB_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA compiler and toolkit found. Building the optional CUDA backend.")
    set(MKLIB_HAS_CUDA_BACKEND 1)
  else()
    message(STATUS "CUDA compiler not found. Building with host reference fallbacks only.")
  endif()
else()
  message(STATUS "CUDA backend disabled. Building with host reference fallbacks only.")
endif()

add_subdirectory(src)

if(MKLIB_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(MKLIB_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
