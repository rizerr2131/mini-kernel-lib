set(MKLIB_SOURCES
  api/conv.cc
  api/gemm.cc
  api/handle.cc
  api/reduction.cc
  api/status.cc
  api/tensor_desc.cc
  backend/cuda/cuda_conv.cc
  backend/cuda/cuda_gemm.cc
  backend/cuda/cuda_reduction.cc
  planner/dispatch_key.cc
  registry/kernel_registry.cc
)

add_library(mklib STATIC ${MKLIB_SOURCES})
add_library(mklib::mklib ALIAS mklib)

target_include_directories(mklib
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(mklib PUBLIC cxx_std_20)
target_compile_definitions(mklib PUBLIC MKLIB_HAS_CUDA_BACKEND=${MKLIB_HAS_CUDA_BACKEND})

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(mklib PRIVATE -Wall -Wextra -Wpedantic)
endif()
