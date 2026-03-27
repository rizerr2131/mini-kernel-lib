#include <chrono>
#include <cstdlib>
#include <iostream>

#include "mklib/mklib.h"

int main(int argc, char** argv) {
  size_t iterations = 100000;
  if (argc > 1) {
    iterations = std::strtoull(argv[1], nullptr, 10);
    if (iterations == 0) {
      iterations = 1;
    }
  }

  mklibHandle_t handle = nullptr;
  const mklibStatus_t create_status = mklibCreate(&handle);
  if (create_status != MKLIB_STATUS_SUCCESS) {
    std::cerr << "mklibCreate failed: " << mklibGetStatusString(create_status) << '\n';
    return 1;
  }

  const mklibGemmDesc_t desc = {
      .a_type = MKLIB_DATA_TYPE_FLOAT32,
      .b_type = MKLIB_DATA_TYPE_FLOAT32,
      .c_type = MKLIB_DATA_TYPE_FLOAT32,
      .compute_type = MKLIB_DATA_TYPE_FLOAT32,
      .trans_a = MKLIB_OP_N,
      .trans_b = MKLIB_OP_N,
      .m = 512,
      .n = 512,
      .k = 512,
      .lda = 512,
      .ldb = 512,
      .ldc = 512,
  };

  size_t workspace_bytes = 0;
  const auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    const mklibStatus_t status = mklibGetGemmWorkspaceSize(handle, &desc, &workspace_bytes);
    if (status != MKLIB_STATUS_SUCCESS) {
      std::cerr << "mklibGetGemmWorkspaceSize failed: " << mklibGetStatusString(status) << '\n';
      mklibDestroy(handle);
      return 1;
    }
  }
  const auto stop = std::chrono::steady_clock::now();

  const double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  const double ns_per_call = total_ns / static_cast<double>(iterations);

  float a[16] = {};
  float b[16] = {};
  float c[16] = {};
  const mklibStatus_t gemm_status = mklibGemm(handle, &desc, a, b, c, nullptr, 0);

  std::cout << "iterations: " << iterations << '\n';
  std::cout << "workspace_bytes: " << workspace_bytes << '\n';
  std::cout << "workspace_query_ns_per_call: " << ns_per_call << '\n';
  std::cout << "gemm_status: " << mklibGetStatusString(gemm_status) << '\n';
#if MKLIB_HAS_CUDA_BACKEND
  std::cout << "cuda_toolkit: found\n";
#else
  std::cout << "cuda_toolkit: not found\n";
#endif

  mklibDestroy(handle);
  return 0;
}
