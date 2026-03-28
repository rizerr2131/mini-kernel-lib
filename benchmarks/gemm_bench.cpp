#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "mklib/mklib.h"

namespace {

int64_t ParsePositiveInt64(const char* value, int64_t fallback) {
  const long long parsed = std::strtoll(value, nullptr, 10);
  if (parsed <= 0) {
    return fallback;
  }
  return static_cast<int64_t>(parsed);
}

size_t ParsePositiveSizeT(const char* value, size_t fallback) {
  const unsigned long long parsed = std::strtoull(value, nullptr, 10);
  if (parsed == 0) {
    return fallback;
  }
  return static_cast<size_t>(parsed);
}

std::vector<float> MakeMatrix(int64_t rows, int64_t cols, int64_t ld, int seed) {
  std::vector<float> matrix(static_cast<size_t>(rows * ld), 0.0f);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      const int value = (seed + static_cast<int>(row * 5 + col * 3)) % 19;
      matrix[static_cast<size_t>(row * ld + col)] = static_cast<float>(value - 9) / 7.0f;
    }
  }
  return matrix;
}

}  // namespace

int main(int argc, char** argv) {
  size_t iterations = 25;
  size_t warmup_iterations = 5;
  int64_t m = 256;
  int64_t n = 256;
  int64_t k = 256;
  bool autotune = false;

  if (argc > 1) {
    iterations = ParsePositiveSizeT(argv[1], iterations);
  }
  if (argc > 2) {
    m = ParsePositiveInt64(argv[2], m);
  }
  if (argc > 3) {
    n = ParsePositiveInt64(argv[3], n);
  }
  if (argc > 4) {
    k = ParsePositiveInt64(argv[4], k);
  }
  if (argc > 5) {
    warmup_iterations = ParsePositiveSizeT(argv[5], warmup_iterations);
  }
  if (argc > 6) {
    autotune = ParsePositiveInt64(argv[6], 0) != 0;
  }

  mklibHandle_t handle = nullptr;
  const mklibStatus_t create_status = mklibCreate(&handle);
  if (create_status != MKLIB_STATUS_SUCCESS) {
    std::cerr << "mklibCreate failed: " << mklibGetStatusString(create_status) << '\n';
    return 1;
  }
  if (autotune) {
    const mklibStatus_t autotune_status = mklibSetAutotuneMode(handle, MKLIB_AUTOTUNE_ON);
    if (autotune_status != MKLIB_STATUS_SUCCESS) {
      std::cerr << "mklibSetAutotuneMode failed: "
                << mklibGetStatusString(autotune_status) << '\n';
      mklibDestroy(handle);
      return 1;
    }
  }

  const mklibGemmDesc_t desc = {
      .a_type = MKLIB_DATA_TYPE_FLOAT32,
      .b_type = MKLIB_DATA_TYPE_FLOAT32,
      .c_type = MKLIB_DATA_TYPE_FLOAT32,
      .compute_type = MKLIB_DATA_TYPE_FLOAT32,
      .trans_a = MKLIB_OP_N,
      .trans_b = MKLIB_OP_N,
      .m = m,
      .n = n,
      .k = k,
      .lda = k,
      .ldb = n,
      .ldc = n,
      .epilogue = MKLIB_POINTWISE_MODE_IDENTITY,
  };

  size_t workspace_bytes = 0;
  const mklibStatus_t workspace_status = mklibGetGemmWorkspaceSize(handle, &desc, &workspace_bytes);
  if (workspace_status != MKLIB_STATUS_SUCCESS) {
    std::cerr << "mklibGetGemmWorkspaceSize failed: "
              << mklibGetStatusString(workspace_status) << '\n';
    mklibDestroy(handle);
    return 1;
  }

  auto a = MakeMatrix(desc.m, desc.k, desc.lda, 3);
  auto b = MakeMatrix(desc.k, desc.n, desc.ldb, 11);
  std::vector<float> c(static_cast<size_t>(desc.m * desc.ldc), 0.0f);
  std::vector<unsigned char> workspace(workspace_bytes, 0);
  void* workspace_ptr = workspace.empty() ? nullptr : workspace.data();

  for (size_t i = 0; i < warmup_iterations; ++i) {
    const mklibStatus_t status =
        mklibGemm(handle, &desc, a.data(), b.data(), c.data(), workspace_ptr, workspace_bytes);
    if (status != MKLIB_STATUS_SUCCESS) {
      std::cerr << "warmup mklibGemm failed: " << mklibGetStatusString(status) << '\n';
      mklibDestroy(handle);
      return 1;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  mklibStatus_t gemm_status = MKLIB_STATUS_SUCCESS;
  for (size_t i = 0; i < iterations; ++i) {
    gemm_status =
        mklibGemm(handle, &desc, a.data(), b.data(), c.data(), workspace_ptr, workspace_bytes);
    if (gemm_status != MKLIB_STATUS_SUCCESS) {
      std::cerr << "mklibGemm failed: " << mklibGetStatusString(gemm_status) << '\n';
      mklibDestroy(handle);
      return 1;
    }
  }
  const auto stop = std::chrono::steady_clock::now();

  const double total_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  const double ns_per_call = total_ns / static_cast<double>(iterations);
  const double total_seconds = total_ns / 1e9;
  const double gflops =
      (2.0 * static_cast<double>(desc.m) * static_cast<double>(desc.n) *
       static_cast<double>(desc.k) * static_cast<double>(iterations)) /
      (total_seconds * 1e9);
  const double checksum = std::accumulate(c.begin(), c.end(), 0.0);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "iterations: " << iterations << '\n';
  std::cout << "warmup_iterations: " << warmup_iterations << '\n';
  std::cout << "m: " << desc.m << '\n';
  std::cout << "n: " << desc.n << '\n';
  std::cout << "k: " << desc.k << '\n';
  std::cout << "workspace_bytes: " << workspace_bytes << '\n';
  std::cout << "autotune_mode: " << (autotune ? "on" : "off") << '\n';
  std::cout << "pointwise_mode: identity\n";
  std::cout << "gemm_ns_per_call: " << ns_per_call << '\n';
  std::cout << "gemm_gflops: " << gflops << '\n';
  std::cout << "checksum: " << checksum << '\n';
  std::cout << "gemm_status: " << mklibGetStatusString(gemm_status) << '\n';
#if MKLIB_HAS_CUDA_BACKEND
  std::cout << "cuda_toolkit: found\n";
#else
  std::cout << "cuda_toolkit: not found\n";
#endif

  mklibDestroy(handle);
  return 0;
}
