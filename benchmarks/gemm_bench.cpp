#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "mklib/mklib.h"

#if MKLIB_HAS_CUDA_BACKEND
#include <cuda_runtime_api.h>
#endif

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

struct StorageShape {
  int64_t rows;
  int64_t cols;
  int64_t ld;
};

StorageShape ShapeForA(const mklibGemmDesc_t& desc) {
  if (desc.trans_a == MKLIB_OP_N) {
    return {.rows = desc.m, .cols = desc.k, .ld = desc.lda};
  }
  return {.rows = desc.k, .cols = desc.m, .ld = desc.lda};
}

StorageShape ShapeForB(const mklibGemmDesc_t& desc) {
  if (desc.trans_b == MKLIB_OP_N) {
    return {.rows = desc.k, .cols = desc.n, .ld = desc.ldb};
  }
  return {.rows = desc.n, .cols = desc.k, .ld = desc.ldb};
}

StorageShape ShapeForC(const mklibGemmDesc_t& desc) {
  return {.rows = desc.m, .cols = desc.n, .ld = desc.ldc};
}

size_t StorageSize(const StorageShape& shape) {
  return static_cast<size_t>(shape.rows * shape.ld);
}

mklibTranspose_t ParseTranspose(const char* value) {
  return ParsePositiveInt64(value, 0) != 0 ? MKLIB_OP_T : MKLIB_OP_N;
}

#if MKLIB_HAS_CUDA_BACKEND
bool HasCudaDevice() {
  int device_count = 0;
  const cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return device_count > 0;
}
#endif

}  // namespace

int main(int argc, char** argv) {
  size_t iterations = 25;
  size_t warmup_iterations = 5;
  int64_t m = 256;
  int64_t n = 256;
  int64_t k = 256;
  mklibTranspose_t trans_a = MKLIB_OP_N;
  mklibTranspose_t trans_b = MKLIB_OP_N;
  bool autotune = false;
  bool device_buffers = false;

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
  if (argc > 7) {
    device_buffers = ParsePositiveInt64(argv[7], 0) != 0;
  }
  if (argc > 8) {
    trans_a = ParseTranspose(argv[8]);
  }
  if (argc > 9) {
    trans_b = ParseTranspose(argv[9]);
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
      .trans_a = trans_a,
      .trans_b = trans_b,
      .m = m,
      .n = n,
      .k = k,
      .lda = trans_a == MKLIB_OP_N ? k : m,
      .ldb = trans_b == MKLIB_OP_N ? n : k,
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

  auto a_shape = ShapeForA(desc);
  auto b_shape = ShapeForB(desc);
  auto c_shape = ShapeForC(desc);
  auto a = MakeMatrix(a_shape.rows, a_shape.cols, a_shape.ld, 3);
  auto b = MakeMatrix(b_shape.rows, b_shape.cols, b_shape.ld, 11);
  std::vector<float> c(StorageSize(c_shape), 0.0f);

  const void* a_ptr = a.data();
  const void* b_ptr = b.data();
  void* c_ptr = c.data();
  std::vector<unsigned char> host_workspace(workspace_bytes, 0);
  void* workspace_ptr = host_workspace.empty() ? nullptr : host_workspace.data();

#if MKLIB_HAS_CUDA_BACKEND
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;
  void* device_workspace = nullptr;
  if (device_buffers) {
    if (!HasCudaDevice()) {
      std::cerr << "device buffer mode requested but no CUDA device is available\n";
      mklibDestroy(handle);
      return 1;
    }

    if (cudaMalloc(reinterpret_cast<void**>(&device_a), a.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_b), b.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_c), c.size() * sizeof(float)) != cudaSuccess) {
      std::cerr << "cudaMalloc failed for benchmark buffers\n";
      if (device_c != nullptr) {
        cudaFree(device_c);
      }
      if (device_b != nullptr) {
        cudaFree(device_b);
      }
      if (device_a != nullptr) {
        cudaFree(device_a);
      }
      mklibDestroy(handle);
      return 1;
    }

    if (workspace_bytes > 0 && cudaMalloc(&device_workspace, workspace_bytes) != cudaSuccess) {
      std::cerr << "cudaMalloc failed for workspace buffer\n";
      cudaFree(device_c);
      cudaFree(device_b);
      cudaFree(device_a);
      mklibDestroy(handle);
      return 1;
    }

    if (cudaMemcpy(device_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(device_c, 0, c.size() * sizeof(float)) != cudaSuccess) {
      std::cerr << "CUDA host-to-device copy failed\n";
      if (device_workspace != nullptr) {
        cudaFree(device_workspace);
      }
      cudaFree(device_c);
      cudaFree(device_b);
      cudaFree(device_a);
      mklibDestroy(handle);
      return 1;
    }

    a_ptr = device_a;
    b_ptr = device_b;
    c_ptr = device_c;
    workspace_ptr = device_workspace;
  }
#else
  if (device_buffers) {
    std::cerr << "device buffer mode requested but this build has no CUDA backend\n";
    mklibDestroy(handle);
    return 1;
  }
#endif

  for (size_t i = 0; i < warmup_iterations; ++i) {
    const mklibStatus_t status =
        mklibGemm(handle, &desc, a_ptr, b_ptr, c_ptr, workspace_ptr, workspace_bytes);
    if (status != MKLIB_STATUS_SUCCESS) {
      std::cerr << "warmup mklibGemm failed: " << mklibGetStatusString(status) << '\n';
#if MKLIB_HAS_CUDA_BACKEND
      if (device_workspace != nullptr) {
        cudaFree(device_workspace);
      }
      if (device_c != nullptr) {
        cudaFree(device_c);
      }
      if (device_b != nullptr) {
        cudaFree(device_b);
      }
      if (device_a != nullptr) {
        cudaFree(device_a);
      }
#endif
      mklibDestroy(handle);
      return 1;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  mklibStatus_t gemm_status = MKLIB_STATUS_SUCCESS;
  for (size_t i = 0; i < iterations; ++i) {
    gemm_status = mklibGemm(handle, &desc, a_ptr, b_ptr, c_ptr, workspace_ptr, workspace_bytes);
    if (gemm_status != MKLIB_STATUS_SUCCESS) {
      std::cerr << "mklibGemm failed: " << mklibGetStatusString(gemm_status) << '\n';
#if MKLIB_HAS_CUDA_BACKEND
      if (device_workspace != nullptr) {
        cudaFree(device_workspace);
      }
      if (device_c != nullptr) {
        cudaFree(device_c);
      }
      if (device_b != nullptr) {
        cudaFree(device_b);
      }
      if (device_a != nullptr) {
        cudaFree(device_a);
      }
#endif
      mklibDestroy(handle);
      return 1;
    }
  }
  const auto stop = std::chrono::steady_clock::now();

#if MKLIB_HAS_CUDA_BACKEND
  if (device_buffers) {
    if (cudaMemcpy(c.data(), device_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
      std::cerr << "CUDA device-to-host copy failed\n";
      if (device_workspace != nullptr) {
        cudaFree(device_workspace);
      }
      cudaFree(device_c);
      cudaFree(device_b);
      cudaFree(device_a);
      mklibDestroy(handle);
      return 1;
    }
  }
#endif

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
  std::cout << "trans_a: " << (desc.trans_a == MKLIB_OP_N ? "n" : "t") << '\n';
  std::cout << "trans_b: " << (desc.trans_b == MKLIB_OP_N ? "n" : "t") << '\n';
  std::cout << "workspace_bytes: " << workspace_bytes << '\n';
  std::cout << "autotune_mode: " << (autotune ? "on" : "off") << '\n';
  std::cout << "buffer_mode: " << (device_buffers ? "device" : "host") << '\n';
  std::cout << "pointwise_mode: identity\n";
  std::cout << "gemm_ns_per_call: " << ns_per_call << '\n';
  std::cout << "gemm_gflops: " << gflops << '\n';
  std::cout << "checksum: " << checksum << '\n';
  std::cout << "gemm_status: " << mklibGetStatusString(gemm_status) << '\n';
#if MKLIB_HAS_CUDA_BACKEND
  std::cout << "cuda_backend: built\n";
#else
  std::cout << "cuda_backend: not built\n";
#endif

#if MKLIB_HAS_CUDA_BACKEND
  if (device_workspace != nullptr) {
    cudaFree(device_workspace);
  }
  if (device_c != nullptr) {
    cudaFree(device_c);
  }
  if (device_b != nullptr) {
    cudaFree(device_b);
  }
  if (device_a != nullptr) {
    cudaFree(device_a);
  }
#endif

  mklibDestroy(handle);
  return 0;
}
