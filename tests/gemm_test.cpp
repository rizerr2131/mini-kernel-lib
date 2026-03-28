#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "mklib/mklib.h"

namespace {

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

float MakeValue(int64_t row, int64_t col, int seed) {
  const int value = (seed + static_cast<int>(row * 7 + col * 3)) % 17;
  return static_cast<float>(value - 8) / 5.0f;
}

std::vector<float> MakeMatrix(const StorageShape& shape, int seed) {
  std::vector<float> matrix(StorageSize(shape), -99.0f);
  for (int64_t row = 0; row < shape.rows; ++row) {
    for (int64_t col = 0; col < shape.cols; ++col) {
      matrix[static_cast<size_t>(row * shape.ld + col)] = MakeValue(row, col, seed);
    }
  }
  return matrix;
}

float LoadA(const std::vector<float>& a, const mklibGemmDesc_t& desc, int64_t row, int64_t depth) {
  if (desc.trans_a == MKLIB_OP_N) {
    return a[static_cast<size_t>(row * desc.lda + depth)];
  }
  return a[static_cast<size_t>(depth * desc.lda + row)];
}

float LoadB(const std::vector<float>& b, const mklibGemmDesc_t& desc, int64_t depth, int64_t col) {
  if (desc.trans_b == MKLIB_OP_N) {
    return b[static_cast<size_t>(depth * desc.ldb + col)];
  }
  return b[static_cast<size_t>(col * desc.ldb + depth)];
}

double ReferenceValue(
    const mklibGemmDesc_t& desc,
    const std::vector<float>& a,
    const std::vector<float>& b,
    int64_t row,
    int64_t col) {
  double sum = 0.0;
  for (int64_t depth = 0; depth < desc.k; ++depth) {
    sum += static_cast<double>(LoadA(a, desc, row, depth)) *
           static_cast<double>(LoadB(b, desc, depth, col));
  }
  return sum;
}

void CheckClose(float actual, double expected) {
  const double diff = std::abs(static_cast<double>(actual) - expected);
  assert(diff <= 1e-5);
}

void RunSupportedCase(mklibHandle_t handle, const mklibGemmDesc_t& desc) {
  const auto a = MakeMatrix(ShapeForA(desc), 3);
  const auto b = MakeMatrix(ShapeForB(desc), 11);
  std::vector<float> c(StorageSize(ShapeForC(desc)), -7.0f);

  size_t workspace_bytes = 1;
  assert(mklibGetGemmWorkspaceSize(handle, &desc, &workspace_bytes) == MKLIB_STATUS_SUCCESS);
  assert(workspace_bytes == 0);
  assert(mklibGemm(handle, &desc, a.data(), b.data(), c.data(), nullptr, workspace_bytes) ==
         MKLIB_STATUS_SUCCESS);

  for (int64_t row = 0; row < desc.m; ++row) {
    for (int64_t col = 0; col < desc.n; ++col) {
      const float actual = c[static_cast<size_t>(row * desc.ldc + col)];
      const double expected = ReferenceValue(desc, a, b, row, col);
      CheckClose(actual, expected);
    }
  }
}

void CheckZeroSizedGemm(mklibHandle_t handle) {
  const mklibGemmDesc_t desc = {
      .a_type = MKLIB_DATA_TYPE_FLOAT32,
      .b_type = MKLIB_DATA_TYPE_FLOAT32,
      .c_type = MKLIB_DATA_TYPE_FLOAT32,
      .compute_type = MKLIB_DATA_TYPE_FLOAT32,
      .trans_a = MKLIB_OP_N,
      .trans_b = MKLIB_OP_N,
      .m = 0,
      .n = 8,
      .k = 16,
      .lda = 16,
      .ldb = 8,
      .ldc = 8,
  };

  size_t workspace_bytes = 0;
  assert(mklibGetGemmWorkspaceSize(handle, &desc, &workspace_bytes) == MKLIB_STATUS_SUCCESS);
  assert(mklibGemm(handle, &desc, nullptr, nullptr, nullptr, nullptr, workspace_bytes) ==
         MKLIB_STATUS_SUCCESS);
}

void CheckUnsupportedDtype(mklibHandle_t handle) {
  const mklibGemmDesc_t desc = {
      .a_type = MKLIB_DATA_TYPE_FLOAT16,
      .b_type = MKLIB_DATA_TYPE_FLOAT16,
      .c_type = MKLIB_DATA_TYPE_FLOAT16,
      .compute_type = MKLIB_DATA_TYPE_FLOAT16,
      .trans_a = MKLIB_OP_N,
      .trans_b = MKLIB_OP_N,
      .m = 4,
      .n = 4,
      .k = 4,
      .lda = 4,
      .ldb = 4,
      .ldc = 4,
  };

  size_t workspace_bytes = 0;
  assert(mklibGetGemmWorkspaceSize(handle, &desc, &workspace_bytes) == MKLIB_STATUS_NOT_SUPPORTED);
}

void CheckInvalidLeadingDimension(mklibHandle_t handle) {
  mklibGemmDesc_t desc = {
      .a_type = MKLIB_DATA_TYPE_FLOAT32,
      .b_type = MKLIB_DATA_TYPE_FLOAT32,
      .c_type = MKLIB_DATA_TYPE_FLOAT32,
      .compute_type = MKLIB_DATA_TYPE_FLOAT32,
      .trans_a = MKLIB_OP_N,
      .trans_b = MKLIB_OP_N,
      .m = 4,
      .n = 4,
      .k = 4,
      .lda = 3,
      .ldb = 4,
      .ldc = 4,
  };

  size_t workspace_bytes = 0;
  assert(mklibGetGemmWorkspaceSize(handle, &desc, &workspace_bytes) ==
         MKLIB_STATUS_INVALID_ARGUMENT);
}

}  // namespace

int main() {
  mklibHandle_t handle = nullptr;
  assert(mklibCreate(&handle) == MKLIB_STATUS_SUCCESS);

  RunSupportedCase(
      handle,
      {
          .a_type = MKLIB_DATA_TYPE_FLOAT32,
          .b_type = MKLIB_DATA_TYPE_FLOAT32,
          .c_type = MKLIB_DATA_TYPE_FLOAT32,
          .compute_type = MKLIB_DATA_TYPE_FLOAT32,
          .trans_a = MKLIB_OP_N,
          .trans_b = MKLIB_OP_N,
          .m = 2,
          .n = 4,
          .k = 3,
          .lda = 5,
          .ldb = 6,
          .ldc = 7,
      });
  RunSupportedCase(
      handle,
      {
          .a_type = MKLIB_DATA_TYPE_FLOAT32,
          .b_type = MKLIB_DATA_TYPE_FLOAT32,
          .c_type = MKLIB_DATA_TYPE_FLOAT32,
          .compute_type = MKLIB_DATA_TYPE_FLOAT32,
          .trans_a = MKLIB_OP_T,
          .trans_b = MKLIB_OP_N,
          .m = 3,
          .n = 2,
          .k = 4,
          .lda = 5,
          .ldb = 4,
          .ldc = 4,
      });
  RunSupportedCase(
      handle,
      {
          .a_type = MKLIB_DATA_TYPE_FLOAT32,
          .b_type = MKLIB_DATA_TYPE_FLOAT32,
          .c_type = MKLIB_DATA_TYPE_FLOAT32,
          .compute_type = MKLIB_DATA_TYPE_FLOAT32,
          .trans_a = MKLIB_OP_N,
          .trans_b = MKLIB_OP_T,
          .m = 3,
          .n = 2,
          .k = 4,
          .lda = 6,
          .ldb = 5,
          .ldc = 3,
      });
  RunSupportedCase(
      handle,
      {
          .a_type = MKLIB_DATA_TYPE_FLOAT32,
          .b_type = MKLIB_DATA_TYPE_FLOAT32,
          .c_type = MKLIB_DATA_TYPE_FLOAT32,
          .compute_type = MKLIB_DATA_TYPE_FLOAT32,
          .trans_a = MKLIB_OP_T,
          .trans_b = MKLIB_OP_T,
          .m = 4,
          .n = 3,
          .k = 2,
          .lda = 6,
          .ldb = 4,
          .ldc = 5,
      });

  CheckZeroSizedGemm(handle);
  CheckUnsupportedDtype(handle);
  CheckInvalidLeadingDimension(handle);

  assert(mklibDestroy(handle) == MKLIB_STATUS_SUCCESS);
  return 0;
}
