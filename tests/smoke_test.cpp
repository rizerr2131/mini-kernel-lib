#include <array>
#include <cassert>
#include <cstdint>

#include "mklib/mklib.h"

int main() {
  mklibHandle_t handle = nullptr;
  assert(mklibCreate(&handle) == MKLIB_STATUS_SUCCESS);

  void* stream = reinterpret_cast<void*>(0x1234);
  assert(mklibSetStream(handle, stream) == MKLIB_STATUS_SUCCESS);

  void* stream_out = nullptr;
  assert(mklibGetStream(handle, &stream_out) == MKLIB_STATUS_SUCCESS);
  assert(stream_out == stream);

  mklibTensorDesc_t desc = nullptr;
  assert(mklibCreateTensorDesc(&desc) == MKLIB_STATUS_SUCCESS);

  const std::array<int64_t, 2> sizes = {2, 3};
  const std::array<int64_t, 2> strides = {3, 1};
  assert(mklibSetTensorDesc(
             desc,
             MKLIB_DATA_TYPE_FLOAT32,
             static_cast<int>(sizes.size()),
             sizes.data(),
             strides.data()) == MKLIB_STATUS_SUCCESS);

  int rank = 0;
  assert(mklibGetTensorRank(desc, &rank) == MKLIB_STATUS_SUCCESS);
  assert(rank == 2);

  mklibDataType_t dtype = MKLIB_DATA_TYPE_INVALID;
  assert(mklibGetTensorDataType(desc, &dtype) == MKLIB_STATUS_SUCCESS);
  assert(dtype == MKLIB_DATA_TYPE_FLOAT32);

  std::array<int64_t, 2> size_out = {};
  std::array<int64_t, 2> stride_out = {};
  assert(mklibGetTensorSizes(desc, size_out.size(), size_out.data()) == MKLIB_STATUS_SUCCESS);
  assert(mklibGetTensorStrides(desc, stride_out.size(), stride_out.data()) == MKLIB_STATUS_SUCCESS);
  assert(size_out == sizes);
  assert(stride_out == strides);

  const mklibGemmDesc_t gemm_desc = {
      .a_type = MKLIB_DATA_TYPE_FLOAT32,
      .b_type = MKLIB_DATA_TYPE_FLOAT32,
      .c_type = MKLIB_DATA_TYPE_FLOAT32,
      .compute_type = MKLIB_DATA_TYPE_FLOAT32,
      .trans_a = MKLIB_OP_N,
      .trans_b = MKLIB_OP_N,
      .m = 16,
      .n = 16,
      .k = 16,
      .lda = 16,
      .ldb = 16,
      .ldc = 16,
  };

  size_t workspace_bytes = 1;
  assert(mklibGetGemmWorkspaceSize(handle, &gemm_desc, &workspace_bytes) == MKLIB_STATUS_SUCCESS);
  assert(workspace_bytes == 0);

  float a[16 * 16] = {};
  float b[16 * 16] = {};
  float c[16 * 16] = {};
  assert(mklibGemm(handle, &gemm_desc, a, b, c, nullptr, 0) == MKLIB_STATUS_NOT_SUPPORTED);

  assert(mklibDestroyTensorDesc(desc) == MKLIB_STATUS_SUCCESS);
  assert(mklibDestroy(handle) == MKLIB_STATUS_SUCCESS);
  return 0;
}
