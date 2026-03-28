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

  mklibAutotuneMode_t autotune_mode = MKLIB_AUTOTUNE_OFF;
  assert(mklibGetAutotuneMode(handle, &autotune_mode) == MKLIB_STATUS_SUCCESS);
  assert(autotune_mode == MKLIB_AUTOTUNE_OFF);
  assert(mklibSetAutotuneMode(handle, MKLIB_AUTOTUNE_ON) == MKLIB_STATUS_SUCCESS);
  assert(mklibGetAutotuneMode(handle, &autotune_mode) == MKLIB_STATUS_SUCCESS);
  assert(autotune_mode == MKLIB_AUTOTUNE_ON);
  assert(mklibClearAutotuneCache(handle) == MKLIB_STATUS_SUCCESS);
  assert(mklibSetAutotuneMode(handle, MKLIB_AUTOTUNE_OFF) == MKLIB_STATUS_SUCCESS);

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
      .epilogue = MKLIB_POINTWISE_MODE_IDENTITY,
  };

  size_t workspace_bytes = 1;
  assert(mklibGetGemmWorkspaceSize(handle, &gemm_desc, &workspace_bytes) == MKLIB_STATUS_SUCCESS);
  assert(workspace_bytes == 0);

  const mklibGemmDesc_t small_gemm_desc = {
      .a_type = MKLIB_DATA_TYPE_FLOAT32,
      .b_type = MKLIB_DATA_TYPE_FLOAT32,
      .c_type = MKLIB_DATA_TYPE_FLOAT32,
      .compute_type = MKLIB_DATA_TYPE_FLOAT32,
      .trans_a = MKLIB_OP_N,
      .trans_b = MKLIB_OP_N,
      .m = 2,
      .n = 2,
      .k = 3,
      .lda = 3,
      .ldb = 2,
      .ldc = 2,
      .epilogue = MKLIB_POINTWISE_MODE_IDENTITY,
  };
  const float a[2 * 3] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  const float b[3 * 2] = {
      7.0f, 8.0f,
      9.0f, 10.0f,
      11.0f, 12.0f,
  };
  float c[2 * 2] = {};
  assert(mklibGemm(handle, &small_gemm_desc, a, b, c, nullptr, 0) == MKLIB_STATUS_SUCCESS);
  assert(c[0] == 58.0f);
  assert(c[1] == 64.0f);
  assert(c[2] == 139.0f);
  assert(c[3] == 154.0f);

  mklibTensorDesc_t reduce_input_desc = nullptr;
  mklibTensorDesc_t reduce_output_desc = nullptr;
  assert(mklibCreateTensorDesc(&reduce_input_desc) == MKLIB_STATUS_SUCCESS);
  assert(mklibCreateTensorDesc(&reduce_output_desc) == MKLIB_STATUS_SUCCESS);

  const std::array<int64_t, 2> reduce_input_sizes = {2, 3};
  const std::array<int64_t, 2> reduce_input_strides = {3, 1};
  assert(mklibSetTensorDesc(
             reduce_input_desc,
             MKLIB_DATA_TYPE_FLOAT32,
             static_cast<int>(reduce_input_sizes.size()),
             reduce_input_sizes.data(),
             reduce_input_strides.data()) == MKLIB_STATUS_SUCCESS);

  const std::array<int64_t, 1> reduce_output_sizes = {2};
  const std::array<int64_t, 1> reduce_output_strides = {1};
  assert(mklibSetTensorDesc(
             reduce_output_desc,
             MKLIB_DATA_TYPE_FLOAT32,
             static_cast<int>(reduce_output_sizes.size()),
             reduce_output_sizes.data(),
             reduce_output_strides.data()) == MKLIB_STATUS_SUCCESS);

  const mklibReduceDesc_t reduce_desc = {
      .op = MKLIB_REDUCE_OP_SUM,
      .axis = 1,
      .keep_dim = 0,
  };
  const float reduce_input[2 * 3] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  float reduce_output[2] = {};
  workspace_bytes = 1;
  assert(mklibGetReduceWorkspaceSize(
             handle,
             reduce_input_desc,
             reduce_output_desc,
             &reduce_desc,
             &workspace_bytes) == MKLIB_STATUS_SUCCESS);
  assert(workspace_bytes == 0);
  assert(mklibReduce(
             handle,
             reduce_input_desc,
             reduce_input,
             reduce_output_desc,
             reduce_output,
             &reduce_desc,
             nullptr,
             workspace_bytes) == MKLIB_STATUS_SUCCESS);
  assert(reduce_output[0] == 6.0f);
  assert(reduce_output[1] == 15.0f);

  const mklibGemmDesc_t large_gemm_desc = {
      .a_type = MKLIB_DATA_TYPE_FLOAT32,
      .b_type = MKLIB_DATA_TYPE_FLOAT32,
      .c_type = MKLIB_DATA_TYPE_FLOAT32,
      .compute_type = MKLIB_DATA_TYPE_FLOAT32,
      .trans_a = MKLIB_OP_N,
      .trans_b = MKLIB_OP_N,
      .m = 256,
      .n = 256,
      .k = 256,
      .lda = 256,
      .ldb = 256,
      .ldc = 256,
      .epilogue = MKLIB_POINTWISE_MODE_RELU,
  };
  workspace_bytes = 0;
  assert(mklibGetGemmWorkspaceSize(handle, &large_gemm_desc, &workspace_bytes) ==
         MKLIB_STATUS_SUCCESS);
  assert(workspace_bytes > 0);

  mklibTensorDesc_t conv_input_desc = nullptr;
  mklibTensorDesc_t conv_filter_desc = nullptr;
  mklibTensorDesc_t conv_output_desc = nullptr;
  assert(mklibCreateTensorDesc(&conv_input_desc) == MKLIB_STATUS_SUCCESS);
  assert(mklibCreateTensorDesc(&conv_filter_desc) == MKLIB_STATUS_SUCCESS);
  assert(mklibCreateTensorDesc(&conv_output_desc) == MKLIB_STATUS_SUCCESS);

  const std::array<int64_t, 4> conv_input_sizes = {1, 1, 3, 3};
  const std::array<int64_t, 4> conv_input_strides = {9, 9, 3, 1};
  const std::array<int64_t, 4> conv_filter_sizes = {1, 1, 2, 2};
  const std::array<int64_t, 4> conv_filter_strides = {4, 4, 2, 1};
  const std::array<int64_t, 4> conv_output_sizes = {1, 1, 2, 2};
  const std::array<int64_t, 4> conv_output_strides = {4, 4, 2, 1};
  assert(mklibSetTensorDesc(
             conv_input_desc,
             MKLIB_DATA_TYPE_FLOAT32,
             static_cast<int>(conv_input_sizes.size()),
             conv_input_sizes.data(),
             conv_input_strides.data()) == MKLIB_STATUS_SUCCESS);
  assert(mklibSetTensorDesc(
             conv_filter_desc,
             MKLIB_DATA_TYPE_FLOAT32,
             static_cast<int>(conv_filter_sizes.size()),
             conv_filter_sizes.data(),
             conv_filter_strides.data()) == MKLIB_STATUS_SUCCESS);
  assert(mklibSetTensorDesc(
             conv_output_desc,
             MKLIB_DATA_TYPE_FLOAT32,
             static_cast<int>(conv_output_sizes.size()),
             conv_output_sizes.data(),
             conv_output_strides.data()) == MKLIB_STATUS_SUCCESS);

  const mklibConv2dDesc_t conv_desc = {
      .pad_h = 0,
      .pad_w = 0,
      .stride_h = 1,
      .stride_w = 1,
      .dilation_h = 1,
      .dilation_w = 1,
  };
  const float conv_input[9] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
  };
  const float conv_filter[4] = {
      1.0f, 0.0f,
      0.0f, -1.0f,
  };
  float conv_output[4] = {};
  workspace_bytes = 1;
  assert(mklibGetConv2dForwardWorkspaceSize(
             handle,
             &conv_desc,
             conv_input_desc,
             conv_filter_desc,
             conv_output_desc,
             &workspace_bytes) == MKLIB_STATUS_SUCCESS);
  assert(workspace_bytes == 0);
  assert(mklibConv2dForward(
             handle,
             &conv_desc,
             conv_input_desc,
             conv_input,
             conv_filter_desc,
             conv_filter,
             conv_output_desc,
             conv_output,
             nullptr,
             workspace_bytes) == MKLIB_STATUS_SUCCESS);
  assert(conv_output[0] == -4.0f);
  assert(conv_output[1] == -4.0f);
  assert(conv_output[2] == -4.0f);
  assert(conv_output[3] == -4.0f);

  assert(mklibDestroyTensorDesc(conv_output_desc) == MKLIB_STATUS_SUCCESS);
  assert(mklibDestroyTensorDesc(conv_filter_desc) == MKLIB_STATUS_SUCCESS);
  assert(mklibDestroyTensorDesc(conv_input_desc) == MKLIB_STATUS_SUCCESS);
  assert(mklibDestroyTensorDesc(reduce_output_desc) == MKLIB_STATUS_SUCCESS);
  assert(mklibDestroyTensorDesc(reduce_input_desc) == MKLIB_STATUS_SUCCESS);
  assert(mklibDestroyTensorDesc(desc) == MKLIB_STATUS_SUCCESS);
  assert(mklibDestroy(handle) == MKLIB_STATUS_SUCCESS);
  return 0;
}
