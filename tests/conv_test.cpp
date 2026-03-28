#include <cassert>
#include <cstdint>
#include <vector>

#include "mklib/mklib.h"

namespace {

class TensorDescOwner {
 public:
  TensorDescOwner() { assert(mklibCreateTensorDesc(&desc_) == MKLIB_STATUS_SUCCESS); }
  ~TensorDescOwner() { assert(mklibDestroyTensorDesc(desc_) == MKLIB_STATUS_SUCCESS); }

  mklibTensorDesc_t get() const { return desc_; }

 private:
  mklibTensorDesc_t desc_ = nullptr;
};

std::vector<int64_t> MakeContiguousStrides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  int64_t stride = 1;
  for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = stride;
    stride *= sizes[static_cast<size_t>(i)];
  }
  return strides;
}

void SetContiguousDesc(mklibTensorDesc_t desc, mklibDataType_t dtype, const std::vector<int64_t>& sizes) {
  const auto strides = MakeContiguousStrides(sizes);
  assert(mklibSetTensorDesc(
             desc,
             dtype,
             static_cast<int>(sizes.size()),
             sizes.data(),
             strides.data()) == MKLIB_STATUS_SUCCESS);
}

std::vector<float> MakeValues(int64_t elements, int seed) {
  std::vector<float> values(static_cast<size_t>(elements), 0.0f);
  for (int64_t i = 0; i < elements; ++i) {
    values[static_cast<size_t>(i)] = static_cast<float>(((seed + i * 3) % 17) - 8) / 5.0f;
  }
  return values;
}

int64_t InputIndex(const std::vector<int64_t>& sizes, int64_t n, int64_t c, int64_t h, int64_t w) {
  return ((n * sizes[1] + c) * sizes[2] + h) * sizes[3] + w;
}

int64_t FilterIndex(const std::vector<int64_t>& sizes, int64_t k, int64_t c, int64_t r, int64_t s) {
  return ((k * sizes[1] + c) * sizes[2] + r) * sizes[3] + s;
}

int64_t OutputIndex(const std::vector<int64_t>& sizes, int64_t n, int64_t k, int64_t p, int64_t q) {
  return ((n * sizes[1] + k) * sizes[2] + p) * sizes[3] + q;
}

std::vector<float> ReferenceConv2d(
    const std::vector<int64_t>& input_sizes,
    const std::vector<int64_t>& filter_sizes,
    const std::vector<int64_t>& output_sizes,
    const mklibConv2dDesc_t& desc,
    const std::vector<float>& input,
    const std::vector<float>& filter) {
  std::vector<float> output(static_cast<size_t>(
      output_sizes[0] * output_sizes[1] * output_sizes[2] * output_sizes[3]), 0.0f);

  for (int64_t n = 0; n < output_sizes[0]; ++n) {
    for (int64_t k = 0; k < output_sizes[1]; ++k) {
      for (int64_t p = 0; p < output_sizes[2]; ++p) {
        for (int64_t q = 0; q < output_sizes[3]; ++q) {
          float sum = 0.0f;
          for (int64_t c = 0; c < input_sizes[1]; ++c) {
            for (int64_t r = 0; r < filter_sizes[2]; ++r) {
              const int64_t input_h = p * desc.stride_h - desc.pad_h + r * desc.dilation_h;
              if (input_h < 0 || input_h >= input_sizes[2]) {
                continue;
              }
              for (int64_t s = 0; s < filter_sizes[3]; ++s) {
                const int64_t input_w = q * desc.stride_w - desc.pad_w + s * desc.dilation_w;
                if (input_w < 0 || input_w >= input_sizes[3]) {
                  continue;
                }
                sum += input[static_cast<size_t>(InputIndex(input_sizes, n, c, input_h, input_w))] *
                       filter[static_cast<size_t>(FilterIndex(filter_sizes, k, c, r, s))];
              }
            }
          }
          output[static_cast<size_t>(OutputIndex(output_sizes, n, k, p, q))] = sum;
        }
      }
    }
  }

  return output;
}

void CheckConvCase(
    mklibHandle_t handle,
    const std::vector<int64_t>& input_sizes,
    const std::vector<int64_t>& filter_sizes,
    const std::vector<int64_t>& output_sizes,
    const mklibConv2dDesc_t& desc) {
  TensorDescOwner input_desc;
  TensorDescOwner filter_desc;
  TensorDescOwner output_desc;

  SetContiguousDesc(input_desc.get(), MKLIB_DATA_TYPE_FLOAT32, input_sizes);
  SetContiguousDesc(filter_desc.get(), MKLIB_DATA_TYPE_FLOAT32, filter_sizes);
  SetContiguousDesc(output_desc.get(), MKLIB_DATA_TYPE_FLOAT32, output_sizes);

  const auto input = MakeValues(input_sizes[0] * input_sizes[1] * input_sizes[2] * input_sizes[3], 3);
  const auto filter =
      MakeValues(filter_sizes[0] * filter_sizes[1] * filter_sizes[2] * filter_sizes[3], 11);
  auto expected = ReferenceConv2d(input_sizes, filter_sizes, output_sizes, desc, input, filter);
  std::vector<float> output(expected.size(), -4.0f);

  size_t workspace_bytes = 5;
  assert(mklibGetConv2dForwardWorkspaceSize(
             handle,
             &desc,
             input_desc.get(),
             filter_desc.get(),
             output_desc.get(),
             &workspace_bytes) == MKLIB_STATUS_SUCCESS);
  assert(workspace_bytes == 0);
  assert(mklibConv2dForward(
             handle,
             &desc,
             input_desc.get(),
             input.data(),
             filter_desc.get(),
             filter.data(),
             output_desc.get(),
             output.data(),
             nullptr,
             workspace_bytes) == MKLIB_STATUS_SUCCESS);
  assert(output == expected);
}

void CheckUnsupportedStridedInput(mklibHandle_t handle) {
  TensorDescOwner input_desc;
  TensorDescOwner filter_desc;
  TensorDescOwner output_desc;

  const std::vector<int64_t> input_sizes = {1, 1, 4, 4};
  const std::vector<int64_t> input_strides = {20, 20, 5, 1};
  assert(mklibSetTensorDesc(
             input_desc.get(),
             MKLIB_DATA_TYPE_FLOAT32,
             static_cast<int>(input_sizes.size()),
             input_sizes.data(),
             input_strides.data()) == MKLIB_STATUS_SUCCESS);

  SetContiguousDesc(filter_desc.get(), MKLIB_DATA_TYPE_FLOAT32, {1, 1, 3, 3});
  SetContiguousDesc(output_desc.get(), MKLIB_DATA_TYPE_FLOAT32, {1, 1, 2, 2});

  const mklibConv2dDesc_t desc = {
      .pad_h = 0,
      .pad_w = 0,
      .stride_h = 1,
      .stride_w = 1,
      .dilation_h = 1,
      .dilation_w = 1,
  };

  size_t workspace_bytes = 0;
  assert(mklibGetConv2dForwardWorkspaceSize(
             handle,
             &desc,
             input_desc.get(),
             filter_desc.get(),
             output_desc.get(),
             &workspace_bytes) == MKLIB_STATUS_NOT_SUPPORTED);
}

void CheckUnsupportedDtype(mklibHandle_t handle) {
  TensorDescOwner input_desc;
  TensorDescOwner filter_desc;
  TensorDescOwner output_desc;

  SetContiguousDesc(input_desc.get(), MKLIB_DATA_TYPE_FLOAT16, {1, 1, 4, 4});
  SetContiguousDesc(filter_desc.get(), MKLIB_DATA_TYPE_FLOAT16, {1, 1, 3, 3});
  SetContiguousDesc(output_desc.get(), MKLIB_DATA_TYPE_FLOAT16, {1, 1, 2, 2});

  const mklibConv2dDesc_t desc = {
      .pad_h = 0,
      .pad_w = 0,
      .stride_h = 1,
      .stride_w = 1,
      .dilation_h = 1,
      .dilation_w = 1,
  };

  size_t workspace_bytes = 0;
  assert(mklibGetConv2dForwardWorkspaceSize(
             handle,
             &desc,
             input_desc.get(),
             filter_desc.get(),
             output_desc.get(),
             &workspace_bytes) == MKLIB_STATUS_NOT_SUPPORTED);
}

void CheckOutputShapeMismatch(mklibHandle_t handle) {
  TensorDescOwner input_desc;
  TensorDescOwner filter_desc;
  TensorDescOwner output_desc;

  SetContiguousDesc(input_desc.get(), MKLIB_DATA_TYPE_FLOAT32, {1, 1, 4, 4});
  SetContiguousDesc(filter_desc.get(), MKLIB_DATA_TYPE_FLOAT32, {1, 1, 3, 3});
  SetContiguousDesc(output_desc.get(), MKLIB_DATA_TYPE_FLOAT32, {1, 1, 4, 4});

  const mklibConv2dDesc_t desc = {
      .pad_h = 0,
      .pad_w = 0,
      .stride_h = 1,
      .stride_w = 1,
      .dilation_h = 1,
      .dilation_w = 1,
  };

  size_t workspace_bytes = 0;
  assert(mklibGetConv2dForwardWorkspaceSize(
             handle,
             &desc,
             input_desc.get(),
             filter_desc.get(),
             output_desc.get(),
             &workspace_bytes) == MKLIB_STATUS_INVALID_ARGUMENT);
}

}  // namespace

int main() {
  mklibHandle_t handle = nullptr;
  assert(mklibCreate(&handle) == MKLIB_STATUS_SUCCESS);

  CheckConvCase(
      handle,
      {1, 1, 4, 4},
      {1, 1, 3, 3},
      {1, 1, 4, 4},
      {
          .pad_h = 1,
          .pad_w = 1,
          .stride_h = 1,
          .stride_w = 1,
          .dilation_h = 1,
          .dilation_w = 1,
      });
  CheckConvCase(
      handle,
      {1, 2, 5, 5},
      {3, 2, 3, 3},
      {1, 3, 3, 3},
      {
          .pad_h = 1,
          .pad_w = 1,
          .stride_h = 2,
          .stride_w = 2,
          .dilation_h = 1,
          .dilation_w = 1,
      });

  CheckUnsupportedStridedInput(handle);
  CheckUnsupportedDtype(handle);
  CheckOutputShapeMismatch(handle);

  assert(mklibDestroy(handle) == MKLIB_STATUS_SUCCESS);
  return 0;
}
