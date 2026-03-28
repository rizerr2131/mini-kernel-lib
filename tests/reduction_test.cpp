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

std::vector<float> MakeInputData(const std::vector<int64_t>& sizes) {
  int64_t elements = 1;
  for (const int64_t size : sizes) {
    elements *= size;
  }

  std::vector<float> values(static_cast<size_t>(elements), 0.0f);
  for (int64_t i = 0; i < elements; ++i) {
    values[static_cast<size_t>(i)] = static_cast<float>((i % 13) - 6) / 4.0f;
  }
  return values;
}

std::vector<float> ReferenceReduce(
    const std::vector<int64_t>& input_sizes,
    int axis,
    bool keep_dim,
    const std::vector<float>& input) {
  int normalized_axis = axis;
  if (normalized_axis < 0) {
    normalized_axis += static_cast<int>(input_sizes.size());
  }

  int64_t outer = 1;
  int64_t reduce = input_sizes[static_cast<size_t>(normalized_axis)];
  int64_t inner = 1;
  for (int i = 0; i < normalized_axis; ++i) {
    outer *= input_sizes[static_cast<size_t>(i)];
  }
  for (size_t i = static_cast<size_t>(normalized_axis + 1); i < input_sizes.size(); ++i) {
    inner *= input_sizes[i];
  }

  std::vector<int64_t> output_sizes;
  output_sizes.reserve(input_sizes.size());
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    if (static_cast<int>(i) == normalized_axis) {
      if (keep_dim) {
        output_sizes.push_back(1);
      }
      continue;
    }
    output_sizes.push_back(input_sizes[i]);
  }

  int64_t output_elements = 1;
  for (const int64_t size : output_sizes) {
    output_elements *= size;
  }
  std::vector<float> output(static_cast<size_t>(output_elements), 0.0f);

  for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
    for (int64_t inner_index = 0; inner_index < inner; ++inner_index) {
      float sum = 0.0f;
      const int64_t base = outer_index * reduce * inner + inner_index;
      for (int64_t reduce_index = 0; reduce_index < reduce; ++reduce_index) {
        sum += input[static_cast<size_t>(base + reduce_index * inner)];
      }
      output[static_cast<size_t>(outer_index * inner + inner_index)] = sum;
    }
  }

  return output;
}

void CheckReduceCase(
    mklibHandle_t handle,
    const std::vector<int64_t>& input_sizes,
    int axis,
    bool keep_dim) {
  TensorDescOwner input_desc;
  TensorDescOwner output_desc;

  SetContiguousDesc(input_desc.get(), MKLIB_DATA_TYPE_FLOAT32, input_sizes);

  std::vector<int64_t> output_sizes;
  output_sizes.reserve(input_sizes.size());
  int normalized_axis = axis;
  if (normalized_axis < 0) {
    normalized_axis += static_cast<int>(input_sizes.size());
  }
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    if (static_cast<int>(i) == normalized_axis) {
      if (keep_dim) {
        output_sizes.push_back(1);
      }
      continue;
    }
    output_sizes.push_back(input_sizes[i]);
  }
  SetContiguousDesc(output_desc.get(), MKLIB_DATA_TYPE_FLOAT32, output_sizes);

  const auto input = MakeInputData(input_sizes);
  auto expected = ReferenceReduce(input_sizes, axis, keep_dim, input);
  std::vector<float> output(expected.size(), -3.0f);

  const mklibReduceDesc_t desc = {
      .op = MKLIB_REDUCE_OP_SUM,
      .axis = axis,
      .keep_dim = keep_dim ? 1 : 0,
  };

  size_t workspace_bytes = 99;
  assert(mklibGetReduceWorkspaceSize(handle, input_desc.get(), output_desc.get(), &desc, &workspace_bytes) ==
         MKLIB_STATUS_SUCCESS);
  assert(workspace_bytes == 0);
  assert(mklibReduce(
             handle,
             input_desc.get(),
             input.data(),
             output_desc.get(),
             output.data(),
             &desc,
             nullptr,
             workspace_bytes) == MKLIB_STATUS_SUCCESS);

  assert(output == expected);
}

void CheckUnsupportedStridedInput(mklibHandle_t handle) {
  TensorDescOwner input_desc;
  TensorDescOwner output_desc;

  const std::vector<int64_t> input_sizes = {2, 3, 4};
  const std::vector<int64_t> input_strides = {16, 4, 1};
  assert(mklibSetTensorDesc(
             input_desc.get(),
             MKLIB_DATA_TYPE_FLOAT32,
             static_cast<int>(input_sizes.size()),
             input_sizes.data(),
             input_strides.data()) == MKLIB_STATUS_SUCCESS);

  SetContiguousDesc(output_desc.get(), MKLIB_DATA_TYPE_FLOAT32, {2, 3});

  const mklibReduceDesc_t desc = {
      .op = MKLIB_REDUCE_OP_SUM,
      .axis = 2,
      .keep_dim = 0,
  };

  size_t workspace_bytes = 0;
  assert(mklibGetReduceWorkspaceSize(handle, input_desc.get(), output_desc.get(), &desc, &workspace_bytes) ==
         MKLIB_STATUS_NOT_SUPPORTED);
}

void CheckUnsupportedDtype(mklibHandle_t handle) {
  TensorDescOwner input_desc;
  TensorDescOwner output_desc;

  SetContiguousDesc(input_desc.get(), MKLIB_DATA_TYPE_FLOAT16, {4, 8});
  SetContiguousDesc(output_desc.get(), MKLIB_DATA_TYPE_FLOAT16, {4});

  const mklibReduceDesc_t desc = {
      .op = MKLIB_REDUCE_OP_SUM,
      .axis = 1,
      .keep_dim = 0,
  };

  size_t workspace_bytes = 0;
  assert(mklibGetReduceWorkspaceSize(handle, input_desc.get(), output_desc.get(), &desc, &workspace_bytes) ==
         MKLIB_STATUS_NOT_SUPPORTED);
}

void CheckShapeMismatch(mklibHandle_t handle) {
  TensorDescOwner input_desc;
  TensorDescOwner output_desc;

  SetContiguousDesc(input_desc.get(), MKLIB_DATA_TYPE_FLOAT32, {2, 3, 4});
  SetContiguousDesc(output_desc.get(), MKLIB_DATA_TYPE_FLOAT32, {2, 4, 1});

  const mklibReduceDesc_t desc = {
      .op = MKLIB_REDUCE_OP_SUM,
      .axis = 1,
      .keep_dim = 0,
  };

  size_t workspace_bytes = 0;
  assert(mklibGetReduceWorkspaceSize(handle, input_desc.get(), output_desc.get(), &desc, &workspace_bytes) ==
         MKLIB_STATUS_INVALID_ARGUMENT);
}

void CheckInvalidAxis(mklibHandle_t handle) {
  TensorDescOwner input_desc;
  TensorDescOwner output_desc;

  SetContiguousDesc(input_desc.get(), MKLIB_DATA_TYPE_FLOAT32, {2, 3, 4});
  SetContiguousDesc(output_desc.get(), MKLIB_DATA_TYPE_FLOAT32, {2, 3});

  const mklibReduceDesc_t desc = {
      .op = MKLIB_REDUCE_OP_SUM,
      .axis = 4,
      .keep_dim = 0,
  };

  size_t workspace_bytes = 0;
  assert(mklibGetReduceWorkspaceSize(handle, input_desc.get(), output_desc.get(), &desc, &workspace_bytes) ==
         MKLIB_STATUS_INVALID_ARGUMENT);
}

void CheckZeroSizedInput(mklibHandle_t handle) {
  TensorDescOwner input_desc;
  TensorDescOwner output_desc;

  SetContiguousDesc(input_desc.get(), MKLIB_DATA_TYPE_FLOAT32, {2, 0, 3});
  SetContiguousDesc(output_desc.get(), MKLIB_DATA_TYPE_FLOAT32, {2, 3});

  const mklibReduceDesc_t desc = {
      .op = MKLIB_REDUCE_OP_SUM,
      .axis = 1,
      .keep_dim = 0,
  };

  size_t workspace_bytes = 5;
  assert(mklibGetReduceWorkspaceSize(handle, input_desc.get(), output_desc.get(), &desc, &workspace_bytes) ==
         MKLIB_STATUS_SUCCESS);
  assert(workspace_bytes == 0);
  assert(mklibReduce(
             handle,
             input_desc.get(),
             nullptr,
             output_desc.get(),
             nullptr,
             &desc,
             nullptr,
             workspace_bytes) == MKLIB_STATUS_SUCCESS);
}

}  // namespace

int main() {
  mklibHandle_t handle = nullptr;
  assert(mklibCreate(&handle) == MKLIB_STATUS_SUCCESS);

  CheckReduceCase(handle, {2, 3, 4}, 2, false);
  CheckReduceCase(handle, {2, 3, 4}, 1, true);
  CheckReduceCase(handle, {4, 5}, -1, false);
  CheckUnsupportedStridedInput(handle);
  CheckUnsupportedDtype(handle);
  CheckShapeMismatch(handle);
  CheckInvalidAxis(handle);
  CheckZeroSizedInput(handle);

  assert(mklibDestroy(handle) == MKLIB_STATUS_SUCCESS);
  return 0;
}
