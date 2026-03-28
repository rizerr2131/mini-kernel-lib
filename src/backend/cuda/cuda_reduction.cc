#include "backend/backend.h"

#include "runtime/tensor_utils.h"

namespace {

mklibStatus_t LaunchReduceF32InnerContiguous(
    const ::mklibTensorDesc& input_desc,
    const void* input,
    const ::mklibTensorDesc& output_desc,
    void* output,
    const mklibReduceDesc_t& desc) {
  (void)output_desc;

  mklib::runtime::ReductionGeometry geometry;
  if (!mklib::runtime::MakeReductionGeometry(input_desc, desc.axis, &geometry)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  const auto* input_data = static_cast<const float*>(input);
  auto* output_data = static_cast<float*>(output);

  for (int64_t outer = 0; outer < geometry.outer_size; ++outer) {
    const int64_t base = outer * geometry.reduce_size;
    float sum = 0.0f;
    int64_t depth = 0;
    for (; depth + 3 < geometry.reduce_size; depth += 4) {
      sum += input_data[base + depth];
      sum += input_data[base + depth + 1];
      sum += input_data[base + depth + 2];
      sum += input_data[base + depth + 3];
    }
    for (; depth < geometry.reduce_size; ++depth) {
      sum += input_data[base + depth];
    }
    output_data[outer] = sum;
  }

  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t LaunchReduceF32GenericContiguous(
    const ::mklibTensorDesc& input_desc,
    const void* input,
    const ::mklibTensorDesc& output_desc,
    void* output,
    const mklibReduceDesc_t& desc) {
  (void)output_desc;

  mklib::runtime::ReductionGeometry geometry;
  if (!mklib::runtime::MakeReductionGeometry(input_desc, desc.axis, &geometry)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  const auto* input_data = static_cast<const float*>(input);
  auto* output_data = static_cast<float*>(output);

  for (int64_t outer = 0; outer < geometry.outer_size; ++outer) {
    for (int64_t inner = 0; inner < geometry.inner_size; ++inner) {
      float sum = 0.0f;
      const int64_t base =
          outer * geometry.reduce_size * geometry.inner_size + inner;
      for (int64_t depth = 0; depth < geometry.reduce_size; ++depth) {
        sum += input_data[base + depth * geometry.inner_size];
      }
      output_data[outer * geometry.inner_size + inner] = sum;
    }
  }

  return MKLIB_STATUS_SUCCESS;
}

}  // namespace

namespace mklib::backend {

size_t GetReduceWorkspaceSize(
    const planner::DispatchKey& key,
    const registry::KernelRecord& kernel,
    const ::mklibTensorDesc& input_desc,
    const ::mklibTensorDesc& output_desc,
    const mklibReduceDesc_t& desc) {
  (void)key;
  (void)kernel;
  (void)input_desc;
  (void)output_desc;
  (void)desc;
  return 0;
}

mklibStatus_t LaunchReduce(
    const ::mklibHandle& handle,
    const planner::DispatchKey& key,
    const registry::KernelRecord& kernel,
    const ::mklibTensorDesc& input_desc,
    const void* input,
    const ::mklibTensorDesc& output_desc,
    void* output,
    const mklibReduceDesc_t& desc,
    void* workspace,
    size_t workspace_size) {
  (void)handle;
  (void)key;
  (void)workspace;
  (void)workspace_size;

  switch (kernel.kind) {
    case registry::KernelKind::kReduceF32InnerContiguous:
      return LaunchReduceF32InnerContiguous(input_desc, input, output_desc, output, desc);
    case registry::KernelKind::kReduceF32GenericContiguous:
      return LaunchReduceF32GenericContiguous(input_desc, input, output_desc, output, desc);
    case registry::KernelKind::kConv2dF32Direct:
    case registry::KernelKind::kReferenceF32Direct:
    case registry::KernelKind::kReferenceF32Blocked:
      break;
  }

  return MKLIB_STATUS_INTERNAL_ERROR;
}

}  // namespace mklib::backend
