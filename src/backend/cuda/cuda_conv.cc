#include "backend/backend.h"

#include "runtime/tensor_utils.h"

namespace {

int64_t InputIndex(const mklib::runtime::Conv2dGeometry& geometry, int64_t n, int64_t c, int64_t h, int64_t w) {
  return ((n * geometry.in_channels + c) * geometry.in_h + h) * geometry.in_w + w;
}

int64_t FilterIndex(
    const mklib::runtime::Conv2dGeometry& geometry,
    int64_t k,
    int64_t c,
    int64_t r,
    int64_t s) {
  return ((k * geometry.in_channels + c) * geometry.kernel_h + r) * geometry.kernel_w + s;
}

int64_t OutputIndex(
    const mklib::runtime::Conv2dGeometry& geometry,
    int64_t n,
    int64_t k,
    int64_t p,
    int64_t q) {
  return ((n * geometry.out_channels + k) * geometry.out_h + p) * geometry.out_w + q;
}

mklibStatus_t LaunchConv2dF32Direct(
    const ::mklibTensorDesc& input_desc,
    const void* input,
    const ::mklibTensorDesc& filter_desc,
    const void* filter,
    const ::mklibTensorDesc& output_desc,
    void* output,
    const mklibConv2dDesc_t& desc) {
  (void)output_desc;

  mklib::runtime::Conv2dGeometry geometry;
  if (!mklib::runtime::MakeConv2dGeometry(input_desc, filter_desc, desc, &geometry)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  const auto* input_data = static_cast<const float*>(input);
  const auto* filter_data = static_cast<const float*>(filter);
  auto* output_data = static_cast<float*>(output);

  for (int64_t n = 0; n < geometry.batch; ++n) {
    for (int64_t k = 0; k < geometry.out_channels; ++k) {
      for (int64_t p = 0; p < geometry.out_h; ++p) {
        for (int64_t q = 0; q < geometry.out_w; ++q) {
          float sum = 0.0f;
          for (int64_t c = 0; c < geometry.in_channels; ++c) {
            for (int64_t r = 0; r < geometry.kernel_h; ++r) {
              const int64_t input_h = p * desc.stride_h - desc.pad_h + r * desc.dilation_h;
              if (input_h < 0 || input_h >= geometry.in_h) {
                continue;
              }
              for (int64_t s = 0; s < geometry.kernel_w; ++s) {
                const int64_t input_w =
                    q * desc.stride_w - desc.pad_w + s * desc.dilation_w;
                if (input_w < 0 || input_w >= geometry.in_w) {
                  continue;
                }
                sum += input_data[InputIndex(geometry, n, c, input_h, input_w)] *
                       filter_data[FilterIndex(geometry, k, c, r, s)];
              }
            }
          }
          output_data[OutputIndex(geometry, n, k, p, q)] = sum;
        }
      }
    }
  }

  return MKLIB_STATUS_SUCCESS;
}

}  // namespace

namespace mklib::backend {

size_t GetConv2dForwardWorkspaceSize(
    const planner::DispatchKey& key,
    const registry::KernelRecord& kernel,
    const ::mklibTensorDesc& input_desc,
    const ::mklibTensorDesc& filter_desc,
    const ::mklibTensorDesc& output_desc,
    const mklibConv2dDesc_t& desc) {
  (void)key;
  (void)kernel;
  (void)input_desc;
  (void)filter_desc;
  (void)output_desc;
  (void)desc;
  return 0;
}

mklibStatus_t LaunchConv2dForward(
    const ::mklibHandle& handle,
    const planner::DispatchKey& key,
    const registry::KernelRecord& kernel,
    const ::mklibTensorDesc& input_desc,
    const void* input,
    const ::mklibTensorDesc& filter_desc,
    const void* filter,
    const ::mklibTensorDesc& output_desc,
    void* output,
    const mklibConv2dDesc_t& desc,
    void* workspace,
    size_t workspace_size) {
  (void)handle;
  (void)key;
  (void)workspace;
  (void)workspace_size;

  switch (kernel.kind) {
    case registry::KernelKind::kConv2dF32Direct:
      return LaunchConv2dF32Direct(input_desc, input, filter_desc, filter, output_desc, output, desc);
    case registry::KernelKind::kReferenceF32Direct:
    case registry::KernelKind::kReferenceF32Blocked:
    case registry::KernelKind::kReduceF32InnerContiguous:
    case registry::KernelKind::kReduceF32GenericContiguous:
      break;
  }

  return MKLIB_STATUS_INTERNAL_ERROR;
}

}  // namespace mklib::backend
