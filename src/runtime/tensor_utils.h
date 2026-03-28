#ifndef MKLIB_RUNTIME_TENSOR_UTILS_H_
#define MKLIB_RUNTIME_TENSOR_UTILS_H_

#include <cstdint>
#include <limits>

#include "mklib/conv.h"
#include "runtime/tensor_desc_state.h"

namespace mklib::runtime {

struct ReductionGeometry {
  int axis = 0;
  int64_t outer_size = 0;
  int64_t reduce_size = 0;
  int64_t inner_size = 0;
};

struct Conv2dGeometry {
  int64_t batch = 0;
  int64_t in_channels = 0;
  int64_t in_h = 0;
  int64_t in_w = 0;
  int64_t out_channels = 0;
  int64_t kernel_h = 0;
  int64_t kernel_w = 0;
  int64_t out_h = 0;
  int64_t out_w = 0;
};

inline bool IsInitialized(const ::mklibTensorDesc_t desc) {
  return desc != nullptr && desc->initialized;
}

inline bool CheckedMultiply(int64_t lhs, int64_t rhs, int64_t* out) {
  if (lhs == 0 || rhs == 0) {
    *out = 0;
    return true;
  }
  if (lhs > std::numeric_limits<int64_t>::max() / rhs) {
    return false;
  }
  *out = lhs * rhs;
  return true;
}

inline bool ElementCount(const ::mklibTensorDesc& desc, int64_t* elements_out) {
  int64_t total = 1;
  for (int i = 0; i < desc.rank; ++i) {
    if (!CheckedMultiply(total, desc.sizes[static_cast<size_t>(i)], &total)) {
      return false;
    }
  }
  *elements_out = total;
  return true;
}

inline bool NormalizeAxis(int axis, int rank, int* axis_out) {
  if (rank <= 0) {
    return false;
  }
  int normalized = axis;
  if (normalized < 0) {
    normalized += rank;
  }
  if (normalized < 0 || normalized >= rank) {
    return false;
  }
  *axis_out = normalized;
  return true;
}

inline bool IsContiguous(const ::mklibTensorDesc& desc) {
  int64_t expected_stride = 1;
  for (int i = desc.rank - 1; i >= 0; --i) {
    if (desc.sizes[static_cast<size_t>(i)] == 0) {
      return true;
    }
    if (desc.strides[static_cast<size_t>(i)] != expected_stride) {
      return false;
    }
    if (!CheckedMultiply(expected_stride, desc.sizes[static_cast<size_t>(i)], &expected_stride)) {
      return false;
    }
  }
  return true;
}

inline bool MakeReductionGeometry(
    const ::mklibTensorDesc& desc,
    int axis,
    ReductionGeometry* geometry_out) {
  int normalized_axis = 0;
  if (!NormalizeAxis(axis, desc.rank, &normalized_axis)) {
    return false;
  }

  ReductionGeometry geometry;
  geometry.axis = normalized_axis;
  geometry.outer_size = 1;
  geometry.reduce_size = desc.sizes[static_cast<size_t>(normalized_axis)];
  geometry.inner_size = 1;

  for (int i = 0; i < normalized_axis; ++i) {
    if (!CheckedMultiply(
            geometry.outer_size,
            desc.sizes[static_cast<size_t>(i)],
            &geometry.outer_size)) {
      return false;
    }
  }
  for (int i = normalized_axis + 1; i < desc.rank; ++i) {
    if (!CheckedMultiply(
            geometry.inner_size,
            desc.sizes[static_cast<size_t>(i)],
            &geometry.inner_size)) {
      return false;
    }
  }

  *geometry_out = geometry;
  return true;
}

inline bool MatchesReducedOutputShape(
    const ::mklibTensorDesc& input_desc,
    const ::mklibTensorDesc& output_desc,
    const ReductionGeometry& geometry,
    bool keep_dim) {
  const int expected_rank = keep_dim ? input_desc.rank : input_desc.rank - 1;
  if (output_desc.rank != expected_rank) {
    return false;
  }

  int output_dim = 0;
  for (int input_dim = 0; input_dim < input_desc.rank; ++input_dim) {
    const int64_t input_size = input_desc.sizes[static_cast<size_t>(input_dim)];
    if (input_dim == geometry.axis) {
      if (keep_dim) {
        if (output_desc.sizes[static_cast<size_t>(output_dim)] != 1) {
          return false;
        }
        ++output_dim;
      }
      continue;
    }
    if (output_desc.sizes[static_cast<size_t>(output_dim)] != input_size) {
      return false;
    }
    ++output_dim;
  }

  return true;
}

inline bool MakeConv2dGeometry(
    const ::mklibTensorDesc& input_desc,
    const ::mklibTensorDesc& filter_desc,
    const mklibConv2dDesc_t& desc,
    Conv2dGeometry* geometry_out) {
  if (input_desc.rank != 4 || filter_desc.rank != 4) {
    return false;
  }
  if (desc.pad_h < 0 || desc.pad_w < 0 ||
      desc.stride_h <= 0 || desc.stride_w <= 0 ||
      desc.dilation_h <= 0 || desc.dilation_w <= 0) {
    return false;
  }

  Conv2dGeometry geometry;
  geometry.batch = input_desc.sizes[0];
  geometry.in_channels = input_desc.sizes[1];
  geometry.in_h = input_desc.sizes[2];
  geometry.in_w = input_desc.sizes[3];
  geometry.out_channels = filter_desc.sizes[0];
  const int64_t filter_in_channels = filter_desc.sizes[1];
  geometry.kernel_h = filter_desc.sizes[2];
  geometry.kernel_w = filter_desc.sizes[3];

  if (geometry.in_channels != filter_in_channels) {
    return false;
  }

  const int64_t effective_kernel_h =
      (geometry.kernel_h - 1) * desc.dilation_h + 1;
  const int64_t effective_kernel_w =
      (geometry.kernel_w - 1) * desc.dilation_w + 1;
  const int64_t padded_h = geometry.in_h + 2 * desc.pad_h;
  const int64_t padded_w = geometry.in_w + 2 * desc.pad_w;
  if (effective_kernel_h <= 0 || effective_kernel_w <= 0 ||
      padded_h < effective_kernel_h || padded_w < effective_kernel_w) {
    return false;
  }

  geometry.out_h = (padded_h - effective_kernel_h) / desc.stride_h + 1;
  geometry.out_w = (padded_w - effective_kernel_w) / desc.stride_w + 1;
  if (geometry.out_h <= 0 || geometry.out_w <= 0) {
    return false;
  }

  *geometry_out = geometry;
  return true;
}

inline bool MatchesConv2dOutputShape(
    const ::mklibTensorDesc& output_desc,
    const Conv2dGeometry& geometry) {
  return output_desc.rank == 4 &&
         output_desc.sizes[0] == geometry.batch &&
         output_desc.sizes[1] == geometry.out_channels &&
         output_desc.sizes[2] == geometry.out_h &&
         output_desc.sizes[3] == geometry.out_w;
}

}  // namespace mklib::runtime

#endif  // MKLIB_RUNTIME_TENSOR_UTILS_H_
