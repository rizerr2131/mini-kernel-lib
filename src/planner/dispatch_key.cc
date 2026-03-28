#include "planner/dispatch_key.h"

#include <algorithm>
#include <string>

#include "runtime/tensor_utils.h"

namespace mklib::planner {
namespace {

ShapeBucket BucketForProblem(const mklibGemmDesc_t& desc) {
  const int64_t max_dim = std::max({desc.m, desc.n, desc.k});
  if (max_dim <= 128) {
    return ShapeBucket::kSmall;
  }
  if (max_dim <= 1024) {
    return ShapeBucket::kMedium;
  }
  return ShapeBucket::kLarge;
}

ShapeBucket BucketForElements(int64_t elements) {
  if (elements <= 4096) {
    return ShapeBucket::kSmall;
  }
  if (elements <= 65536) {
    return ShapeBucket::kMedium;
  }
  return ShapeBucket::kLarge;
}

const char* BucketString(ShapeBucket bucket) {
  switch (bucket) {
    case ShapeBucket::kSmall:
      return "small";
    case ShapeBucket::kMedium:
      return "medium";
    case ShapeBucket::kLarge:
      return "large";
  }
  return "unknown";
}

const char* DataTypeString(mklibDataType_t type) {
  switch (type) {
    case MKLIB_DATA_TYPE_FLOAT32:
      return "f32";
    case MKLIB_DATA_TYPE_FLOAT16:
      return "f16";
    case MKLIB_DATA_TYPE_BFLOAT16:
      return "bf16";
    case MKLIB_DATA_TYPE_INVALID:
      return "invalid";
  }
  return "unknown";
}

const char* TransposeString(mklibTranspose_t transpose) {
  switch (transpose) {
    case MKLIB_OP_N:
      return "n";
    case MKLIB_OP_T:
      return "t";
  }
  return "?";
}

const char* LayoutString(LayoutKind layout) {
  switch (layout) {
    case LayoutKind::kContiguous:
      return "contiguous";
    case LayoutKind::kStrided:
      return "strided";
  }
  return "unknown";
}

const char* AxisRoleString(AxisRole axis_role) {
  switch (axis_role) {
    case AxisRole::kNone:
      return "none";
    case AxisRole::kInner:
      return "inner";
    case AxisRole::kMiddle:
      return "middle";
    case AxisRole::kOuter:
      return "outer";
  }
  return "unknown";
}

const char* PointwiseString(mklibPointwiseMode_t pointwise) {
  switch (pointwise) {
    case MKLIB_POINTWISE_MODE_IDENTITY:
      return "identity";
    case MKLIB_POINTWISE_MODE_RELU:
      return "relu";
  }
  return "unknown";
}

}  // namespace

DispatchKey BuildGemmDispatchKey(const mklibGemmDesc_t& desc) {
  DispatchKey key;
  key.operation = OperationKind::kGemm;
  key.a_type = desc.a_type;
  key.b_type = desc.b_type;
  key.c_type = desc.c_type;
  key.compute_type = desc.compute_type;
  key.trans_a = desc.trans_a;
  key.trans_b = desc.trans_b;
  key.pointwise = desc.epilogue;
  key.shape_bucket = BucketForProblem(desc);
  key.deterministic = true;
  return key;
}

DispatchKey BuildReduceDispatchKey(
    const ::mklibTensorDesc& input_desc,
    const mklibReduceDesc_t& desc) {
  DispatchKey key;
  key.operation = OperationKind::kReduce;
  key.a_type = input_desc.dtype;
  key.c_type = input_desc.dtype;
  key.compute_type = input_desc.dtype;
  key.rank = input_desc.rank;
  key.layout = runtime::IsContiguous(input_desc) ? LayoutKind::kContiguous : LayoutKind::kStrided;

  runtime::ReductionGeometry geometry;
  if (runtime::MakeReductionGeometry(input_desc, desc.axis, &geometry)) {
    key.reduction_extent = geometry.reduce_size;
    key.inner_extent = geometry.inner_size;
    key.shape_bucket =
        BucketForElements(geometry.outer_size * geometry.reduce_size * geometry.inner_size);
    if (geometry.inner_size == 1) {
      key.axis_role = AxisRole::kInner;
    } else if (geometry.outer_size == 1) {
      key.axis_role = AxisRole::kOuter;
    } else {
      key.axis_role = AxisRole::kMiddle;
    }
  }

  return key;
}

DispatchKey BuildConv2dForwardDispatchKey(
    const ::mklibTensorDesc& input_desc,
    const ::mklibTensorDesc& filter_desc,
    const mklibConv2dDesc_t& desc) {
  DispatchKey key;
  key.operation = OperationKind::kConv2dForward;
  key.a_type = input_desc.dtype;
  key.b_type = filter_desc.dtype;
  key.c_type = input_desc.dtype;
  key.compute_type = input_desc.dtype;
  key.rank = input_desc.rank;
  key.layout =
      runtime::IsContiguous(input_desc) && runtime::IsContiguous(filter_desc)
          ? LayoutKind::kContiguous
          : LayoutKind::kStrided;

  runtime::Conv2dGeometry geometry;
  if (runtime::MakeConv2dGeometry(input_desc, filter_desc, desc, &geometry)) {
    key.reduction_extent = geometry.in_channels * geometry.kernel_h * geometry.kernel_w;
    key.inner_extent = geometry.out_h * geometry.out_w;
    key.shape_bucket = BucketForElements(
        geometry.batch * geometry.out_channels * geometry.out_h * geometry.out_w);
  }

  return key;
}

std::string ToString(const DispatchKey& key) {
  switch (key.operation) {
    case OperationKind::kGemm:
      return "gemm:" + std::string(DataTypeString(key.compute_type)) + ":" +
             TransposeString(key.trans_a) + TransposeString(key.trans_b) + ":" +
             PointwiseString(key.pointwise) + ":" + BucketString(key.shape_bucket);
    case OperationKind::kReduce:
      return "reduce:" + std::string(DataTypeString(key.compute_type)) + ":" +
             LayoutString(key.layout) + ":" + AxisRoleString(key.axis_role) + ":" +
             BucketString(key.shape_bucket);
    case OperationKind::kConv2dForward:
      return "conv2d_fwd:" + std::string(DataTypeString(key.compute_type)) + ":" +
             LayoutString(key.layout) + ":" + BucketString(key.shape_bucket);
  }
  return "unknown";
}

}  // namespace mklib::planner
