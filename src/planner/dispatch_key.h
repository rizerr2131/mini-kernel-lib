#ifndef MKLIB_PLANNER_DISPATCH_KEY_H_
#define MKLIB_PLANNER_DISPATCH_KEY_H_

#include <cstdint>
#include <string>

#include "mklib/conv.h"
#include "mklib/gemm.h"
#include "mklib/reduction.h"
#include "runtime/tensor_desc_state.h"

namespace mklib::planner {

enum class OperationKind {
  kGemm,
  kReduce,
  kConv2dForward,
};

enum class ShapeBucket {
  kSmall,
  kMedium,
  kLarge,
};

enum class LayoutKind {
  kContiguous,
  kStrided,
};

enum class AxisRole {
  kNone,
  kInner,
  kMiddle,
  kOuter,
};

struct DispatchKey {
  OperationKind operation = OperationKind::kGemm;
  mklibDataType_t a_type = MKLIB_DATA_TYPE_INVALID;
  mklibDataType_t b_type = MKLIB_DATA_TYPE_INVALID;
  mklibDataType_t c_type = MKLIB_DATA_TYPE_INVALID;
  mklibDataType_t compute_type = MKLIB_DATA_TYPE_INVALID;
  mklibTranspose_t trans_a = MKLIB_OP_N;
  mklibTranspose_t trans_b = MKLIB_OP_N;
  mklibPointwiseMode_t pointwise = MKLIB_POINTWISE_MODE_IDENTITY;
  int rank = 0;
  LayoutKind layout = LayoutKind::kContiguous;
  AxisRole axis_role = AxisRole::kNone;
  int64_t reduction_extent = 0;
  int64_t inner_extent = 0;
  ShapeBucket shape_bucket = ShapeBucket::kSmall;
  bool deterministic = true;
};

DispatchKey BuildGemmDispatchKey(const mklibGemmDesc_t& desc);
DispatchKey BuildReduceDispatchKey(const ::mklibTensorDesc& input_desc, const mklibReduceDesc_t& desc);
DispatchKey BuildConv2dForwardDispatchKey(
    const ::mklibTensorDesc& input_desc,
    const ::mklibTensorDesc& filter_desc,
    const mklibConv2dDesc_t& desc);
std::string ToString(const DispatchKey& key);

}  // namespace mklib::planner

#endif  // MKLIB_PLANNER_DISPATCH_KEY_H_
