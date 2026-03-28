#include "registry/kernel_registry.h"

namespace mklib::registry {
namespace {

constexpr KernelRecord kReferenceF32DirectKernel = {
    .name = "reference_f32_direct_gemm",
    .kind = KernelKind::kReferenceF32Direct,
};

constexpr KernelRecord kReferenceF32BlockedKernel = {
    .name = "reference_f32_blocked_gemm",
    .kind = KernelKind::kReferenceF32Blocked,
};

constexpr KernelRecord kReduceF32InnerContiguousKernel = {
    .name = "reference_f32_inner_reduce",
    .kind = KernelKind::kReduceF32InnerContiguous,
};

constexpr KernelRecord kReduceF32GenericContiguousKernel = {
    .name = "reference_f32_generic_reduce",
    .kind = KernelKind::kReduceF32GenericContiguous,
};

constexpr KernelRecord kConv2dF32DirectKernel = {
    .name = "reference_f32_direct_conv2d_fwd",
    .kind = KernelKind::kConv2dF32Direct,
};

bool SupportsReferenceF32Gemm(const planner::DispatchKey& key) {
  return key.operation == planner::OperationKind::kGemm &&
         key.a_type == MKLIB_DATA_TYPE_FLOAT32 &&
         key.b_type == MKLIB_DATA_TYPE_FLOAT32 &&
         key.c_type == MKLIB_DATA_TYPE_FLOAT32 &&
         key.compute_type == MKLIB_DATA_TYPE_FLOAT32;
}

std::span<const KernelRecord> SupportedReferenceF32Candidates() {
  static constexpr KernelRecord kCandidates[] = {
      kReferenceF32DirectKernel,
      kReferenceF32BlockedKernel,
  };
  return std::span<const KernelRecord>(kCandidates);
}

bool SupportsReferenceF32Reduce(const planner::DispatchKey& key) {
  return key.operation == planner::OperationKind::kReduce &&
         key.compute_type == MKLIB_DATA_TYPE_FLOAT32 &&
         key.layout == planner::LayoutKind::kContiguous;
}

bool SupportsReferenceF32Conv2d(const planner::DispatchKey& key) {
  return key.operation == planner::OperationKind::kConv2dForward &&
         key.a_type == MKLIB_DATA_TYPE_FLOAT32 &&
         key.b_type == MKLIB_DATA_TYPE_FLOAT32 &&
         key.compute_type == MKLIB_DATA_TYPE_FLOAT32 &&
         key.layout == planner::LayoutKind::kContiguous;
}

}  // namespace

std::span<const KernelRecord> GetGemmKernelCandidates(const planner::DispatchKey& key) {
  if (!SupportsReferenceF32Gemm(key)) {
    return {};
  }
  return SupportedReferenceF32Candidates();
}

const KernelRecord* SelectGemmKernel(const planner::DispatchKey& key) {
  if (!SupportsReferenceF32Gemm(key)) {
    return nullptr;
  }

  switch (key.shape_bucket) {
    case planner::ShapeBucket::kSmall:
      return &kReferenceF32DirectKernel;
    case planner::ShapeBucket::kMedium:
    case planner::ShapeBucket::kLarge:
      return &kReferenceF32BlockedKernel;
  }
  return nullptr;
}

const KernelRecord* SelectReduceKernel(const planner::DispatchKey& key) {
  if (!SupportsReferenceF32Reduce(key)) {
    return nullptr;
  }

  switch (key.axis_role) {
    case planner::AxisRole::kInner:
      return &kReduceF32InnerContiguousKernel;
    case planner::AxisRole::kMiddle:
    case planner::AxisRole::kOuter:
      return &kReduceF32GenericContiguousKernel;
    case planner::AxisRole::kNone:
      return nullptr;
  }
  return nullptr;
}

const KernelRecord* SelectConv2dForwardKernel(const planner::DispatchKey& key) {
  if (!SupportsReferenceF32Conv2d(key)) {
    return nullptr;
  }
  return &kConv2dF32DirectKernel;
}

}  // namespace mklib::registry
