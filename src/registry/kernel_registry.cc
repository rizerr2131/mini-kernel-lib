#include "registry/kernel_registry.h"

namespace mklib::registry {
namespace {

constexpr KernelRecord kReferenceF32Kernel = {
    .name = "reference_f32_gemm",
    .kind = KernelKind::kReferenceF32,
};

bool SupportsReferenceF32(const planner::DispatchKey& key) {
  return key.operation == planner::OperationKind::kGemm &&
         key.a_type == MKLIB_DATA_TYPE_FLOAT32 &&
         key.b_type == MKLIB_DATA_TYPE_FLOAT32 &&
         key.c_type == MKLIB_DATA_TYPE_FLOAT32 &&
         key.compute_type == MKLIB_DATA_TYPE_FLOAT32;
}

}  // namespace

const KernelRecord* SelectGemmKernel(const planner::DispatchKey& key) {
  if (SupportsReferenceF32(key)) {
    return &kReferenceF32Kernel;
  }
  return nullptr;
}

}  // namespace mklib::registry
