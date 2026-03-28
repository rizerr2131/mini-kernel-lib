#include "backend/backend.h"

namespace {

float LoadA(const float* a, const mklibGemmDesc_t& desc, int64_t row, int64_t depth) {
  if (desc.trans_a == MKLIB_OP_N) {
    return a[row * desc.lda + depth];
  }
  return a[depth * desc.lda + row];
}

float LoadB(const float* b, const mklibGemmDesc_t& desc, int64_t depth, int64_t col) {
  if (desc.trans_b == MKLIB_OP_N) {
    return b[depth * desc.ldb + col];
  }
  return b[col * desc.ldb + depth];
}

mklibStatus_t LaunchReferenceF32Gemm(
    const mklibGemmDesc_t& desc,
    const void* a,
    const void* b,
    void* c) {
  const auto* a_data = static_cast<const float*>(a);
  const auto* b_data = static_cast<const float*>(b);
  auto* c_data = static_cast<float*>(c);

  for (int64_t row = 0; row < desc.m; ++row) {
    for (int64_t col = 0; col < desc.n; ++col) {
      float sum = 0.0f;
      for (int64_t depth = 0; depth < desc.k; ++depth) {
        sum += LoadA(a_data, desc, row, depth) * LoadB(b_data, desc, depth, col);
      }
      c_data[row * desc.ldc + col] = sum;
    }
  }

  return MKLIB_STATUS_SUCCESS;
}

}  // namespace

namespace mklib::backend {

mklibStatus_t LaunchGemm(
    const ::mklibHandle& handle,
    const planner::DispatchKey& key,
    const registry::KernelRecord& kernel,
    const mklibGemmDesc_t& desc,
    const void* a,
    const void* b,
    void* c,
    void* workspace,
    size_t workspace_size) {
  (void)handle;
  (void)key;
  (void)workspace;
  (void)workspace_size;

  switch (kernel.kind) {
    case registry::KernelKind::kReferenceF32:
      return LaunchReferenceF32Gemm(desc, a, b, c);
  }

  return MKLIB_STATUS_INTERNAL_ERROR;
}

}  // namespace mklib::backend
