#include "mklib/gemm.h"

#include "backend/backend.h"
#include "planner/dispatch_key.h"
#include "registry/kernel_registry.h"

namespace {

bool IsValidTranspose(mklibTranspose_t transpose) {
  switch (transpose) {
    case MKLIB_OP_N:
    case MKLIB_OP_T:
      return true;
  }
  return false;
}

bool IsValidDataType(mklibDataType_t dtype) {
  switch (dtype) {
    case MKLIB_DATA_TYPE_FLOAT32:
    case MKLIB_DATA_TYPE_FLOAT16:
    case MKLIB_DATA_TYPE_BFLOAT16:
      return true;
    case MKLIB_DATA_TYPE_INVALID:
      return false;
  }
  return false;
}

int64_t MinimumLeadingDimensionForA(const mklibGemmDesc_t& desc) {
  return desc.trans_a == MKLIB_OP_N ? desc.k : desc.m;
}

int64_t MinimumLeadingDimensionForB(const mklibGemmDesc_t& desc) {
  return desc.trans_b == MKLIB_OP_N ? desc.n : desc.k;
}

mklibStatus_t ValidateGemmDesc(const mklibGemmDesc_t* desc) {
  if (desc == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (!IsValidDataType(desc->a_type) ||
      !IsValidDataType(desc->b_type) ||
      !IsValidDataType(desc->c_type) ||
      !IsValidDataType(desc->compute_type)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (!IsValidTranspose(desc->trans_a) || !IsValidTranspose(desc->trans_b)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (desc->m < 0 || desc->n < 0 || desc->k < 0) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (desc->lda <= 0 || desc->ldb <= 0 || desc->ldc <= 0) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (desc->lda < MinimumLeadingDimensionForA(*desc) ||
      desc->ldb < MinimumLeadingDimensionForB(*desc) ||
      desc->ldc < desc->n) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  return MKLIB_STATUS_SUCCESS;
}

}  // namespace

mklibStatus_t mklibGetGemmWorkspaceSize(
    mklibHandle_t handle,
    const mklibGemmDesc_t* desc,
    size_t* bytes_out) {
  if (handle == nullptr || bytes_out == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  const mklibStatus_t status = ValidateGemmDesc(desc);
  if (status != MKLIB_STATUS_SUCCESS) {
    return status;
  }

  const auto key = mklib::planner::BuildGemmDispatchKey(*desc);
  const auto* kernel = mklib::registry::SelectGemmKernel(key);
  if (kernel == nullptr) {
    return MKLIB_STATUS_NOT_SUPPORTED;
  }

  *bytes_out = 0;
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t mklibGemm(
    mklibHandle_t handle,
    const mklibGemmDesc_t* desc,
    const void* a,
    const void* b,
    void* c,
    void* workspace,
    size_t workspace_size) {
  if (handle == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  const mklibStatus_t status = ValidateGemmDesc(desc);
  if (status != MKLIB_STATUS_SUCCESS) {
    return status;
  }

  const auto key = mklib::planner::BuildGemmDispatchKey(*desc);
  const auto* kernel = mklib::registry::SelectGemmKernel(key);
  if (kernel == nullptr) {
    return MKLIB_STATUS_NOT_SUPPORTED;
  }

  if (desc->m == 0 || desc->n == 0 || desc->k == 0) {
    (void)a;
    (void)b;
    (void)c;
    (void)workspace;
    (void)workspace_size;
    return MKLIB_STATUS_SUCCESS;
  }

  if (a == nullptr || b == nullptr || c == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  return mklib::backend::LaunchGemm(
      *handle,
      key,
      *kernel,
      *desc,
      a,
      b,
      c,
      workspace,
      workspace_size);
}
