#include "mklib/gemm.h"

#include <algorithm>
#include <chrono>
#include <string>

#include "backend/backend.h"
#include "planner/dispatch_key.h"
#include "registry/kernel_registry.h"
#include "runtime/handle_state.h"

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

bool IsValidPointwiseMode(mklibPointwiseMode_t mode) {
  switch (mode) {
    case MKLIB_POINTWISE_MODE_IDENTITY:
    case MKLIB_POINTWISE_MODE_RELU:
      return true;
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
  if (!IsValidPointwiseMode(desc->epilogue)) {
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

bool ShouldUseAutotune(const ::mklibHandle& handle, const mklib::planner::DispatchKey& key) {
  return handle.autotune_mode == MKLIB_AUTOTUNE_ON &&
         key.shape_bucket != mklib::planner::ShapeBucket::kSmall;
}

std::string MakeAutotuneCacheKey(const mklibGemmDesc_t& desc) {
  return std::to_string(desc.m) + ":" + std::to_string(desc.n) + ":" +
         std::to_string(desc.k) + ":" + std::to_string(desc.lda) + ":" +
         std::to_string(desc.ldb) + ":" + std::to_string(desc.ldc) + ":" +
         std::to_string(static_cast<int>(desc.trans_a)) + ":" +
         std::to_string(static_cast<int>(desc.trans_b)) + ":" +
         std::to_string(static_cast<int>(desc.epilogue));
}

const mklib::registry::KernelRecord* FindKernelByKind(
    const mklib::planner::DispatchKey& key,
    mklib::registry::KernelKind kind) {
  for (const auto& candidate : mklib::registry::GetGemmKernelCandidates(key)) {
    if (candidate.kind == kind) {
      return &candidate;
    }
  }
  return nullptr;
}

const mklib::registry::KernelRecord* ResolveCachedOrHeuristicKernel(
    mklibHandle_t handle,
    const mklib::planner::DispatchKey& key,
    const mklibGemmDesc_t& desc) {
  if (!ShouldUseAutotune(*handle, key)) {
    return mklib::registry::SelectGemmKernel(key);
  }

  const auto cache_it = handle->gemm_autotune_cache.find(MakeAutotuneCacheKey(desc));
  if (cache_it == handle->gemm_autotune_cache.end()) {
    return nullptr;
  }

  const auto* cached_kernel = FindKernelByKind(key, cache_it->second);
  if (cached_kernel == nullptr) {
    handle->gemm_autotune_cache.erase(cache_it);
  }
  return cached_kernel;
}

size_t GetRequiredWorkspaceBytes(
    const mklib::planner::DispatchKey& key,
    const mklibGemmDesc_t& desc,
    const mklib::registry::KernelRecord* kernel) {
  if (kernel != nullptr) {
    return mklib::backend::GetGemmWorkspaceSize(key, *kernel, desc);
  }

  size_t workspace_bytes = 0;
  for (const auto& candidate : mklib::registry::GetGemmKernelCandidates(key)) {
    workspace_bytes = std::max(
        workspace_bytes,
        mklib::backend::GetGemmWorkspaceSize(key, candidate, desc));
  }
  return workspace_bytes;
}

mklibStatus_t AutotuneKernelSelection(
    mklibHandle_t handle,
    const mklib::planner::DispatchKey& key,
    const mklibGemmDesc_t& desc,
    const void* a,
    const void* b,
    void* c,
    void* workspace,
    size_t workspace_size,
    const mklib::registry::KernelRecord** kernel_out) {
  const auto candidates = mklib::registry::GetGemmKernelCandidates(key);
  if (candidates.empty()) {
    return MKLIB_STATUS_NOT_SUPPORTED;
  }

  const mklib::registry::KernelRecord* best_kernel = nullptr;
  double best_ns = 0.0;
  for (const auto& candidate : candidates) {
    const size_t required_workspace =
        mklib::backend::GetGemmWorkspaceSize(key, candidate, desc);
    if (workspace_size < required_workspace) {
      return MKLIB_STATUS_INVALID_ARGUMENT;
    }

    const auto warmup_status =
        mklib::backend::LaunchGemm(*handle, key, candidate, desc, a, b, c, workspace, workspace_size);
    if (warmup_status != MKLIB_STATUS_SUCCESS) {
      return warmup_status;
    }

    const auto start = std::chrono::steady_clock::now();
    const auto timed_status =
        mklib::backend::LaunchGemm(*handle, key, candidate, desc, a, b, c, workspace, workspace_size);
    const auto stop = std::chrono::steady_clock::now();
    if (timed_status != MKLIB_STATUS_SUCCESS) {
      return timed_status;
    }

    const double elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    if (best_kernel == nullptr || elapsed_ns < best_ns) {
      best_kernel = &candidate;
      best_ns = elapsed_ns;
    }
  }

  if (best_kernel == nullptr) {
    return MKLIB_STATUS_INTERNAL_ERROR;
  }

  handle->gemm_autotune_cache[MakeAutotuneCacheKey(desc)] = best_kernel->kind;
  *kernel_out = best_kernel;
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
  const auto* kernel = ResolveCachedOrHeuristicKernel(handle, key, *desc);
  if (kernel == nullptr && mklib::registry::GetGemmKernelCandidates(key).empty()) {
    return MKLIB_STATUS_NOT_SUPPORTED;
  }

  *bytes_out = GetRequiredWorkspaceBytes(key, *desc, kernel);
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
  const auto* kernel = ResolveCachedOrHeuristicKernel(handle, key, *desc);
  if (kernel == nullptr && mklib::registry::GetGemmKernelCandidates(key).empty()) {
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

  const size_t required_workspace = GetRequiredWorkspaceBytes(key, *desc, kernel);
  if (workspace_size < required_workspace) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (required_workspace > 0 && workspace == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  if (kernel == nullptr) {
    const mklibStatus_t autotune_status = AutotuneKernelSelection(
        handle,
        key,
        *desc,
        a,
        b,
        c,
        workspace,
        workspace_size,
        &kernel);
    if (autotune_status != MKLIB_STATUS_SUCCESS) {
      return autotune_status;
    }
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
