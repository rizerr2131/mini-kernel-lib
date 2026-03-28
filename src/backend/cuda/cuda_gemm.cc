#include "backend/backend.h"

#include <algorithm>

namespace {

constexpr int64_t kBlockedTileRows = 64;
constexpr int64_t kBlockedTileCols = 64;
constexpr int64_t kBlockedTileDepth = 64;

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

float ApplyPointwise(float value, mklibPointwiseMode_t pointwise) {
  switch (pointwise) {
    case MKLIB_POINTWISE_MODE_IDENTITY:
      return value;
    case MKLIB_POINTWISE_MODE_RELU:
      return value > 0.0f ? value : 0.0f;
  }
  return value;
}

size_t BlockedWorkspaceBytes(const mklibGemmDesc_t& desc) {
  if (desc.m == 0 || desc.n == 0 || desc.k == 0) {
    return 0;
  }
  return static_cast<size_t>(kBlockedTileCols * kBlockedTileDepth * sizeof(float));
}

mklibStatus_t LaunchReferenceF32DirectGemm(
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
      c_data[row * desc.ldc + col] = ApplyPointwise(sum, desc.epilogue);
    }
  }

  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t LaunchReferenceF32BlockedGemm(
    const mklibGemmDesc_t& desc,
    const void* a,
    const void* b,
    void* c,
    void* workspace,
    size_t workspace_size) {
  const size_t required_workspace = BlockedWorkspaceBytes(desc);
  if (workspace_size < required_workspace) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (required_workspace > 0 && workspace == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  const auto* a_data = static_cast<const float*>(a);
  const auto* b_data = static_cast<const float*>(b);
  auto* c_data = static_cast<float*>(c);
  auto* packed_b = static_cast<float*>(workspace);

  for (int64_t row_tile = 0; row_tile < desc.m; row_tile += kBlockedTileRows) {
    const int64_t tile_rows = std::min(kBlockedTileRows, desc.m - row_tile);
    for (int64_t col_tile = 0; col_tile < desc.n; col_tile += kBlockedTileCols) {
      const int64_t tile_cols = std::min(kBlockedTileCols, desc.n - col_tile);

      for (int64_t row = 0; row < tile_rows; ++row) {
        float* c_row = c_data + (row_tile + row) * desc.ldc + col_tile;
        for (int64_t col = 0; col < tile_cols; ++col) {
          c_row[col] = 0.0f;
        }
      }

      for (int64_t depth_tile = 0; depth_tile < desc.k; depth_tile += kBlockedTileDepth) {
        const int64_t tile_depth = std::min(kBlockedTileDepth, desc.k - depth_tile);

        for (int64_t depth = 0; depth < tile_depth; ++depth) {
          for (int64_t col = 0; col < tile_cols; ++col) {
            packed_b[depth * kBlockedTileCols + col] =
                LoadB(b_data, desc, depth_tile + depth, col_tile + col);
          }
        }

        for (int64_t row = 0; row < tile_rows; ++row) {
          float* c_row = c_data + (row_tile + row) * desc.ldc + col_tile;
          for (int64_t depth = 0; depth < tile_depth; ++depth) {
            const float a_value = LoadA(a_data, desc, row_tile + row, depth_tile + depth);
            const float* packed_b_row = packed_b + depth * kBlockedTileCols;
            for (int64_t col = 0; col < tile_cols; ++col) {
              c_row[col] += a_value * packed_b_row[col];
            }
          }
        }
      }

      for (int64_t row = 0; row < tile_rows; ++row) {
        float* c_row = c_data + (row_tile + row) * desc.ldc + col_tile;
        for (int64_t col = 0; col < tile_cols; ++col) {
          c_row[col] = ApplyPointwise(c_row[col], desc.epilogue);
        }
      }
    }
  }

  return MKLIB_STATUS_SUCCESS;
}

}  // namespace

namespace mklib::backend {

size_t GetGemmWorkspaceSize(
    const planner::DispatchKey& key,
    const registry::KernelRecord& kernel,
    const mklibGemmDesc_t& desc) {
  (void)key;

  switch (kernel.kind) {
    case registry::KernelKind::kReferenceF32Direct:
      return 0;
    case registry::KernelKind::kReferenceF32Blocked:
      return BlockedWorkspaceBytes(desc);
    case registry::KernelKind::kReduceF32InnerContiguous:
    case registry::KernelKind::kReduceF32GenericContiguous:
    case registry::KernelKind::kConv2dF32Direct:
      return 0;
  }
  return 0;
}

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
    case registry::KernelKind::kReferenceF32Direct:
      return LaunchReferenceF32DirectGemm(desc, a, b, c);
    case registry::KernelKind::kReferenceF32Blocked:
      return LaunchReferenceF32BlockedGemm(desc, a, b, c, workspace, workspace_size);
    case registry::KernelKind::kReduceF32InnerContiguous:
    case registry::KernelKind::kReduceF32GenericContiguous:
    case registry::KernelKind::kConv2dF32Direct:
      break;
  }

  return MKLIB_STATUS_INTERNAL_ERROR;
}

}  // namespace mklib::backend
