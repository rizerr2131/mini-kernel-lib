#ifndef MKLIB_PLANNER_DISPATCH_KEY_H_
#define MKLIB_PLANNER_DISPATCH_KEY_H_

#include <string>

#include "mklib/gemm.h"

namespace mklib::planner {

enum class OperationKind {
  kGemm,
};

enum class ShapeBucket {
  kSmall,
  kMedium,
  kLarge,
};

struct DispatchKey {
  OperationKind operation = OperationKind::kGemm;
  mklibDataType_t compute_type = MKLIB_DATA_TYPE_INVALID;
  mklibTranspose_t trans_a = MKLIB_OP_N;
  mklibTranspose_t trans_b = MKLIB_OP_N;
  ShapeBucket shape_bucket = ShapeBucket::kSmall;
  bool deterministic = true;
};

DispatchKey BuildGemmDispatchKey(const mklibGemmDesc_t& desc);
std::string ToString(const DispatchKey& key);

}  // namespace mklib::planner

#endif  // MKLIB_PLANNER_DISPATCH_KEY_H_
