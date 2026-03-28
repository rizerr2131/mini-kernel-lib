#include "planner/dispatch_key.h"

#include <algorithm>
#include <string>

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
  key.shape_bucket = BucketForProblem(desc);
  key.deterministic = true;
  return key;
}

std::string ToString(const DispatchKey& key) {
  return "gemm:" + std::string(DataTypeString(key.compute_type)) + ":" +
         TransposeString(key.trans_a) + TransposeString(key.trans_b) + ":" +
         BucketString(key.shape_bucket);
}

}  // namespace mklib::planner
