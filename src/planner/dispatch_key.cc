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

}  // namespace

DispatchKey BuildGemmDispatchKey(const mklibGemmDesc_t& desc) {
  DispatchKey key;
  key.operation = OperationKind::kGemm;
  key.compute_type = desc.compute_type;
  key.trans_a = desc.trans_a;
  key.trans_b = desc.trans_b;
  key.shape_bucket = BucketForProblem(desc);
  key.deterministic = true;
  return key;
}

std::string ToString(const DispatchKey& key) {
  return "gemm:" + std::string(BucketString(key.shape_bucket));
}

}  // namespace mklib::planner
