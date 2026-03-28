#ifndef MKLIB_REGISTRY_KERNEL_REGISTRY_H_
#define MKLIB_REGISTRY_KERNEL_REGISTRY_H_

#include <span>

#include "planner/dispatch_key.h"

namespace mklib::registry {

enum class KernelKind {
  kReferenceF32Direct,
  kReferenceF32Blocked,
  kReduceF32InnerContiguous,
  kReduceF32GenericContiguous,
  kConv2dF32Direct,
};

struct KernelRecord {
  const char* name;
  KernelKind kind;
};

std::span<const KernelRecord> GetGemmKernelCandidates(const planner::DispatchKey& key);
const KernelRecord* SelectGemmKernel(const planner::DispatchKey& key);
const KernelRecord* SelectReduceKernel(const planner::DispatchKey& key);
const KernelRecord* SelectConv2dForwardKernel(const planner::DispatchKey& key);

}  // namespace mklib::registry

#endif  // MKLIB_REGISTRY_KERNEL_REGISTRY_H_
