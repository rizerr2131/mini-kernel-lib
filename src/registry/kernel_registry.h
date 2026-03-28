#ifndef MKLIB_REGISTRY_KERNEL_REGISTRY_H_
#define MKLIB_REGISTRY_KERNEL_REGISTRY_H_

#include "planner/dispatch_key.h"

namespace mklib::registry {

enum class KernelKind {
  kReferenceF32,
};

struct KernelRecord {
  const char* name;
  KernelKind kind;
};

const KernelRecord* SelectGemmKernel(const planner::DispatchKey& key);

}  // namespace mklib::registry

#endif  // MKLIB_REGISTRY_KERNEL_REGISTRY_H_
