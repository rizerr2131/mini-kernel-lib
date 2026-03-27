#ifndef MKLIB_REGISTRY_KERNEL_REGISTRY_H_
#define MKLIB_REGISTRY_KERNEL_REGISTRY_H_

#include "planner/dispatch_key.h"

namespace mklib::registry {

struct KernelRecord {
  const char* name;
};

const KernelRecord* SelectGemmKernel(const planner::DispatchKey& key);

}  // namespace mklib::registry

#endif  // MKLIB_REGISTRY_KERNEL_REGISTRY_H_
