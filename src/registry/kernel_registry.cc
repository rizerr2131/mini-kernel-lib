#include "registry/kernel_registry.h"

namespace mklib::registry {

const KernelRecord* SelectGemmKernel(const planner::DispatchKey& key) {
  (void)key;
  return nullptr;
}

}  // namespace mklib::registry
