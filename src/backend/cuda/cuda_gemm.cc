#include "backend/backend.h"

namespace mklib::backend {

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
  (void)kernel;
  (void)desc;
  (void)a;
  (void)b;
  (void)c;
  (void)workspace;
  (void)workspace_size;
  return MKLIB_STATUS_NOT_SUPPORTED;
}

}  // namespace mklib::backend
