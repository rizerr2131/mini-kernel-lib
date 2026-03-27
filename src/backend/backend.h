#ifndef MKLIB_BACKEND_BACKEND_H_
#define MKLIB_BACKEND_BACKEND_H_

#include <stddef.h>

#include "mklib/gemm.h"
#include "mklib/status.h"
#include "planner/dispatch_key.h"
#include "registry/kernel_registry.h"
#include "runtime/handle_state.h"

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
    size_t workspace_size);

}  // namespace mklib::backend

#endif  // MKLIB_BACKEND_BACKEND_H_
