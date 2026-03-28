#ifndef MKLIB_RUNTIME_HANDLE_STATE_H_
#define MKLIB_RUNTIME_HANDLE_STATE_H_

#include <string>
#include <unordered_map>

#include "mklib/handle.h"
#include "registry/kernel_registry.h"

struct mklibHandle {
  void* stream = nullptr;
  const char* backend_name = "cuda-stub";
  mklibAutotuneMode_t autotune_mode = MKLIB_AUTOTUNE_OFF;
  std::unordered_map<std::string, mklib::registry::KernelKind> gemm_autotune_cache;
};

#endif  // MKLIB_RUNTIME_HANDLE_STATE_H_
