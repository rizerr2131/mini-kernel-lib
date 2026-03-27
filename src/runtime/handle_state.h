#ifndef MKLIB_RUNTIME_HANDLE_STATE_H_
#define MKLIB_RUNTIME_HANDLE_STATE_H_

#include "mklib/handle.h"

struct mklibHandle {
  void* stream = nullptr;
  const char* backend_name = "cuda-stub";
};

#endif  // MKLIB_RUNTIME_HANDLE_STATE_H_
