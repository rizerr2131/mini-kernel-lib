#include "mklib/handle.h"

#include <new>

#include "runtime/handle_state.h"

mklibStatus_t mklibCreate(mklibHandle_t* out) {
  if (out == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  auto* handle = new (std::nothrow) mklibHandle;
  if (handle == nullptr) {
    return MKLIB_STATUS_OUT_OF_MEMORY;
  }

  *out = handle;
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t mklibDestroy(mklibHandle_t handle) {
  if (handle == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  delete handle;
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t mklibSetStream(mklibHandle_t handle, void* stream) {
  if (handle == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  handle->stream = stream;
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t mklibGetStream(mklibHandle_t handle, void** stream_out) {
  if (handle == nullptr || stream_out == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  *stream_out = handle->stream;
  return MKLIB_STATUS_SUCCESS;
}
