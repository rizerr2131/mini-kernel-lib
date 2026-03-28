#include "mklib/handle.h"

#include <new>

#include "runtime/handle_state.h"

namespace {

bool IsValidAutotuneMode(mklibAutotuneMode_t mode) {
  switch (mode) {
    case MKLIB_AUTOTUNE_OFF:
    case MKLIB_AUTOTUNE_ON:
      return true;
  }
  return false;
}

}  // namespace

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

mklibStatus_t mklibSetAutotuneMode(mklibHandle_t handle, mklibAutotuneMode_t mode) {
  if (handle == nullptr || !IsValidAutotuneMode(mode)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  handle->autotune_mode = mode;
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t mklibGetAutotuneMode(
    mklibHandle_t handle,
    mklibAutotuneMode_t* mode_out) {
  if (handle == nullptr || mode_out == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  *mode_out = handle->autotune_mode;
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t mklibClearAutotuneCache(mklibHandle_t handle) {
  if (handle == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  handle->gemm_autotune_cache.clear();
  return MKLIB_STATUS_SUCCESS;
}
