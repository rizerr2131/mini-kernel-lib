#ifndef MKLIB_HANDLE_H_
#define MKLIB_HANDLE_H_

#include "mklib/export.h"
#include "mklib/status.h"

MKLIB_EXTERN_C_BEGIN

typedef enum mklibAutotuneMode {
  MKLIB_AUTOTUNE_OFF = 0,
  MKLIB_AUTOTUNE_ON = 1
} mklibAutotuneMode_t;

typedef struct mklibHandle* mklibHandle_t;

MKLIB_API mklibStatus_t mklibCreate(mklibHandle_t* out);
MKLIB_API mklibStatus_t mklibDestroy(mklibHandle_t handle);
MKLIB_API mklibStatus_t mklibSetStream(mklibHandle_t handle, void* stream);
MKLIB_API mklibStatus_t mklibGetStream(mklibHandle_t handle, void** stream_out);
MKLIB_API mklibStatus_t mklibSetAutotuneMode(
    mklibHandle_t handle,
    mklibAutotuneMode_t mode);
MKLIB_API mklibStatus_t mklibGetAutotuneMode(
    mklibHandle_t handle,
    mklibAutotuneMode_t* mode_out);
MKLIB_API mklibStatus_t mklibClearAutotuneCache(mklibHandle_t handle);

MKLIB_EXTERN_C_END

#endif  // MKLIB_HANDLE_H_
