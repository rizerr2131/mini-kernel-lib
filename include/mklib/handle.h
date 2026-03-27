#ifndef MKLIB_HANDLE_H_
#define MKLIB_HANDLE_H_

#include "mklib/export.h"
#include "mklib/status.h"

MKLIB_EXTERN_C_BEGIN

typedef struct mklibHandle* mklibHandle_t;

MKLIB_API mklibStatus_t mklibCreate(mklibHandle_t* out);
MKLIB_API mklibStatus_t mklibDestroy(mklibHandle_t handle);
MKLIB_API mklibStatus_t mklibSetStream(mklibHandle_t handle, void* stream);
MKLIB_API mklibStatus_t mklibGetStream(mklibHandle_t handle, void** stream_out);

MKLIB_EXTERN_C_END

#endif  // MKLIB_HANDLE_H_
