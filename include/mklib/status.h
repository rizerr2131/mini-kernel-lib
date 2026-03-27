#ifndef MKLIB_STATUS_H_
#define MKLIB_STATUS_H_

#include "mklib/export.h"

MKLIB_EXTERN_C_BEGIN

typedef enum mklibStatus {
  MKLIB_STATUS_SUCCESS = 0,
  MKLIB_STATUS_INVALID_ARGUMENT = 1,
  MKLIB_STATUS_BAD_STATE = 2,
  MKLIB_STATUS_NOT_SUPPORTED = 3,
  MKLIB_STATUS_OUT_OF_MEMORY = 4,
  MKLIB_STATUS_INTERNAL_ERROR = 5
} mklibStatus_t;

MKLIB_API const char* mklibGetStatusString(mklibStatus_t status);

MKLIB_EXTERN_C_END

#endif  // MKLIB_STATUS_H_
