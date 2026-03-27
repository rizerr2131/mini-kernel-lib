#include "mklib/status.h"

const char* mklibGetStatusString(mklibStatus_t status) {
  switch (status) {
    case MKLIB_STATUS_SUCCESS:
      return "success";
    case MKLIB_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case MKLIB_STATUS_BAD_STATE:
      return "bad state";
    case MKLIB_STATUS_NOT_SUPPORTED:
      return "not supported";
    case MKLIB_STATUS_OUT_OF_MEMORY:
      return "out of memory";
    case MKLIB_STATUS_INTERNAL_ERROR:
      return "internal error";
  }
  return "unknown status";
}
