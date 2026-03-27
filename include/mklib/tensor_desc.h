#ifndef MKLIB_TENSOR_DESC_H_
#define MKLIB_TENSOR_DESC_H_

#include <stddef.h>
#include <stdint.h>

#include "mklib/export.h"
#include "mklib/status.h"

MKLIB_EXTERN_C_BEGIN

typedef enum mklibDataType {
  MKLIB_DATA_TYPE_INVALID = 0,
  MKLIB_DATA_TYPE_FLOAT32 = 1,
  MKLIB_DATA_TYPE_FLOAT16 = 2,
  MKLIB_DATA_TYPE_BFLOAT16 = 3
} mklibDataType_t;

typedef struct mklibTensorDesc* mklibTensorDesc_t;

MKLIB_API mklibStatus_t mklibCreateTensorDesc(mklibTensorDesc_t* out);
MKLIB_API mklibStatus_t mklibDestroyTensorDesc(mklibTensorDesc_t desc);
MKLIB_API mklibStatus_t mklibSetTensorDesc(
    mklibTensorDesc_t desc,
    mklibDataType_t dtype,
    int rank,
    const int64_t* sizes,
    const int64_t* strides);
MKLIB_API mklibStatus_t mklibGetTensorDataType(
    mklibTensorDesc_t desc,
    mklibDataType_t* dtype_out);
MKLIB_API mklibStatus_t mklibGetTensorRank(
    mklibTensorDesc_t desc,
    int* rank_out);
MKLIB_API mklibStatus_t mklibGetTensorSizes(
    mklibTensorDesc_t desc,
    size_t capacity,
    int64_t* sizes_out);
MKLIB_API mklibStatus_t mklibGetTensorStrides(
    mklibTensorDesc_t desc,
    size_t capacity,
    int64_t* strides_out);

MKLIB_EXTERN_C_END

#endif  // MKLIB_TENSOR_DESC_H_
