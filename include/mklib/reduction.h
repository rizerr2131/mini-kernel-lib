#ifndef MKLIB_REDUCTION_H_
#define MKLIB_REDUCTION_H_

#include <stddef.h>

#include "mklib/export.h"
#include "mklib/handle.h"
#include "mklib/status.h"
#include "mklib/tensor_desc.h"

MKLIB_EXTERN_C_BEGIN

typedef enum mklibReduceOp {
  MKLIB_REDUCE_OP_SUM = 0
} mklibReduceOp_t;

typedef struct mklibReduceDesc {
  mklibReduceOp_t op;
  int axis;
  int keep_dim;
} mklibReduceDesc_t;

MKLIB_API mklibStatus_t mklibGetReduceWorkspaceSize(
    mklibHandle_t handle,
    mklibTensorDesc_t input_desc,
    mklibTensorDesc_t output_desc,
    const mklibReduceDesc_t* desc,
    size_t* bytes_out);

MKLIB_API mklibStatus_t mklibReduce(
    mklibHandle_t handle,
    mklibTensorDesc_t input_desc,
    const void* input,
    mklibTensorDesc_t output_desc,
    void* output,
    const mklibReduceDesc_t* desc,
    void* workspace,
    size_t workspace_size);

MKLIB_EXTERN_C_END

#endif  // MKLIB_REDUCTION_H_
