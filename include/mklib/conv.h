#ifndef MKLIB_CONV_H_
#define MKLIB_CONV_H_

#include <stddef.h>
#include <stdint.h>

#include "mklib/export.h"
#include "mklib/handle.h"
#include "mklib/status.h"
#include "mklib/tensor_desc.h"

MKLIB_EXTERN_C_BEGIN

typedef struct mklibConv2dDesc {
  int64_t pad_h;
  int64_t pad_w;
  int64_t stride_h;
  int64_t stride_w;
  int64_t dilation_h;
  int64_t dilation_w;
} mklibConv2dDesc_t;

MKLIB_API mklibStatus_t mklibGetConv2dForwardWorkspaceSize(
    mklibHandle_t handle,
    const mklibConv2dDesc_t* desc,
    mklibTensorDesc_t input_desc,
    mklibTensorDesc_t filter_desc,
    mklibTensorDesc_t output_desc,
    size_t* bytes_out);

MKLIB_API mklibStatus_t mklibConv2dForward(
    mklibHandle_t handle,
    const mklibConv2dDesc_t* desc,
    mklibTensorDesc_t input_desc,
    const void* input,
    mklibTensorDesc_t filter_desc,
    const void* filter,
    mklibTensorDesc_t output_desc,
    void* output,
    void* workspace,
    size_t workspace_size);

MKLIB_EXTERN_C_END

#endif  // MKLIB_CONV_H_
