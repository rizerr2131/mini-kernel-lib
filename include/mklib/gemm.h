#ifndef MKLIB_GEMM_H_
#define MKLIB_GEMM_H_

#include <stddef.h>
#include <stdint.h>

#include "mklib/export.h"
#include "mklib/handle.h"
#include "mklib/status.h"
#include "mklib/tensor_desc.h"

MKLIB_EXTERN_C_BEGIN

typedef enum mklibTranspose {
  MKLIB_OP_N = 0,
  MKLIB_OP_T = 1
} mklibTranspose_t;

typedef struct mklibGemmDesc {
  mklibDataType_t a_type;
  mklibDataType_t b_type;
  mklibDataType_t c_type;
  mklibDataType_t compute_type;
  mklibTranspose_t trans_a;
  mklibTranspose_t trans_b;
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t lda;
  int64_t ldb;
  int64_t ldc;
} mklibGemmDesc_t;

/*
 * The current implementation provides a row-major FP32 GEMM path.
 * lda, ldb, and ldc are row strides for the underlying matrix storage.
 */

MKLIB_API mklibStatus_t mklibGetGemmWorkspaceSize(
    mklibHandle_t handle,
    const mklibGemmDesc_t* desc,
    size_t* bytes_out);

MKLIB_API mklibStatus_t mklibGemm(
    mklibHandle_t handle,
    const mklibGemmDesc_t* desc,
    const void* a,
    const void* b,
    void* c,
    void* workspace,
    size_t workspace_size);

MKLIB_EXTERN_C_END

#endif  // MKLIB_GEMM_H_
