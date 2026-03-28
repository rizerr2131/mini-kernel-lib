#include "mklib/reduction.h"

#include "backend/backend.h"
#include "planner/dispatch_key.h"
#include "registry/kernel_registry.h"
#include "runtime/tensor_utils.h"

namespace {

bool IsValidDataType(mklibDataType_t dtype) {
  switch (dtype) {
    case MKLIB_DATA_TYPE_FLOAT32:
    case MKLIB_DATA_TYPE_FLOAT16:
    case MKLIB_DATA_TYPE_BFLOAT16:
      return true;
    case MKLIB_DATA_TYPE_INVALID:
      return false;
  }
  return false;
}

bool IsValidReduceOp(mklibReduceOp_t op) {
  switch (op) {
    case MKLIB_REDUCE_OP_SUM:
      return true;
  }
  return false;
}

mklibStatus_t CheckInitialized(mklibTensorDesc_t desc) {
  if (desc == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (!desc->initialized) {
    return MKLIB_STATUS_BAD_STATE;
  }
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t ValidateReduceCall(
    mklibTensorDesc_t input_desc,
    mklibTensorDesc_t output_desc,
    const mklibReduceDesc_t* desc) {
  if (desc == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  const mklibStatus_t input_status = CheckInitialized(input_desc);
  if (input_status != MKLIB_STATUS_SUCCESS) {
    return input_status;
  }
  const mklibStatus_t output_status = CheckInitialized(output_desc);
  if (output_status != MKLIB_STATUS_SUCCESS) {
    return output_status;
  }

  if (!IsValidReduceOp(desc->op)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (desc->keep_dim != 0 && desc->keep_dim != 1) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (input_desc->rank <= 0) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (!IsValidDataType(input_desc->dtype) || !IsValidDataType(output_desc->dtype)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (input_desc->dtype != output_desc->dtype) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  mklib::runtime::ReductionGeometry geometry;
  if (!mklib::runtime::MakeReductionGeometry(*input_desc, desc->axis, &geometry)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (!mklib::runtime::MatchesReducedOutputShape(
          *input_desc,
          *output_desc,
          geometry,
          desc->keep_dim != 0)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  return MKLIB_STATUS_SUCCESS;
}

}  // namespace

mklibStatus_t mklibGetReduceWorkspaceSize(
    mklibHandle_t handle,
    mklibTensorDesc_t input_desc,
    mklibTensorDesc_t output_desc,
    const mklibReduceDesc_t* desc,
    size_t* bytes_out) {
  if (handle == nullptr || bytes_out == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  const mklibStatus_t status = ValidateReduceCall(input_desc, output_desc, desc);
  if (status != MKLIB_STATUS_SUCCESS) {
    return status;
  }

  const auto key = mklib::planner::BuildReduceDispatchKey(*input_desc, *desc);
  const auto* kernel = mklib::registry::SelectReduceKernel(key);
  if (kernel == nullptr) {
    return MKLIB_STATUS_NOT_SUPPORTED;
  }

  *bytes_out =
      mklib::backend::GetReduceWorkspaceSize(key, *kernel, *input_desc, *output_desc, *desc);
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t mklibReduce(
    mklibHandle_t handle,
    mklibTensorDesc_t input_desc,
    const void* input,
    mklibTensorDesc_t output_desc,
    void* output,
    const mklibReduceDesc_t* desc,
    void* workspace,
    size_t workspace_size) {
  if (handle == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  const mklibStatus_t status = ValidateReduceCall(input_desc, output_desc, desc);
  if (status != MKLIB_STATUS_SUCCESS) {
    return status;
  }

  const auto key = mklib::planner::BuildReduceDispatchKey(*input_desc, *desc);
  const auto* kernel = mklib::registry::SelectReduceKernel(key);
  if (kernel == nullptr) {
    return MKLIB_STATUS_NOT_SUPPORTED;
  }

  int64_t input_elements = 0;
  if (!mklib::runtime::ElementCount(*input_desc, &input_elements)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (input_elements == 0) {
    return MKLIB_STATUS_SUCCESS;
  }

  if (input == nullptr || output == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  const size_t required_workspace =
      mklib::backend::GetReduceWorkspaceSize(key, *kernel, *input_desc, *output_desc, *desc);
  if (workspace_size < required_workspace) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (required_workspace > 0 && workspace == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  return mklib::backend::LaunchReduce(
      *handle,
      key,
      *kernel,
      *input_desc,
      input,
      *output_desc,
      output,
      *desc,
      workspace,
      workspace_size);
}
