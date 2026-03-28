#include "mklib/conv.h"

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

mklibStatus_t CheckInitialized(mklibTensorDesc_t desc) {
  if (desc == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (!desc->initialized) {
    return MKLIB_STATUS_BAD_STATE;
  }
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t ValidateConv2dForward(
    mklibTensorDesc_t input_desc,
    mklibTensorDesc_t filter_desc,
    mklibTensorDesc_t output_desc,
    const mklibConv2dDesc_t* desc) {
  if (desc == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  const mklibStatus_t input_status = CheckInitialized(input_desc);
  if (input_status != MKLIB_STATUS_SUCCESS) {
    return input_status;
  }
  const mklibStatus_t filter_status = CheckInitialized(filter_desc);
  if (filter_status != MKLIB_STATUS_SUCCESS) {
    return filter_status;
  }
  const mklibStatus_t output_status = CheckInitialized(output_desc);
  if (output_status != MKLIB_STATUS_SUCCESS) {
    return output_status;
  }

  if (!IsValidDataType(input_desc->dtype) ||
      !IsValidDataType(filter_desc->dtype) ||
      !IsValidDataType(output_desc->dtype)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (input_desc->dtype != filter_desc->dtype ||
      input_desc->dtype != output_desc->dtype) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  mklib::runtime::Conv2dGeometry geometry;
  if (!mklib::runtime::MakeConv2dGeometry(*input_desc, *filter_desc, *desc, &geometry)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (!mklib::runtime::MatchesConv2dOutputShape(*output_desc, geometry)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  return MKLIB_STATUS_SUCCESS;
}

}  // namespace

mklibStatus_t mklibGetConv2dForwardWorkspaceSize(
    mklibHandle_t handle,
    const mklibConv2dDesc_t* desc,
    mklibTensorDesc_t input_desc,
    mklibTensorDesc_t filter_desc,
    mklibTensorDesc_t output_desc,
    size_t* bytes_out) {
  if (handle == nullptr || bytes_out == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  const mklibStatus_t status =
      ValidateConv2dForward(input_desc, filter_desc, output_desc, desc);
  if (status != MKLIB_STATUS_SUCCESS) {
    return status;
  }

  const auto key = mklib::planner::BuildConv2dForwardDispatchKey(*input_desc, *filter_desc, *desc);
  const auto* kernel = mklib::registry::SelectConv2dForwardKernel(key);
  if (kernel == nullptr) {
    return MKLIB_STATUS_NOT_SUPPORTED;
  }

  *bytes_out = mklib::backend::GetConv2dForwardWorkspaceSize(
      key,
      *kernel,
      *input_desc,
      *filter_desc,
      *output_desc,
      *desc);
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t mklibConv2dForward(
    mklibHandle_t handle,
    const mklibConv2dDesc_t* desc,
    mklibTensorDesc_t input_desc,
    const void* input,
    mklibTensorDesc_t filter_desc,
    const void* filter,
    mklibTensorDesc_t output_desc,
    void* output,
    void* workspace,
    size_t workspace_size) {
  if (handle == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  const mklibStatus_t status =
      ValidateConv2dForward(input_desc, filter_desc, output_desc, desc);
  if (status != MKLIB_STATUS_SUCCESS) {
    return status;
  }

  const auto key = mklib::planner::BuildConv2dForwardDispatchKey(*input_desc, *filter_desc, *desc);
  const auto* kernel = mklib::registry::SelectConv2dForwardKernel(key);
  if (kernel == nullptr) {
    return MKLIB_STATUS_NOT_SUPPORTED;
  }

  int64_t output_elements = 0;
  if (!mklib::runtime::ElementCount(*output_desc, &output_elements)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (output_elements == 0) {
    return MKLIB_STATUS_SUCCESS;
  }

  if (input == nullptr || filter == nullptr || output == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  const size_t required_workspace = mklib::backend::GetConv2dForwardWorkspaceSize(
      key,
      *kernel,
      *input_desc,
      *filter_desc,
      *output_desc,
      *desc);
  if (workspace_size < required_workspace) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (required_workspace > 0 && workspace == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  return mklib::backend::LaunchConv2dForward(
      *handle,
      key,
      *kernel,
      *input_desc,
      input,
      *filter_desc,
      filter,
      *output_desc,
      output,
      *desc,
      workspace,
      workspace_size);
}
