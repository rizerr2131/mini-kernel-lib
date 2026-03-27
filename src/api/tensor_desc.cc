#include "mklib/tensor_desc.h"

#include <algorithm>
#include <new>

#include "runtime/tensor_desc_state.h"

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

}  // namespace

mklibStatus_t mklibCreateTensorDesc(mklibTensorDesc_t* out) {
  if (out == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  auto* desc = new (std::nothrow) mklibTensorDesc;
  if (desc == nullptr) {
    return MKLIB_STATUS_OUT_OF_MEMORY;
  }

  *out = desc;
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t mklibDestroyTensorDesc(mklibTensorDesc_t desc) {
  if (desc == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  delete desc;
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t mklibSetTensorDesc(
    mklibTensorDesc_t desc,
    mklibDataType_t dtype,
    int rank,
    const int64_t* sizes,
    const int64_t* strides) {
  if (desc == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (!IsValidDataType(dtype) || rank < 0) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  if (rank > 0 && (sizes == nullptr || strides == nullptr)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  for (int i = 0; i < rank; ++i) {
    if (sizes[i] < 0 || strides[i] < 0) {
      return MKLIB_STATUS_INVALID_ARGUMENT;
    }
  }

  desc->dtype = dtype;
  desc->rank = rank;
  desc->sizes.assign(sizes, sizes + rank);
  desc->strides.assign(strides, strides + rank);
  desc->initialized = true;
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t mklibGetTensorDataType(
    mklibTensorDesc_t desc,
    mklibDataType_t* dtype_out) {
  if (dtype_out == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  const mklibStatus_t status = CheckInitialized(desc);
  if (status != MKLIB_STATUS_SUCCESS) {
    return status;
  }

  *dtype_out = desc->dtype;
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t mklibGetTensorRank(mklibTensorDesc_t desc, int* rank_out) {
  if (rank_out == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  const mklibStatus_t status = CheckInitialized(desc);
  if (status != MKLIB_STATUS_SUCCESS) {
    return status;
  }

  *rank_out = desc->rank;
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t mklibGetTensorSizes(
    mklibTensorDesc_t desc,
    size_t capacity,
    int64_t* sizes_out) {
  if (sizes_out == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  const mklibStatus_t status = CheckInitialized(desc);
  if (status != MKLIB_STATUS_SUCCESS) {
    return status;
  }
  if (capacity < static_cast<size_t>(desc->rank)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  std::copy(desc->sizes.begin(), desc->sizes.end(), sizes_out);
  return MKLIB_STATUS_SUCCESS;
}

mklibStatus_t mklibGetTensorStrides(
    mklibTensorDesc_t desc,
    size_t capacity,
    int64_t* strides_out) {
  if (strides_out == nullptr) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }
  const mklibStatus_t status = CheckInitialized(desc);
  if (status != MKLIB_STATUS_SUCCESS) {
    return status;
  }
  if (capacity < static_cast<size_t>(desc->rank)) {
    return MKLIB_STATUS_INVALID_ARGUMENT;
  }

  std::copy(desc->strides.begin(), desc->strides.end(), strides_out);
  return MKLIB_STATUS_SUCCESS;
}
