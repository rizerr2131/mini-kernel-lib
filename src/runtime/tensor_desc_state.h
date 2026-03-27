#ifndef MKLIB_RUNTIME_TENSOR_DESC_STATE_H_
#define MKLIB_RUNTIME_TENSOR_DESC_STATE_H_

#include <vector>

#include "mklib/tensor_desc.h"

struct mklibTensorDesc {
  mklibDataType_t dtype = MKLIB_DATA_TYPE_INVALID;
  int rank = 0;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  bool initialized = false;
};

#endif  // MKLIB_RUNTIME_TENSOR_DESC_STATE_H_
