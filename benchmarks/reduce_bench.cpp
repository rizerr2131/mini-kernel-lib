#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "mklib/mklib.h"

namespace {

int64_t ParsePositiveInt64(const char* value, int64_t fallback) {
  const long long parsed = std::strtoll(value, nullptr, 10);
  if (parsed <= 0) {
    return fallback;
  }
  return static_cast<int64_t>(parsed);
}

size_t ParsePositiveSizeT(const char* value, size_t fallback) {
  const unsigned long long parsed = std::strtoull(value, nullptr, 10);
  if (parsed == 0) {
    return fallback;
  }
  return static_cast<size_t>(parsed);
}

std::vector<int64_t> MakeContiguousStrides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  int64_t stride = 1;
  for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = stride;
    stride *= sizes[static_cast<size_t>(i)];
  }
  return strides;
}

std::vector<float> MakeInput(int64_t outer, int64_t reduce, int64_t inner) {
  const int64_t elements = outer * reduce * inner;
  std::vector<float> input(static_cast<size_t>(elements), 0.0f);
  for (int64_t i = 0; i < elements; ++i) {
    input[static_cast<size_t>(i)] = static_cast<float>((i % 23) - 11) / 9.0f;
  }
  return input;
}

}  // namespace

int main(int argc, char** argv) {
  size_t iterations = 50;
  size_t warmup_iterations = 5;
  int64_t outer = 512;
  int64_t reduce = 1024;
  int64_t inner = 1;

  if (argc > 1) {
    iterations = ParsePositiveSizeT(argv[1], iterations);
  }
  if (argc > 2) {
    outer = ParsePositiveInt64(argv[2], outer);
  }
  if (argc > 3) {
    reduce = ParsePositiveInt64(argv[3], reduce);
  }
  if (argc > 4) {
    inner = ParsePositiveInt64(argv[4], inner);
  }
  if (argc > 5) {
    warmup_iterations = ParsePositiveSizeT(argv[5], warmup_iterations);
  }

  mklibHandle_t handle = nullptr;
  const mklibStatus_t create_status = mklibCreate(&handle);
  if (create_status != MKLIB_STATUS_SUCCESS) {
    std::cerr << "mklibCreate failed: " << mklibGetStatusString(create_status) << '\n';
    return 1;
  }

  mklibTensorDesc_t input_desc = nullptr;
  mklibTensorDesc_t output_desc = nullptr;
  const mklibStatus_t create_input_desc_status = mklibCreateTensorDesc(&input_desc);
  const mklibStatus_t create_output_desc_status = mklibCreateTensorDesc(&output_desc);
  if (create_input_desc_status != MKLIB_STATUS_SUCCESS ||
      create_output_desc_status != MKLIB_STATUS_SUCCESS) {
    std::cerr << "mklibCreateTensorDesc failed: "
              << mklibGetStatusString(create_input_desc_status != MKLIB_STATUS_SUCCESS
                                          ? create_input_desc_status
                                          : create_output_desc_status)
              << '\n';
    return 1;
  }

  const std::vector<int64_t> input_sizes = {outer, reduce, inner};
  const std::vector<int64_t> output_sizes = {outer, inner};
  const auto input_strides = MakeContiguousStrides(input_sizes);
  const auto output_strides = MakeContiguousStrides(output_sizes);

  const mklibStatus_t set_input_desc_status = mklibSetTensorDesc(
      input_desc,
      MKLIB_DATA_TYPE_FLOAT32,
      static_cast<int>(input_sizes.size()),
      input_sizes.data(),
      input_strides.data());
  const mklibStatus_t set_output_desc_status = mklibSetTensorDesc(
      output_desc,
      MKLIB_DATA_TYPE_FLOAT32,
      static_cast<int>(output_sizes.size()),
      output_sizes.data(),
      output_strides.data());
  if (set_input_desc_status != MKLIB_STATUS_SUCCESS ||
      set_output_desc_status != MKLIB_STATUS_SUCCESS) {
    std::cerr << "mklibSetTensorDesc failed: "
              << mklibGetStatusString(set_input_desc_status != MKLIB_STATUS_SUCCESS
                                          ? set_input_desc_status
                                          : set_output_desc_status)
              << '\n';
    return 1;
  }

  const mklibReduceDesc_t desc = {
      .op = MKLIB_REDUCE_OP_SUM,
      .axis = 1,
      .keep_dim = 0,
  };

  size_t workspace_bytes = 0;
  const mklibStatus_t workspace_status =
      mklibGetReduceWorkspaceSize(handle, input_desc, output_desc, &desc, &workspace_bytes);
  if (workspace_status != MKLIB_STATUS_SUCCESS) {
    std::cerr << "mklibGetReduceWorkspaceSize failed: "
              << mklibGetStatusString(workspace_status) << '\n';
    return 1;
  }

  auto input = MakeInput(outer, reduce, inner);
  std::vector<float> output(static_cast<size_t>(outer * inner), 0.0f);

  for (size_t i = 0; i < warmup_iterations; ++i) {
    const mklibStatus_t status = mklibReduce(
        handle,
        input_desc,
        input.data(),
        output_desc,
        output.data(),
        &desc,
        nullptr,
        workspace_bytes);
    if (status != MKLIB_STATUS_SUCCESS) {
      std::cerr << "warmup mklibReduce failed: " << mklibGetStatusString(status) << '\n';
      return 1;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    const mklibStatus_t status = mklibReduce(
        handle,
        input_desc,
        input.data(),
        output_desc,
        output.data(),
        &desc,
        nullptr,
        workspace_bytes);
    if (status != MKLIB_STATUS_SUCCESS) {
      std::cerr << "mklibReduce failed: " << mklibGetStatusString(status) << '\n';
      return 1;
    }
  }
  const auto stop = std::chrono::steady_clock::now();

  const double total_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  const double ns_per_call = total_ns / static_cast<double>(iterations);
  const double total_seconds = total_ns / 1e9;
  const double elements_per_second =
      static_cast<double>(outer) * static_cast<double>(reduce) * static_cast<double>(inner) *
      static_cast<double>(iterations) / total_seconds;
  const double checksum = std::accumulate(output.begin(), output.end(), 0.0);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "iterations: " << iterations << '\n';
  std::cout << "warmup_iterations: " << warmup_iterations << '\n';
  std::cout << "outer: " << outer << '\n';
  std::cout << "reduce: " << reduce << '\n';
  std::cout << "inner: " << inner << '\n';
  std::cout << "workspace_bytes: " << workspace_bytes << '\n';
  std::cout << "reduce_ns_per_call: " << ns_per_call << '\n';
  std::cout << "reduce_elements_per_second: " << elements_per_second << '\n';
  std::cout << "checksum: " << checksum << '\n';

  mklibDestroyTensorDesc(output_desc);
  mklibDestroyTensorDesc(input_desc);
  mklibDestroy(handle);
  return 0;
}
