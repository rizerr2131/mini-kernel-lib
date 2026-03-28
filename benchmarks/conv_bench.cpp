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

std::vector<float> MakeValues(int64_t elements, int seed) {
  std::vector<float> values(static_cast<size_t>(elements), 0.0f);
  for (int64_t i = 0; i < elements; ++i) {
    values[static_cast<size_t>(i)] = static_cast<float>(((seed + i * 5) % 29) - 14) / 11.0f;
  }
  return values;
}

}  // namespace

int main(int argc, char** argv) {
  size_t iterations = 10;
  size_t warmup_iterations = 2;
  int64_t batch = 1;
  int64_t in_channels = 32;
  int64_t in_h = 32;
  int64_t in_w = 32;
  int64_t out_channels = 64;
  int64_t kernel = 3;

  if (argc > 1) {
    iterations = ParsePositiveSizeT(argv[1], iterations);
  }
  if (argc > 2) {
    batch = ParsePositiveInt64(argv[2], batch);
  }
  if (argc > 3) {
    in_channels = ParsePositiveInt64(argv[3], in_channels);
  }
  if (argc > 4) {
    in_h = ParsePositiveInt64(argv[4], in_h);
  }
  if (argc > 5) {
    in_w = ParsePositiveInt64(argv[5], in_w);
  }
  if (argc > 6) {
    out_channels = ParsePositiveInt64(argv[6], out_channels);
  }
  if (argc > 7) {
    kernel = ParsePositiveInt64(argv[7], kernel);
  }
  if (argc > 8) {
    warmup_iterations = ParsePositiveSizeT(argv[8], warmup_iterations);
  }

  const mklibConv2dDesc_t desc = {
      .pad_h = kernel / 2,
      .pad_w = kernel / 2,
      .stride_h = 1,
      .stride_w = 1,
      .dilation_h = 1,
      .dilation_w = 1,
  };

  const std::vector<int64_t> input_sizes = {batch, in_channels, in_h, in_w};
  const std::vector<int64_t> filter_sizes = {out_channels, in_channels, kernel, kernel};
  const int64_t out_h =
      (in_h + 2 * desc.pad_h - ((kernel - 1) * desc.dilation_h + 1)) / desc.stride_h + 1;
  const int64_t out_w =
      (in_w + 2 * desc.pad_w - ((kernel - 1) * desc.dilation_w + 1)) / desc.stride_w + 1;
  const std::vector<int64_t> output_sizes = {batch, out_channels, out_h, out_w};
  const auto input_strides = MakeContiguousStrides(input_sizes);
  const auto filter_strides = MakeContiguousStrides(filter_sizes);
  const auto output_strides = MakeContiguousStrides(output_sizes);

  mklibHandle_t handle = nullptr;
  const mklibStatus_t create_status = mklibCreate(&handle);
  if (create_status != MKLIB_STATUS_SUCCESS) {
    std::cerr << "mklibCreate failed: " << mklibGetStatusString(create_status) << '\n';
    return 1;
  }

  mklibTensorDesc_t input_desc = nullptr;
  mklibTensorDesc_t filter_desc = nullptr;
  mklibTensorDesc_t output_desc = nullptr;
  const mklibStatus_t input_desc_status = mklibCreateTensorDesc(&input_desc);
  const mklibStatus_t filter_desc_status = mklibCreateTensorDesc(&filter_desc);
  const mklibStatus_t output_desc_status = mklibCreateTensorDesc(&output_desc);
  if (input_desc_status != MKLIB_STATUS_SUCCESS ||
      filter_desc_status != MKLIB_STATUS_SUCCESS ||
      output_desc_status != MKLIB_STATUS_SUCCESS) {
    std::cerr << "mklibCreateTensorDesc failed\n";
    return 1;
  }

  const mklibStatus_t set_input_status = mklibSetTensorDesc(
      input_desc,
      MKLIB_DATA_TYPE_FLOAT32,
      static_cast<int>(input_sizes.size()),
      input_sizes.data(),
      input_strides.data());
  const mklibStatus_t set_filter_status = mklibSetTensorDesc(
      filter_desc,
      MKLIB_DATA_TYPE_FLOAT32,
      static_cast<int>(filter_sizes.size()),
      filter_sizes.data(),
      filter_strides.data());
  const mklibStatus_t set_output_status = mklibSetTensorDesc(
      output_desc,
      MKLIB_DATA_TYPE_FLOAT32,
      static_cast<int>(output_sizes.size()),
      output_sizes.data(),
      output_strides.data());
  if (set_input_status != MKLIB_STATUS_SUCCESS ||
      set_filter_status != MKLIB_STATUS_SUCCESS ||
      set_output_status != MKLIB_STATUS_SUCCESS) {
    std::cerr << "mklibSetTensorDesc failed\n";
    return 1;
  }

  size_t workspace_bytes = 0;
  const mklibStatus_t workspace_status = mklibGetConv2dForwardWorkspaceSize(
      handle,
      &desc,
      input_desc,
      filter_desc,
      output_desc,
      &workspace_bytes);
  if (workspace_status != MKLIB_STATUS_SUCCESS) {
    std::cerr << "mklibGetConv2dForwardWorkspaceSize failed: "
              << mklibGetStatusString(workspace_status) << '\n';
    return 1;
  }

  auto input = MakeValues(batch * in_channels * in_h * in_w, 3);
  auto filter = MakeValues(out_channels * in_channels * kernel * kernel, 11);
  std::vector<float> output(static_cast<size_t>(batch * out_channels * out_h * out_w), 0.0f);

  for (size_t i = 0; i < warmup_iterations; ++i) {
    const mklibStatus_t status = mklibConv2dForward(
        handle,
        &desc,
        input_desc,
        input.data(),
        filter_desc,
        filter.data(),
        output_desc,
        output.data(),
        nullptr,
        workspace_bytes);
    if (status != MKLIB_STATUS_SUCCESS) {
      std::cerr << "warmup mklibConv2dForward failed: "
                << mklibGetStatusString(status) << '\n';
      return 1;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    const mklibStatus_t status = mklibConv2dForward(
        handle,
        &desc,
        input_desc,
        input.data(),
        filter_desc,
        filter.data(),
        output_desc,
        output.data(),
        nullptr,
        workspace_bytes);
    if (status != MKLIB_STATUS_SUCCESS) {
      std::cerr << "mklibConv2dForward failed: " << mklibGetStatusString(status) << '\n';
      return 1;
    }
  }
  const auto stop = std::chrono::steady_clock::now();

  const double total_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  const double ns_per_call = total_ns / static_cast<double>(iterations);
  const double total_seconds = total_ns / 1e9;
  const double gflops =
      (2.0 * static_cast<double>(batch) * static_cast<double>(out_channels) *
       static_cast<double>(out_h) * static_cast<double>(out_w) *
       static_cast<double>(in_channels) * static_cast<double>(kernel) *
       static_cast<double>(kernel) * static_cast<double>(iterations)) /
      (total_seconds * 1e9);
  const double checksum = std::accumulate(output.begin(), output.end(), 0.0);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "iterations: " << iterations << '\n';
  std::cout << "warmup_iterations: " << warmup_iterations << '\n';
  std::cout << "batch: " << batch << '\n';
  std::cout << "in_channels: " << in_channels << '\n';
  std::cout << "out_channels: " << out_channels << '\n';
  std::cout << "height: " << in_h << '\n';
  std::cout << "width: " << in_w << '\n';
  std::cout << "out_height: " << out_h << '\n';
  std::cout << "out_width: " << out_w << '\n';
  std::cout << "kernel: " << kernel << '\n';
  std::cout << "workspace_bytes: " << workspace_bytes << '\n';
  std::cout << "conv_ns_per_call: " << ns_per_call << '\n';
  std::cout << "conv_gflops: " << gflops << '\n';
  std::cout << "checksum: " << checksum << '\n';

  mklibDestroyTensorDesc(output_desc);
  mklibDestroyTensorDesc(filter_desc);
  mklibDestroyTensorDesc(input_desc);
  mklibDestroy(handle);
  return 0;
}
