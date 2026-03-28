#ifndef MKLIB_BACKEND_BACKEND_H_
#define MKLIB_BACKEND_BACKEND_H_

#include <stddef.h>

#include "mklib/conv.h"
#include "mklib/gemm.h"
#include "mklib/reduction.h"
#include "mklib/status.h"
#include "planner/dispatch_key.h"
#include "registry/kernel_registry.h"
#include "runtime/handle_state.h"
#include "runtime/tensor_desc_state.h"

namespace mklib::backend {

size_t GetGemmWorkspaceSize(
    const planner::DispatchKey& key,
    const registry::KernelRecord& kernel,
    const mklibGemmDesc_t& desc);

size_t GetReduceWorkspaceSize(
    const planner::DispatchKey& key,
    const registry::KernelRecord& kernel,
    const ::mklibTensorDesc& input_desc,
    const ::mklibTensorDesc& output_desc,
    const mklibReduceDesc_t& desc);

size_t GetConv2dForwardWorkspaceSize(
    const planner::DispatchKey& key,
    const registry::KernelRecord& kernel,
    const ::mklibTensorDesc& input_desc,
    const ::mklibTensorDesc& filter_desc,
    const ::mklibTensorDesc& output_desc,
    const mklibConv2dDesc_t& desc);

mklibStatus_t LaunchGemm(
    const ::mklibHandle& handle,
    const planner::DispatchKey& key,
    const registry::KernelRecord& kernel,
    const mklibGemmDesc_t& desc,
    const void* a,
    const void* b,
    void* c,
    void* workspace,
    size_t workspace_size);

mklibStatus_t LaunchReduce(
    const ::mklibHandle& handle,
    const planner::DispatchKey& key,
    const registry::KernelRecord& kernel,
    const ::mklibTensorDesc& input_desc,
    const void* input,
    const ::mklibTensorDesc& output_desc,
    void* output,
    const mklibReduceDesc_t& desc,
    void* workspace,
    size_t workspace_size);

mklibStatus_t LaunchConv2dForward(
    const ::mklibHandle& handle,
    const planner::DispatchKey& key,
    const registry::KernelRecord& kernel,
    const ::mklibTensorDesc& input_desc,
    const void* input,
    const ::mklibTensorDesc& filter_desc,
    const void* filter,
    const ::mklibTensorDesc& output_desc,
    void* output,
    const mklibConv2dDesc_t& desc,
    void* workspace,
    size_t workspace_size);

}  // namespace mklib::backend

#endif  // MKLIB_BACKEND_BACKEND_H_
