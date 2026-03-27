# Design Doc

## Overview

`mini-kernel-lib` is a small CUDA-first kernel library.

The first version is not trying to match cuBLAS or cuDNN. The goal is a small
API and a codebase with a clean split between API, runtime, planner, registry,
and kernels.

## Why This Exists

Writing custom CUDA kernels directly inside an app gets messy quickly. Full
vendor libraries solve a lot, but they also come with a large API surface and
their own complexity.

This project sits in the middle. It should cover a few common ops with a small
API and predictable dispatch.

## Goals

- Keep the public API small and explicit.
- Keep API code separate from kernel selection and backend code.
- Make dispatch decisions visible and testable.
- Treat correctness tests and benchmarks as required, not optional.
- Leave room for autotuning or more backends later without rewriting the API.

## Not in Scope

- Full cuBLAS or cuDNN compatibility.
- Wide support for every dtype, layout, and architecture.
- A graph compiler, fusion compiler, or training runtime.
- Hiding all hardware details from the caller.
- Building a portability layer too early if CUDA is the only backend.

## Principles

- Start with a C ABI. Add a C++ wrapper later if it earns its keep.
- Use opaque handles and descriptors.
- Make stream and workspace handling explicit.
- Keep the public status/error model small and stable.
- Keep public entry points small even if internals use templates.

## Initial Target Scope

Phase 1:

- GEMM
- Pointwise ops and fused epilogues
- Reductions
- Normalization primitives

Phase 2:

- Convolution forward
- Additional epilogue fusion patterns
- Autotuning and algorithm caching
- More dtype support, including lower precision formats

## Public API Model

Start with a C ABI, opaque handles, and explicit descriptors.

Core API concepts:

- `handle`: library state, bound to a device/context and stream
- `tensor descriptor`: dtype, rank, sizes, strides, layout info
- `operation descriptor`: operation-specific configuration
- `workspace query`: caller asks how much temporary memory an op needs
- `status`: every public call returns one

Do not use `mkl*` as a prefix because Intel MKL already owns that space. Use
`mklib*` until the project has a final name.

Example shape:

```c
typedef struct mklibHandle* mklibHandle_t;
typedef struct mklibTensorDesc* mklibTensorDesc_t;

mklibStatus_t mklibCreate(mklibHandle_t* out);
mklibStatus_t mklibDestroy(mklibHandle_t handle);
mklibStatus_t mklibSetStream(mklibHandle_t handle, void* stream);

mklibStatus_t mklibCreateTensorDesc(mklibTensorDesc_t* out);
mklibStatus_t mklibSetTensorDesc(
    mklibTensorDesc_t desc,
    mklibDataType_t dtype,
    int rank,
    const int64_t* sizes,
    const int64_t* strides);

mklibStatus_t mklibGetGemmWorkspaceSize(
    mklibHandle_t handle,
    const mklibGemmDesc_t* desc,
    size_t* bytes_out);

mklibStatus_t mklibGemm(
    mklibHandle_t handle,
    const mklibGemmDesc_t* desc,
    const void* a,
    const void* b,
    void* c,
    void* workspace,
    size_t workspace_size);
```

Notes:

- `void* stream` is just a placeholder in the sketch.
- The first version should have a small set of operation entry points.
- Descriptor creation should be cheap. Large allocations inside descriptors are
  a bad sign.

## Internal Architecture

Split the code into these layers:

1. API layer
   - Argument validation
   - Status/error mapping
   - Public object lifetime
2. Runtime layer
   - Device and stream binding
   - Capability queries
   - Workspace helpers
3. Planner/dispatcher
   - Convert descriptors into a dispatch key
   - Choose algorithm/kernel variant
   - Apply heuristics and determinism rules
4. Kernel registry
   - Static registration of available kernels
   - Metadata: supported dtypes, layouts, tile shapes, alignment requirements,
     architecture support
5. Backend implementation
   - CUDA launchers
   - Kernel source and low-level tuning code

Public API code should not contain device-specific dispatch logic. Keep that in
the planner/registry path.

## Dispatch Model

Use a structured dispatch key instead of scattered `if` chains.

The dispatch key will likely include:

- operation family
- architecture / compute capability
- input and output dtypes
- layout or stride class
- alignment guarantees
- problem-shape bucket
- math mode
- determinism requirement
- fused epilogue kind, if any

Phase 1 should use static heuristics and hand-written registry metadata.

## Kernel Strategy

Start with a few kernels that matter instead of a long list of weak ones.

Suggested order:

1. GEMM
2. Pointwise fusion around GEMM outputs
3. Reductions
4. Normalization
5. Convolution forward

Early rules:

- Prefer hand-written kernels and simple template specialization.
- Keep generated code optional.
- Keep kernel metadata close to the implementation that owns it.
- Make launch configuration visible in logs or debug builds.

## Data Types and Layouts

Initial dtypes:

- FP32
- FP16
- BF16

Later:

- INT8
- FP8

Layouts:

- GEMM should support row-major and column-major style access patterns via
  strides rather than separate type hierarchies.
- DNN-style ops should likely optimize for NHWC on CUDA, but descriptors should
  stay stride-based so the API is not hard-wired to one layout.

## Memory and Workspace Model

The caller owns tensor memory. Public operation calls should not allocate hidden
scratch buffers in the hot path.

Rules:

- Workspace size can be queried ahead of time.
- Workspace can be reused across calls.
- No-workspace paths should exist where practical.
- Temporary allocations in public op entry points are a bug unless documented.

## Error Handling and Diagnostics

Public calls should return stable status codes. Internals can use richer error
types, but they need to map back to a small public status set.

Diagnostics should include:

- invalid argument reporting
- unsupported configuration reporting
- optional verbose dispatch logging
- version and capability queries

## Testing Strategy

Correctness and performance both matter.

Correctness:

- reference comparisons against CPU implementations where practical
- randomized shape coverage
- edge-case coverage for zero sizes, alignment boundaries, and non-contiguous
  strides
- determinism tests where determinism is part of the contract

Performance:

- benchmark harness checked into the repo
- architecture and dtype recorded with every result
- no performance claim without measurement methodology
- compare against strong baselines when the comparison is fair and repeatable

## Build and Tooling Direction

Starting point:

- CMake
- C++20 for implementation code
- CUDA toolchain for the first backend

Assume out-of-tree builds and a clean split between public headers and backend
code.

## Proposed Repository Layout

```text
include/
  mklib/
src/
  api/
  runtime/
  planner/
  registry/
  backend/
    cuda/
kernels/
  gemm/
  pointwise/
  reduction/
  normalization/
tests/
benchmarks/
docs/
cmake/
```

Not every directory needs to exist on day one, but the repo should grow in that
direction instead of turning into a flat `src/` dump.

## Milestones

M0:

- repository bootstrap
- README
- design doc
- basic contribution conventions

M1:

- build system skeleton
- status codes and handle lifetime
- device/stream binding
- first benchmark harness skeleton

M2:

- first GEMM path end to end
- descriptor plumbing
- correctness tests

M3:

- fused pointwise and reduction primitives
- workspace query and reuse model stabilized

M4:

- convolution forward prototype
- better dispatch heuristics

M5:

- autotuning or algorithm-cache experiments
- packaging and release hygiene

## Open Questions

- What final project name and API prefix should replace `mklib*`?
- Should the first public surface be pure C, or C plus a small C++ wrapper?
- How much should descriptors normalize layouts versus just preserving raw
  strides?
- Is CUDA-only the right first step, or should a portability layer exist from
  the start?
- Which baselines matter most early on: custom kernels, cuBLAS/cuDNN, or
  framework-level calls?
