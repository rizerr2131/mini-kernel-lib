# Design Doc

## What This Repo Is

This repo is me trying to build a small CUDA-first kernel library.

I am not trying to clone cuBLAS or cuDNN. The part I care about is the overall
shape:

- a small public API
- explicit handles, descriptors, streams, and workspace
- a dispatch layer between the API and the actual kernels

Right now I want something small enough that I can actually finish and iterate
on without needing a huge amount of compute.

## Current State

What exists in the repo so far:

- CMake skeleton
- C API scaffold
- handle API with experimental autotune state
- tensor descriptor API
- GEMM, reduction, and `conv2d` forward entry points
- planner / registry / backend split
- two real FP32 reference GEMM paths with fused ReLU epilogue support
- one real CUDA FP32 tiled GEMM path for the current `N/T` transpose combinations
- real CUDA FP32 contiguous reduction-sum paths for device buffers, including
  an inner-axis specialization plus a generic contiguous fallback
- one real FP32 reduction-sum reference path for the broader contiguous cases
- one real CUDA FP32 direct `conv2d` forward path for contiguous device buffers
  using the existing pad / stride / dilation descriptor fields
- one real FP32 `conv2d` forward reference path
- smoke test
- GEMM correctness test
- CUDA GEMM host-fallback plus device-buffer correctness test when the CUDA
  backend is built
- CUDA reduction host-fallback plus device-buffer correctness test when the
  CUDA backend is built
- CUDA convolution host-fallback plus device-buffer correctness test when the
  CUDA backend is built, including a non-unit dilation case
- reduction correctness test
- convolution correctness test
- GEMM / reduction / convolution benchmark harnesses

What does not exist yet:

- broad real CUDA kernel coverage
- broad dtype or layout coverage
- real production dispatch heuristics
- broad correctness tests against actual GPU work beyond the current GEMM,
  reduction, and convolution slices
- benchmark numbers on real GPU kernels

## What I Want Out Of This

The goal is to have a small library for a few common ops with a clean enough
structure that I can keep adding kernels without rewriting everything every
time.

The main things I care about:

- small public API
- explicit state instead of hidden global behavior
- room for multiple kernel variants later
- correctness and benchmarking from the start

I do not want this to turn into a random pile of kernels dumped into one file.

## Scope Right Now

Current plan:

- CUDA only
- keep the API handle / descriptor / workspace shape small and explicit
- keep GEMM / pointwise / reduction / conv forward working end to end
- use reference kernels to keep dispatch honest while GPU kernels are still incomplete
- use autotune only as an explicit experiment, not magic default behavior

Later, if the basic shape holds up:

- more real CUDA kernels for the existing ops
- better dispatch heuristics
- autotuning or algorithm cache beyond a small experiment
- normalization
- wider dtype support

I am intentionally not trying to cover everything up front.

## What I Am Not Doing Yet

Not trying to do any of this right now:

- full cuBLAS compatibility
- full cuDNN compatibility
- every dtype / layout / architecture
- graph compiler stuff
- training runtime stuff
- multi-backend portability for day one

If CUDA-first ends up being too limiting later, I can revisit that then.

## API Direction

I want the public surface to stay simple and explicit.

The rough model is:

- `handle` for library state
- `tensor descriptor` for dtype / rank / sizes / strides
- operation-specific descriptors where needed
- explicit workspace size queries
- explicit status return on every public call

I am using `mklib*` as a temporary prefix for now. I do not want to use `mkl*`
because that already collides with Intel MKL.

Current GEMM API shape in the repo is basically:

```c
typedef struct mklibHandle* mklibHandle_t;
typedef struct mklibTensorDesc* mklibTensorDesc_t;

mklibStatus_t mklibCreate(mklibHandle_t* out);
mklibStatus_t mklibDestroy(mklibHandle_t handle);
mklibStatus_t mklibSetStream(mklibHandle_t handle, void* stream);
mklibStatus_t mklibSetAutotuneMode(
    mklibHandle_t handle,
    mklibAutotuneMode_t mode);

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

Reduction and `conv2d` forward follow the same basic pattern:
descriptor-backed validation, explicit workspace query, explicit status
returns, and a planner / registry / backend hop before the actual kernel.
This is still early and I expect it to move around.

## Code Layout I Am Aiming For

The current split is:

1. API
   - public entry points
   - validation
   - object lifetime
2. Runtime
   - handle state
   - descriptor state
3. Planner
   - build a dispatch key from the op description
4. Registry
   - choose a kernel from available metadata
5. Backend
   - actual launch path
   - CUDA-specific code

The main reason for this split is that I do not want the public API code to
turn into a mess of hard-coded kernel decisions.

## Dispatch Plan

I want dispatch to be based on a structured key instead of scattered condition
chains.

For GEMM, that key will probably include at least:

- dtype / compute type
- transpose flags
- problem size bucket
- pointwise epilogue
- workspace class

For reduction and convolution, the same idea now extends to layout class,
axis role, and coarse problem sizing. The current M5 implementation still uses
simple keys, but it is enough to keep the public API separated from kernel
selection and to let multiple reference kernels coexist.

## First Kernel Plan

The first real thing I wanted was one actual GEMM path end to end:

- descriptor / API path
- dispatch key
- one registered kernel
- CUDA launch
- correctness check
- benchmark result

That first real CUDA step is now partially done, and the repo has moved a bit
further:

- one real CUDA FP32 tiled GEMM path for the current `N/T` transpose combinations
- real CUDA FP32 contiguous reduction paths for device buffers, including an
  inner-axis specialization plus a generic contiguous path
- one real CUDA FP32 direct `conv2d` forward path for contiguous device buffers
  using the descriptor-driven pad / stride / dilation controls
- two reference GEMM kernels
- one reference reduction kernel family for the broader contiguous cases
- one reference `conv2d` forward kernel
- an opt-in GEMM autotune experiment on the handle

The important bit is that the public API shape did not need to change. The
current GEMM flow still goes through descriptor validation, dispatch-key
construction, registry selection, backend launch, and workspace handling. The
CUDA path is just one more kernel record in that stack. When callers keep using
plain host buffers, the dispatch still falls back to the host reference kernels.
The GEMM launch path now keeps that fallback behavior even when autotune or a
cached preferred kernel points at a higher-workspace variant: the API retries
other compatible candidates before it gives up. That means the workspace query
can stay conservative for the preferred kernel without turning every smaller
workspace launch into a hard failure.
The current CUDA launch wrapper also synchronizes the selected stream before it
returns so correctness tests and autotune timings stay deterministic while the
backend is still narrow.
Reduction now follows the same pattern for the contiguous cases too: the
registry prefers the CUDA kernels when those slices are available, but the API
still retries the reference reduction kernels when callers pass host buffers.
`conv2d` forward now follows the same pattern for the direct contiguous case.
The current validation and benchmark harnesses exercise that CUDA path across
more than the trivial unit-stride shape, including non-unit stride and
dilation descriptors, while still retrying the reference kernel when the
buffers are not CUDA-device-accessible.
The CUDA pointer inspection layer now treats ordinary host allocations as
host-only buffers instead of dropping them on the floor as
`cudaMemoryTypeUnregistered`, which keeps those reference fallbacks working in
CUDA-enabled builds too.

## Dtypes And Layouts

The first dtypes I care about are:

- FP32
- FP16
- BF16

I will probably start with the simplest thing that gets a real result and not
pretend I support more than I actually do.

For layouts, I want descriptors to stay stride-based instead of baking too much
layout logic into the type system.

## Workspace / Memory

The caller should own tensor memory.

I want operation calls to be explicit about workspace:

- query workspace size first
- pass workspace in explicitly
- avoid hidden allocations in the hot path

If I ever add exceptions to that, they should be obvious and documented.

## Testing / Benchmarking

I want both from the start, even if they start small.

Correctness plan:

- CPU reference checks where practical
- shape coverage
- edge cases
- simple deterministic behavior where applicable

Benchmarking plan:

- benchmark harness in repo
- record hardware / dtype / shape
- no big performance claims without measurements

The repo now has benchmark targets for GEMM, reduction, and convolution, plus
an opt-in autotuned GEMM benchmark mode and optional device-buffer GEMM,
reduction, and convolution benchmark modes when the CUDA backend is built. The
next step is to compare multiple CUDA kernel variants instead of only one
tiled GEMM path, the current contiguous reduction paths, and one direct
convolution path versus the reference implementations.

## Repo Shape

This is the layout I am building toward:

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

Not every directory needs to exist immediately, but this is the structure I
want instead of a flat repo.

## Milestones

M0:

- repo bootstrap
- README
- design doc
- initial API / build scaffold

M1:

- status / handle / descriptor path usable
- benchmark and test skeletons in place
- planner / registry / backend split stubbed out

M2:

- first real GEMM path
- correctness test
- first benchmark number

Status: done

M3:

- better GEMM path or second GEMM variant
- fused pointwise work
- more validation around workspace / descriptors

Status: done

M4:

- reductions or normalization
- better dispatch logic

Status: done via reductions plus broader dispatch keys

M5:

- convolution forward if the rest still feels clean
- autotuning experiments if they seem worth it

Status: done via reference `conv2d` forward plus handle-scoped GEMM autotuning

## Open Questions

- What final project name should replace `mini-kernel-lib`?
- What final API prefix should replace `mklib*`?
- Should the public surface stay pure C for a while, or should I add a C++
  wrapper early?
- How much logic should live in descriptors versus raw op descriptors?
- What is the cheapest useful GPU target for early tuning passes?
