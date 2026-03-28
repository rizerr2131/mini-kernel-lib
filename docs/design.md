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
- one real FP32 reduction-sum path
- one real FP32 `conv2d` forward path
- smoke test
- GEMM correctness test
- reduction correctness test
- convolution correctness test
- GEMM / reduction / convolution benchmark harnesses

What does not exist yet:

- real CUDA kernels
- broad dtype or layout coverage
- real production dispatch heuristics
- correctness tests against actual GPU work
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
- use reference kernels to keep dispatch honest while GPU kernels are still missing
- use autotune only as an explicit experiment, not magic default behavior

Later, if the basic shape holds up:

- real CUDA kernels for the existing ops
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

That first step is done now, and the repo has moved a bit further:

- two reference GEMM kernels
- one reference reduction kernel family
- one reference `conv2d` forward kernel
- an opt-in GEMM autotune experiment on the handle

The actual CUDA kernel work is still pending, but the API and dispatch path are
already exercising the shape I wanted.

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
an opt-in autotuned GEMM benchmark mode. The next step is to compare actual
CUDA kernel variants instead of only reference paths.

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
