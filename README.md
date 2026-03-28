# mini-kernel-lib

`mini-kernel-lib` is a small CUDA-first kernel library scaffold.

The repo is about getting the shape right first: explicit handles,
descriptors, streams, workspace queries, and a dispatch layer between the API
and the implementation. The current checkpoint has completed the initial
roadmap through M5 and now includes narrow real CUDA execution slices for
FP32 GEMM, reduction, and `conv2d` forward.

## TL;DR

What is real today:

- Static C++20 library with a small C API in `include/mklib/`
- Public entry points for status, handle/stream/autotune state, tensor
  descriptors, GEMM, reduction, and `conv2d` forward
- Planner/registry/backend split shared across GEMM, reduction, and
  convolution dispatch
- Two working row-major FP32 GEMM reference implementations with `N/T`
  transpose support, fused ReLU epilogue support, explicit workspace
  validation, and experimental handle-scoped autotuning
- Optional real CUDA FP32 tiled GEMM kernel for device buffers across the
  current `N/T` transpose combinations, with fused ReLU support and
  stream-aware launch plumbing
- Optional real CUDA FP32 contiguous reduction kernels for device buffers,
  covering both the inner-axis specialization and the generic contiguous
  one-axis cases
- Optional real CUDA FP32 direct `conv2d` forward kernel for contiguous NCHW
  device buffers, driven by the existing pad / stride / dilation descriptor
  fields
- Host buffers still fall back to the reference GEMM and reduction kernels so
  the existing CPU-backed tests and examples stay valid without CUDA memory
  management; `conv2d` now uses the same fallback pattern
- CUDA pointer inspection now classifies ordinary host allocations as
  host-only buffers instead of rejecting them, so CUDA-enabled builds still
  route plain host pointers into the reference fallbacks cleanly
- GEMM kernel preference and the experimental autotune cache now remain
  candidate-aware at launch time, so a cached higher-workspace kernel can still
  fall back to a compatible lower-workspace path when that is the only runnable
  option; the workspace query can therefore remain a conservative upper bound
  for the preferred path instead of a strict minimum for every successful launch
- The first CUDA GEMM launch path currently synchronizes its stream before
  returning so correctness checks and autotune timing stay deterministic
- The CUDA reduction launch path currently synchronizes its stream before
  returning for the same deterministic behavior
- Working FP32 contiguous reduction-sum path for one-axis tensor reductions
- Working FP32 contiguous NCHW `conv2d` forward reference implementation plus
  an optional direct CUDA path
- Smoke test plus dedicated GEMM, reduction, and convolution correctness tests
- CUDA-enabled correctness tests for GEMM, reduction, and convolution that now
  cover host fallbacks plus device-buffer cases when a CUDA device is available;
  the convolution coverage now includes non-unit dilation descriptors too
- CUDA-aware GEMM, reduction, and convolution benchmark modes

What is not real yet:

- Stable API
- Broad CUDA coverage beyond the current narrow FP32 GEMM, reduction, and
  convolution slices
- Broad dtype / layout coverage
- Production-quality dispatch heuristics or autotuning
- Normalization, backward ops, or convolution variants beyond one direct path

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Optional benchmarks:

Run `mklib_gemm_bench`, `mklib_reduce_bench`, or `mklib_conv_bench` from your
build tree after building. The exact paths depend on the CMake generator.
`mklib_gemm_bench` accepts optional trailing flags:

- argument `6`: `1` enables the experimental GEMM autotune path
- argument `7`: `1` switches the benchmark to device buffers when the CUDA
  backend is built and a CUDA device is available
- argument `8`: `1` switches `trans_a` to `T` instead of `N`
- argument `9`: `1` switches `trans_b` to `T` instead of `N`

`mklib_reduce_bench` accepts one optional trailing flag:

- argument `6`: `1` switches the benchmark to device buffers when the CUDA
  backend is built and a CUDA device is available

`mklib_conv_bench` accepts these optional trailing flags:

- argument `9`: `1` switches the benchmark to device buffers when the CUDA
  backend is built and a CUDA device is available
- argument `10`: override `pad_h`
- argument `11`: override `pad_w`
- argument `12`: override `stride_h`
- argument `13`: override `stride_w`
- argument `14`: override `dilation_h`
- argument `15`: override `dilation_w`

The optional CUDA backend now builds only when CMake can find a CUDA compiler
and toolkit. Without that environment, the library stays on the reference
paths.

## Repo Layout

- `include/` public headers
- `src/api/` public entry points and validation
- `src/runtime/` internal handle and descriptor state
- `src/planner/` dispatch-key construction
- `src/registry/` kernel selection
- `src/backend/cuda/` current launch path
- `src/backend/cuda/cuda_conv_kernel.cu` first real CUDA conv kernel
- `src/backend/cuda/cuda_gemm_kernel.cu` first real CUDA GEMM kernel
- `src/backend/cuda/cuda_reduction_kernel.cu` first real CUDA reduction kernel
- `tests/` correctness coverage
- `benchmarks/` benchmark harnesses
- `docs/` design notes

## Docs

- [Design doc](docs/design.md)
- [Contributing guide](CONTRIBUTING.md)

## License

No license yet.
