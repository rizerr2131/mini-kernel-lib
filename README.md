# mini-kernel-lib

`mini-kernel-lib` is a small CUDA-first kernel library scaffold.

The repo is about getting the shape right first: explicit handles,
descriptors, streams, workspace queries, and a dispatch layer between the API
and the implementation. The current checkpoint has completed the initial
roadmap through M5, but it is still using host reference kernels instead of
real GPU execution.

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
- Working FP32 contiguous reduction-sum path for one-axis tensor reductions
- Working FP32 contiguous NCHW `conv2d` forward reference implementation
- Smoke test plus dedicated GEMM, reduction, and convolution correctness tests
- GEMM, reduction, and convolution benchmark targets

What is not real yet:

- Stable API
- Actual CUDA kernels or GPU execution
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
`mklib_gemm_bench` accepts an optional final `1` argument to enable the
experimental GEMM autotune path.

If CMake finds the CUDA toolkit it records that in the build, but the current
GEMM path is still a host reference implementation.

## Repo Layout

- `include/` public headers
- `src/api/` public entry points and validation
- `src/runtime/` internal handle and descriptor state
- `src/planner/` dispatch-key construction
- `src/registry/` kernel selection
- `src/backend/cuda/` current launch path
- `tests/` correctness coverage
- `benchmarks/` benchmark harnesses
- `docs/` design notes

## Docs

- [Design doc](docs/design.md)
- [Contributing guide](CONTRIBUTING.md)

## License

No license yet.
