# mini-kernel-lib

`mini-kernel-lib` is a small CUDA-first kernel library scaffold.

The repo is about getting the shape right first: explicit handles,
descriptors, streams, workspace queries, and a dispatch layer between the API
and the implementation.

## TL;DR

What is real today:

- Static C++20 library with a small C API in `include/mklib/`
- Public entry points for status, handle/stream state, tensor descriptors, and
  GEMM
- Planner/registry/backend split with one registered GEMM path
- Working row-major FP32 GEMM reference implementation with `N/T`
  transpose support and zero workspace
- Smoke test, GEMM correctness test, and a GEMM benchmark target

What is not real yet:

- Stable API
- Actual CUDA kernels or GPU execution
- More than one registered kernel
- Real dispatch heuristics or autotuning
- Pointwise, reduction, normalization, or convolution implementations

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Optional benchmark:

Run `mklib_gemm_bench` from your build tree after building. The exact path
depends on the CMake generator.

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
