# mini-kernel-lib

`mini-kernel-lib` is a CUDA-first GPU kernel library.

The plan is to build a small set of math and tensor ops with explicit handles,
descriptors, streams, workspace queries, and kernel dispatch.

## Status

Early repo. There is no stable API or working CUDA kernel implementation yet.

Current scaffold:

- CMake build skeleton
- C API skeleton for status, handle, tensor descriptor, and GEMM entry points
- planner/registry/backend split with stubs
- one smoke test and one benchmark stub

## Scope

Current targets:

- CUDA runtime and backend
- GEMM
- Pointwise and fused epilogues
- Reductions
- Normalization
- Benchmarks and correctness tests

Later:

- Convolution forward
- Better dispatch heuristics and autotuning
- More dtypes and layout coverage

## Principles

- Keep the public API small.
- Keep state explicit.
- Split API, runtime, planner, and kernels cleanly.
- Do not claim performance wins without tests and benchmarks.
- Keep internals replaceable without breaking the public surface.

## Repo Plan

The rough layout is:

- `include/` for public headers
- `src/` for API, runtime, planner, and backend code
- `kernels/` for kernel implementations
- `tests/` for correctness tests
- `benchmarks/` for benchmarks
- `docs/` for design notes

More detail is in [docs/design.md](docs/design.md).

## Docs

- [Design doc](docs/design.md)
- [Contributing guide](CONTRIBUTING.md)

## License

No license yet.
