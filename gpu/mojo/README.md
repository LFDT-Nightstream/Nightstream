# Nightstream Mojo GPU Scaffold

This directory is the staged Mojo shared-library project for prover-side GPU acceleration.

Current scope:

- keep proof format, verifier logic, and Poseidon2 transcript semantics unchanged,
- expose a small C ABI that Rust can load through `crates/neo-gpu`,
- start with Split-NC FE/NC evaluator hooks,
- leave witness decoding, transcript control, and final proof assembly in Rust.

Project status:

- `mojo` is installed locally through `pixi`,
- the Rust side is already wired to load a shared library and select it explicitly through
  `ProverComputeBackend::Mojo(...)` or automatically through `ProverComputeBackend::auto()`,
- the shared library now contains a bit-exact width-8 Poseidon2 permutation primitive,
- `src/poseidon_gpu_compare.mojo` is the first Mojo GPU API harness for CPU-vs-GPU parity checks.

## Layout

- `pixi.toml`: pinned Mojo toolchain environment
- `mojoproject.toml`: Nightstream-local build and ABI metadata
- `src/lib.mojo`: shared-library export surface
- `src/nightstream_gpu/field.mojo`: Goldilocks/K/Rq arithmetic
- `src/nightstream_gpu/ring.mojo`: ring helpers and reductions
- `src/nightstream_gpu/superneo.mojo`: SuperNeo row/block helpers
- `src/nightstream_gpu/sumcheck.mojo`: FE/NC evaluator state and fold kernels
- `src/nightstream_gpu/poseidon.mojo`: Poseidon2 primitive entrypoints used by the shared library
- `src/nightstream_gpu/ffi.mojo`: ABI-facing stub implementations
- `src/poseidon_gpu_compare.mojo`: batched Mojo GPU API parity check for Poseidon2
- `src/poseidon_gpu_bench.mojo`: Mojo CPU vs batched Mojo GPU throughput benchmark

## Build

Current Modular docs use `pixi.toml` to manage Mojo projects, so this scaffold includes both:

- `pixi.toml` for the pinned environment
- `mojoproject.toml` for repo-local metadata and expected output naming

Build the shared library from this directory with:

```bash
pixi run mojo build --emit shared-lib src/lib.mojo -o build/libnightstream_mojo_gpu.dylib
```

On Linux or Windows, adjust the output filename to `.so` or `.dll`.

Run the direct GPU parity harness with:

```bash
pixi run mojo run src/poseidon_gpu_compare.mojo
```

This harness uses `gpu.host.DeviceContext`, launches a batched Poseidon2 kernel across many
independent width-8 states, and compares the GPU result against the CPU Mojo implementation.

Current verified status:

- the standalone Mojo GPU scripts are the authoritative CPU-vs-GPU parity and perf path,
- the shared library exports the same Poseidon2 batch symbol and can be used from Rust in
  CPU/direct mode today,
- Rust-side auto selection prefers `Metal` on macOS and `Cuda`/`Hip` on non-macOS hosts, then
  falls back to CPU if no accelerator session opens,
- real shared-library Metal sessions are still experimental in this toolchain and may report
  unavailable or fail to launch GPU work even when the standalone scripts succeed.

Run the throughput benchmark with:

```bash
pixi run mojo run src/poseidon_gpu_bench.mojo
```

The benchmark reports:

- `cpu`: serial Mojo CPU throughput over many states
- `gpu_steady`: GPU throughput with persistent device buffers and repeated kernel launches
- `gpu_roundtrip`: GPU throughput including host-device copies each iteration

## Version Pin

The environment is pinned to `mojo==0.25.7` for reproducibility. Update that pin deliberately when
we start relying on newer GPU APIs or language changes.

## Next Steps

- stabilize shared-library GPU session open/probe on Metal so Rust can use the same GPU path that
  already works in `src/poseidon_gpu_compare.mojo`,
- flatten Rust-built FE/NC snapshot tables into ABI payloads,
- port exact Goldilocks/K/Rq arithmetic and FE/NC fold logic,
- add Mojo golden-vector tests once a Mojo toolchain is available in CI or on a supported dev host,
- replace deprecated `enqueue_function` launches with `enqueue_function_checked` and keep one
  parity test per converted kernel,
- move Metal shared-library calls behind a dedicated long-lived bridge thread in every production
  path that touches Mojo GPU APIs, then remove any remaining direct-thread launch sites,
- add explicit backend fallback diagnostics in the Rust bridge and Mojo layer so we can tell
  whether a path ran on CPU, Metal host mode, or real accelerator execution,
- add threshold sweep benches for Poseidon, FE row phase, and NC column phase, and store the
  current cutovers in one place in Rust instead of scattering ad hoc constants,
- add persistent session-owned host/device buffers for Poseidon batch and Split-NC FE/NC evals so
  we stop rebuilding buffers on every shared-library call,
- add reusable compiled-kernel and precomputed-table state on the Mojo side, following the
  `neo-midnight-mojo-bridge` pattern of caching GPU context and uploaded tables,
- add CUDA-only real-accelerator parity tests for Split-NC FE and NC in CI or on the Ubuntu GPU VM
  before we optimize those kernels further,
- add stage-specific snapshot fixtures for FE chunk, FE aggregate, NC chunk, NC aggregate, and
  terminal FE/NC rounds so parity failures localize to one proving stage instead of only showing up
  in end-to-end proofs,
- implement Mojo-side `fe_fold` and `nc_fold` state updates so evaluators can stay resident instead
  of depending on Rust to rebuild folded snapshots every round,
- finish `src/nightstream_gpu/ring.mojo` and `src/nightstream_gpu/superneo.mojo` so Ajtai/commit
  work can move onto the same backend architecture,
- add one backend-compare performance test for a real `neo-fold` multi-step proof on CUDA, similar
  to the bridge repo's backend-compare fixtures, so we track end-to-end speedup instead of only
  microbench numbers,
- document the supported backend matrix clearly:
  CUDA is the primary acceleration target, Metal Poseidon shared-lib is enabled through the Rust
  bridge fix, and Metal Split-NC remains disabled until its separate shared-lib instability is
  resolved.
