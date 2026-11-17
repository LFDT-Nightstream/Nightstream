# neo-prover-cuda

CUDA prover backend for NVIDIA GPUs, written in Rust with cuda-oxide. See
[ARCHITECTURE.md](ARCHITECTURE.md) for what the crate owns.

The `cuda` feature builds the kernels with the cuda-oxide codegen backend
(`cargo +nightly-2026-04-03 oxide`, cuda-oxide `baa1773a`). `cargo oxide
build` sets the backend flags for a build. A test run needs the same
flags:

```bash
cd crates/neo-prover-cuda
RUSTFLAGS="-Z codegen-backend=$HOME/.cargo/cuda-oxide/librustc_codegen_cuda.so -C opt-level=3 -C debug-assertions=off -Z mir-enable-passes=-JumpThreading -Csymbol-mangling-version=v0" \
LD_LIBRARY_PATH="$(rustc +nightly-2026-04-03 --print sysroot)/lib" \
CARGO_TARGET_DIR=../../target/oxide \
timeout 300s cargo +nightly-2026-04-03 test --release --features cuda
```

The separate target directory keeps these builds apart from the stable
toolchain builds. `tests/cuda_arithmetic.rs` checks the Merkle, DFT, GKR,
and key-weight kernels against CPU references. The unit tests in
`tests/unit/` check the field, Poseidon2, SHAKE128 and the commitments.
