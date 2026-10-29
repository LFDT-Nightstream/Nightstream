# neo-prover-metal

Metal prover backend for Apple GPUs. `nightstream` uses it through
`MetalRowProver` when you build `nightstream` with its `metal` feature. See
[ARCHITECTURE.md](ARCHITECTURE.md) for what the crate owns.

`build.rs` compiles the shaders with `xcrun metal`. This needs Xcode; the
Command Line Tools do not include the Metal compiler.

Run the tests on Apple Silicon macOS:

```bash
timeout 300s cargo test -p neo-prover-metal --release --features metal
```

`tests/metal_arithmetic.rs` checks the Merkle, DFT, GKR, and key-weight
kernels against CPU references. The unit tests in `tests/unit/` check the
joint oracle, the openings, and the commitments.
