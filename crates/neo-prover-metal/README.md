# neo-prover-metal

Metal prover backend for Apple GPUs. `nightstream` does not use it now. Its
steps prove and verify on the CPU path, and `Engine::Metal` returns an error.
See [ARCHITECTURE.md](ARCHITECTURE.md) for what the crate owns.

`build.rs` compiles the shaders with `xcrun metal`. This needs Xcode; the
Command Line Tools do not include the Metal compiler.

Run the tests on Apple Silicon macOS:

```bash
timeout 300s cargo test -p neo-prover-metal --release --features metal
```

The unit tests in `tests/unit/` check the joint oracle, the openings, and the
commitments.
