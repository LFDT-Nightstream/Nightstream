# neo-prover-metal

Native Metal substrate for a byte-identical Nightstream prover on Apple GPUs.
The canonical protocol remains in Rust; macOS and iOS use the same MSL source
and compile separate SDK-specific `.metallib` artifacts.

Current slice:

- exact Goldilocks add, subtract, multiply, and low-norm multiply;
- exact quadratic-extension multiplication;
- serial Poseidon2 permutation, stateless hashing, and transcript operations;
- direct Metal device/queue/pipeline ownership through `objc2-metal`;
- a macOS GPU parity test against canonical Rust values.

Build shader libraries on a Mac with Xcode installed:

```bash
./scripts/build_metal_shaders.sh --sdk macosx --out dist/metal/macos/nightstream.metallib
./scripts/build_metal_shaders.sh --sdk iphoneos --out dist/metal/ios/nightstream.metallib
```

Run the first device parity gate on Apple Silicon macOS:

```bash
timeout 300s cargo test -p neo-prover-metal --release --features metal --test apple_parity -- --nocapture
```

This is not yet a `NifsProverAdapter`. The next slices are a resident Metal
session and transcript, followed by Ajtai, Pi_CCS, Pi_RLC, Pi_DEC, and the
deferred fold-output carrier. No stage may claim completion until its outputs
and final proof bytes match the CPU backend.
