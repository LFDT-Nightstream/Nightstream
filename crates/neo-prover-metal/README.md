# neo-prover-metal

Native Metal NIFS acceleration for Apple GPUs. Rust owns transcript order,
proof assembly, and verification. `MetalSession` owns the device, production
shader library, and reusable proof plans.

Build the production shader library for an Apple SDK:

```bash
./scripts/build_metal_shaders.sh --sdk macosx --out dist/metal/macos/nightstream.metallib
./scripts/build_metal_shaders.sh --sdk iphoneos --out dist/metal/ios/nightstream.metallib
```

Run the production arithmetic and NIFS parity gates on Apple Silicon macOS:

```bash
timeout 300s cargo test -p neo-prover-metal --release --features metal --test metal_arithmetic -- --nocapture
timeout 300s cargo test -p neo-prover-metal --release --features metal --test nifs_adapter -- --nocapture
```
