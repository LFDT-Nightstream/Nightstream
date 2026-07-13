# iOS Demo (TestingWasm) — run in Simulator

## Prereqs
- Xcode (and iOS Simulator)
- Rust toolchain (`rustup`, `cargo`)
- `wasm-pack` (`cargo install wasm-pack`)

## Steps
1. `cd demos/ios-demo`

2. Build the WASM bundles (used by the app):
   - `./scripts/build_wasm.sh --release`
   - Optional (threads bundle for WKWebView): `./scripts/build_wasm.sh --release --threads`

3. Build the Metal benchmark XCFramework:
   - `./scripts/build_metal_bench.sh --release`

4. Open the Xcode project:
   - `open TestingWasm.xcodeproj`

5. Select a physical iPhone for performance measurements, then press **Run**. The simulator is a correctness target only.

## Quick build (CLI)
- Rebuild Metal benchmark + build the app: `./scripts/build.sh --metal`
- Rebuild wasm + build the app: `./scripts/build.sh --wasm`
- Rebuild everything + build the app: `./scripts/build.sh --all`

## Troubleshooting
- Metal tab says the benchmark is unavailable: confirm `Frameworks/NeoMetalBench.xcframework` exists, then clean and rebuild.

The full profile reports M5 stage, residency, and deferred-carrier measurements
for the four-chunk SHA-256 and two-step Nebula memory lifecycles. **M6 soak**
adds five measured SHA lifecycles plus independent 60-second CPU and Metal
runs. It requires byte-identical proofs, 1.5x median and 1.25x p95 SHA speedup,
and at least 1.15x sustained Metal throughput.

The M6 contract passes on the Apple M5 Max development host. That result does
not establish iPhone performance: run **M6 soak** on a physical device before
making crossover, energy, or thermal claims. The two-step Nebula fixture is a
small parity and command-overhead probe; SHA-256 is the required crossover
workload.
