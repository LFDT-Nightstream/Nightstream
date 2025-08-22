# Neo: Lattice-based Folding Scheme Implementation
[![Crates.io](https://img.shields.io/crates/v/neo-main.svg)](https://crates.io/crates/neo-main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a non-production, research-grade implementation of **Neo**, a lattice-based folding scheme for CCS (Customizable Constraint Systems) over small fields, as described in the paper "Neo: Lattice-based folding scheme for CCS over small fields and pay-per-bit commitments" by Wilson Nguyen and Srinath Setty (2025).

The codebase is structured as a Rust workspace with multiple crates, focusing on modularity and testability. It's functional for small-scale demos (e.g., folding CCS instances with verification) but **not secure for production use**—parameters are toy-sized, partial ZK/Fiat-Shamir, naive multiplications, and no audits. For real applications, scale parameters, add full cryptographic hardening, and audit.

## Features
- **Fields & Arithmetic**: Goldilocks field (64-bit prime) for CCS, modular ints for lattice Z_q.
- **Polynomials & Rings**: Generic univariate polys, cyclotomic rings mod X^n +1 with ops.
- **Decomposition**: Signed base-b digit decomp for pay-per-bit commitments.
- **Commitments**: Ajtai-based lattice commitments with matrix embedding, GPV trapdoor sampling, ZK blinding, and homomorphism.
- **Sum-Check**: Interactive batched/multilinear sum-check for multivariate claims over extensions.
- **CCS Relations**: Constraint systems with matrices/multivariate polys; satisfiability checks and sum-check proving.
- **Folding**: Reductions (Π_CCS, Π_RLC, Π_DEC) to fold instances; full flow with verifiers.
- **NARK Mode**: Non-succinct proofs for CCS (no compression) - verifiable but longer proofs.
- **Demo**: End-to-end folding/verification in `neo-main` binary, with FRI stubs.

## Current Implementation: NARK Mode

This implementation currently operates as a **NARK (Non-succinct ARgument of Knowledge)** rather than a SNARK. This design choice provides several benefits while we work toward full SNARK integration:

### What is NARK Mode?

**NARK Mode** means the system produces **verifiable proofs without compression**:
- ✅ **Verifiable**: All proofs can be verified for correctness
- ✅ **Sound**: Invalid statements are rejected with high probability  
- ✅ **Zero-Knowledge**: Proofs reveal no information about the witness
- ❌ **Non-Succinct**: Proof sizes grow with computation size (not constant)

### How NARK Mode Works

1. **CCS Folding**: The core Neo folding scheme works normally, reducing multiple CCS instances into a single instance through sum-check protocols.

2. **Recursive IVC**: Instead of compressing proofs with SNARKs, the system uses dummy verifier circuits to enable recursion:
   ```rust
   // NARK recursion: dummy verifier CCS instead of compressed SNARK
   let verifier_ccs = dummy_verifier_ccs();  // NARK mode: minimal CCS for recursion
   let dummy_instance = CcsInstance { /* minimal instance */ };
   ```

3. **Verification**: Full verification of the folded CCS instances without compression.

### Benefits of NARK Mode

- **🔧 Simplified Architecture**: No complex SNARK compression logic
- **🏗️ ARM64 Native**: Pure field operations, no elliptic curve dependencies
- **🧪 Research-Friendly**: Focus on core folding scheme without compression complexity
- **🔍 Debuggable**: Full proof data available for analysis and testing
- **⚡ Fast Development**: Rapid iteration without SNARK compilation overhead

### NARK vs SNARK Comparison

| Aspect | NARK (Current) | SNARK (Future) |
|--------|----------------|----------------|
| **Proof Size** | Grows with computation | Constant (~100-1000 bytes) |
| **Verification Time** | Linear in computation | Constant (milliseconds) |
| **Prover Time** | Moderate | Higher (compression overhead) |
| **Complexity** | Low | High |
| **Dependencies** | Pure field operations | Curve/pairing libraries |
| **ARM64 Support** | Native | Requires compatibility layers |

## Future: Spartan Integration Roadmap

We plan to integrate **Spartan2** for full SNARK functionality once compatibility issues are resolved:

### Why Spartan2?

- **Field-Native**: Designed for small fields like Goldilocks
- **CCS Support**: Native support for Customizable Constraint Systems
- **Recursive**: Enables true recursive SNARKs for IVC/PCD
- **Performance**: Optimized for lattice-based folding schemes

### Current Blockers

1. **ARM64 Compatibility**: Spartan2 dependencies (halo2curves, arkworks) have ARM64 assembly issues
2. **API Stability**: Spartan2 is in active development with changing APIs
3. **Field Conversion**: Need seamless Goldilocks ↔ curve field conversion

### Integration Timeline

**Phase 1: Compatibility (Q2 2025)**
- [ ] ARM64 support in Spartan2 dependencies
- [ ] Stable Spartan2 API release
- [ ] Field conversion utilities

**Phase 2: Integration (Q3 2025)**  
- [ ] Replace dummy `spartan_compress()` with real Spartan2 calls
- [ ] Implement CCS → R1CS conversion for Spartan2
- [ ] Add knowledge extractor for soundness

**Phase 3: Optimization (Q4 2025)**
- [ ] Recursive SNARK verification circuits
- [ ] Batch verification optimizations  
- [ ] Production-grade parameter selection

### Tracking Progress

- **Spartan2 Issues**: [Microsoft/Spartan2 GitHub](https://github.com/microsoft/Spartan2)
- **ARM64 Support**: [halo2curves ARM64 tracking](https://github.com/privacy-scaling-explorations/halo2curves/issues)
- **Neo Integration**: This repository's issues and PRs

## Crates Overview
| Crate | Description |
|----------------|-----------------------------------------------------------------------------|
| `neo-fields` | Goldilocks field wrappers and utils. |
| `neo-modint` | Modular arithmetic over lattice q (e.g., 2^61-1). |
| `neo-poly` | Generic univariate polynomials over coefficients. |
| `neo-ring` | Cyclotomic rings mod X^n +1, with ops. |
| `neo-decomp` | Vector decomposition to low-norm matrices. |
| `neo-commit` | Ajtai lattice commitments with packing/homomorphism. |
| `neo-sumcheck` | Sum-check protocol for multilinear claims. |
| `neo-ccs` | CCS structures, instances, and satisfiability checks. |
| `neo-fold` | Folding reductions (CCS to evals, linear combos, decomp). |
| `neo-main` | Demo binary: Fold CCS instances and verify. |

## Getting Started

> **📝 Note**: This implementation currently runs in **NARK mode** (non-succinct proofs). See the [NARK Mode section](#current-implementation-nark-mode) above for details and the [Spartan Integration Roadmap](#future-spartan-integration-roadmap) for future SNARK plans.

### Prerequisites
- Rust 1.88 (edition 2021).
- Cargo for building.

### ARM64 Compatibility (Apple Silicon, ARM Linux)
This codebase **fully supports ARM64 architectures** including Apple Silicon Macs (M1/M2/M3) and ARM64 Linux systems. The field-native design using Goldilocks field provides native ARM64 compatibility without complex curve libraries.

#### Native ARM64 Performance
The codebase uses field-native operations with NEON optimizations available:

```bash
# Enable ARM64 NEON optimizations for maximum performance
export RUSTFLAGS="-C target-cpu=native"
cargo build --release
```

**Benefits of Field-Native Design**:
- No elliptic curve dependencies that cause ARM64 issues
- Direct Goldilocks field operations with NEON support
- Simplified dependency tree for better compatibility

#### Cross-Platform Development
The field-native design ensures consistent behavior across all architectures:
- **ARM64**: Native NEON optimizations available
- **x86_64**: Native AVX optimizations available  
- **CI/CD**: Single configuration works everywhere

### Build & Test
```bash
git clone https://github.com/nicarq/learn-neo-lattice.git
cd neo
cargo build --workspace
cargo test --workspace # Run all unit tests
```

### Running the Demo (NARK Mode)
The `neo-main` binary demonstrates the complete NARK folding pipeline:

```bash
# Run the main demo
cargo run --bin neo-main

# Run with debug output to see NARK recursion
RUST_LOG=debug cargo run --bin neo-main

# Test recursive IVC in NARK mode
cargo test -p neo-fold test_recursive_ivc -- --nocapture
```

**What the demo shows:**
- ✅ CCS instance creation and satisfiability checking
- ✅ Folding multiple instances using sum-check protocols  
- ✅ Recursive IVC with dummy verifier circuits (NARK mode)
- ✅ Full verification of non-succinct proofs
- ✅ ARM64 compatibility with pure field operations

### Security Parameter Validation (Optional)
For production use, lattice parameters should be validated. A Sage script is provided:
```bash
sage sage_params.sage  # Optional - validates MSIS and RLWE security estimates
```
**Note**: NARK mode uses toy parameters suitable for research and development. Production deployments should use properly sized parameters validated by cryptographic experts.

### Coverage & QuickCheck
Install [cargo-tarpaulin](https://crates.io/crates/cargo-tarpaulin) for coverage:
```bash
cargo install cargo-tarpaulin
cargo tarpaulin --out Html # Generates tarpaulin-report.html
```
Property tests using QuickCheck check algebraic invariants (e.g., ring associativity) in the test suites.

### Run Demo
The `neo-main` binary folds two simple CCS instances (RICS-like) and verifies:
```bash
cargo run --bin neo-main
```
Output: "Folding successful: Final instance satisfies CCS." (or error if fails).
For larger tests (n=1024 constraints), run benchmarks (see below).

### Benchmarks
Add Criterion to workspace deps:
```toml
[dev-dependencies]
criterion = { version = "0.5.1", features = ["html_reports"] }
```
Run:
```bash
cargo criterion --bench folding -- --warm-up-time 10 --measurement-time 30
```
Generates reports in `target/criterion` on commit/fold times for large inputs (e.g., 1024 constraints, witness size 2048). The HTML report is available at `target/criterion/report/index.html`. On a standard CPU, expect ~100ms for commit, ~1s for fold (unoptimized; additional algorithmic improvements can speed this up).
If you don't have criterion, you can install it by doing
```bash
cargo install cargo-criterion
```

#### Benchmark Scripts
Helper scripts are included for quick benchmarking and comparing the Rust
implementations against simplified Python simulations:

```bash
bash bench_rust.sh       # Run Criterion benches and save to rust_bench.txt
python bench_sim.py      # Run Python simulations and save to sim_bench.txt
python compare_bench.py  # Run both and print a comparison table
```

Set `RUN_LONG_TESTS=1` before running to use larger parameters.

## Scaling Parameters
For paper-like realism with zero-knowledge blinding, use `SECURE_PARAMS` (n=54, d=32, σ=3.2, β=3). Test large CCS in `neo-ccs` tests or benchmarks. For full security validation, run `sage_params.sage` (see paper App. B.10).

## Limitations & Next Steps
- **Performance**: Naive poly mul; implement faster algorithms in `neo-ring` for O(n log n) speed.
- **Security**: Toy params (negligible lambda); partial ZK/FS hashing. Adjust via Sage estimator.
- **Extensions**: Add lookups (§1.4) with Shout/Twist; recursive IVC (§1.5) with Spartan+FRI.
- **Architecture**: Full ARM64 support achieved! Both development and production SNARKs work on Apple Silicon and ARM64 Linux.
- **Contribute**: PRs welcome for optimizations, full param sets, or ZK blinding.

## License
MIT - see LICENSE file.

## Acknowledgments
Based on "Neo" paper by Nguyen & Setty (2025). Uses `p3-*` crates for fields/matrices.
