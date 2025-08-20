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
- **Demo**: End-to-end folding/verification in `neo-main` binary, with FRI stubs.

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
### Prerequisites
- Rust 1.88 (edition 2021).
- Cargo for building.

### Build & Test
```bash
git clone https://github.com/nicarq/learn-neo-lattice.git
cd neo
cargo build --workspace
cargo test --workspace # Run all unit tests
```

### Security Parameter Validation
To sanity-check the lattice parameter choices for `SECURE_PARAMS`, run the Sage script:
```bash
sage sage_params.sage
```
It prints rough MSIS and RLWE security estimates and asserts they exceed 128 bits.

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
- **Contribute**: PRs welcome for optimizations, full param sets, or ZK blinding.

## License
MIT - see LICENSE file.

## Acknowledgments
Based on "Neo" paper by Nguyen & Setty (2025). Uses `p3-*` crates for fields/matrices.
