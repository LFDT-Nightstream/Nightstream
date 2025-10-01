# Nightstream — Lattice‑based Folding

[![GitHub License](https://img.shields.io/github/license/nicarq/halo3)](LICENSE)
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/00000/badge)](https://bestpractices.coreinfrastructure.org/projects/00000)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/nicarq/halo3/badge)](https://scorecard.dev/viewer/?uri=github.com/nicarq/halo3)

Nightstream is an end‑to‑end **post‑quantum** proving system that couples a lattice‑based folding scheme with a **Hash‑MLE** polynomial commitment SNARK. It targets circuits expressed as **Customizable Constraint Systems (CCS)** over Goldilocks field, supports efficient recursion, and avoids elliptic curves, pairings, and FRI. The design is informed by recent folding systems (e.g., Nova/HyperNova) and adapts them to a lattice setting with a practical Hash‑MLE backend.

Nightstream is an implementation of the protocol introduced in the Neo paper "Lattice‑based folding scheme for CCS over small fields" (Nguyen & Setty, 2025); see References & Background → Academic Foundation.

> **🚧 Status**: Research prototype with end‑to‑end **prove/verify** and IVC/NIVC demos. Security guards and canonical transcript/public‑IO binding are implemented. Lean proofs (VK registry) shipped; sparse weights and memory reductions are in progress.

---

## Why Nightstream?

* **🔒 Post‑quantum security**: **Hash‑MLE** polynomial commitment scheme (currently) using only hash functions and multilinear extension evaluations
* **⚡ Optimized for modern fields**: CCS over Goldilocks provides excellent prover performance with **degree‑2 extension field** for sum‑check soundness
* **🎯 Simple API**: Clean two‑function interface (`neo::prove` and `neo::verify`) hides complexity while maintaining full functionality  
* **🔐 Cryptographic hygiene**: Unified **Poseidon2** transcript across folding and Hash‑MLE phases with anti‑replay protection

---

## Quick Start

### Prerequisites
* **Rust** ≥ 1.88 (MSRV; CI uses stable channel)
* Sufficient RAM for demo proofs (~2GB recommended)
* `git` and a C compiler (gcc or clang) for the mimalloc allocator

### One-line demo
```bash
# Option A: run the benchmarked demo
cargo run -p neo --example fib_benchmark --release

# Option B: use the CLI to generate and verify a Fibonacci proof
cargo run -p neo-cli -- gen -n 32 --release              # writes fib_proof.bin
cargo run -p neo-cli -- verify -f fib_proof.bin --release
```

### Build & test everything
```bash
cargo build --release                           # Build all crates
cargo test --workspace                          # Run comprehensive test suite
cargo run -p neo --example fib_benchmark --release # Demo end-to-end Fibonacci proof
```

---

## Simple API Usage

```rust
use neo::{prove, verify, ProveInput, CcsStructure, NeoParams, F};
use anyhow::Result;

fn main() -> Result<()> {
    // 1) Define your CCS constraint system and witness
    let ccs: CcsStructure<F> = build_your_circuit(); // Your constraint system
    let witness: Vec<F> = generate_witness();         // Satisfying witness  
    let public_input: Vec<F> = vec![];                // Usually empty
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2); // Auto-tuned params

    // 2) Generate proof
    let proof = prove(ProveInput::new(
        &params,
        &ccs,
        &public_input,
        &witness,
        &[],  // output_claims (optional public outputs)
    ))?;
    
    println!("✅ Proof generated! Size: {} bytes", proof.size());

    // 3) Verify proof  
    let is_valid = verify(&ccs, &public_input, &proof)?;
    println!("🔍 Verification result: {}", if is_valid { "PASSED" } else { "FAILED" });
    
    Ok(())
}
```

---

## Architecture & Pipeline

Nightstream implements a four-stage proving pipeline:

### 1. **Ajtai Commitment (Module-SIS)**
The witness undergoes base-`b` decomposition and lattice commitment, establishing the linear algebra foundation for subsequent reductions.

### 2. **Folding Pipeline: Π_CCS → Π_RLC → Π_DEC**
* **Π_CCS**: Reduces CCS satisfaction to multilinear evaluation (**ME**) claims over extension field `K = F²`
* **Π_RLC**: Folds multiple ME instances via random linear combination with transcript-bound soundness
* **Π_DEC**: Decomposes folded witness into base-`b` digits with verified range constraints

### 3. **Bridge to Spartan2**  
The final ME claim converts to Spartan2 R1CS format with **Hash-MLE** polynomial commitments and unified **Poseidon2** transcripts.

### 4. **SNARK Generation**
Spartan2 produces constant-size proofs with logarithmic verification time and post-quantum security guarantees.

---

## Repository Structure

```
crates/
  neo/                  # 🎯 Main API: prove() and verify() functions + IVC/NIVC
  neo-ajtai/            # 🔐 Lattice (Ajtai) commitments over module-SIS  
  neo-ccs/              # ⚙️  Customizable Constraint Systems, matrices, utilities
  neo-fold/             # 🔄 Folding pipeline: CCS→RLC→DEC reductions + transcripts
  neo-spartan-bridge/   # 🌉 ME → Spartan2 R1CS conversion with Hash-MLE PCS
  neo-math/             # 🧮 Field arithmetic, rings, polynomial operations
  neo-challenge/        # 🎲 Challenge generation and strong sets
  neo-params/           # ⚙️  Parameter management and security validation
  neo-transcript/       # 📝 Poseidon2 transcript for Fiat-Shamir
  neo-cli/              # 🔧 Command-line interface
  neo-tests/            # ✅ Integration tests
  neo-redteam-tests/    # 🔴 Security and attack tests
  neo-quickcheck-tests/ # ⚡ Property-based tests

crates/neo/examples/
  fib_benchmark.rs      # 📊 Fibonacci SNARK benchmark (various sizes)
  fib_folding_nivc.rs   # 🔄 Fibonacci with IVC folding
  incrementer_folding.rs # 🔢 Simple incrementer IVC example
  nivc_demo.rs          # 🎯 Non-uniform IVC (NIVC) demo
```

---

## Examples

### Fibonacci Benchmark
```bash
cargo run -p neo --example fib_benchmark --release
```

This benchmarks Nightstream SNARK proving for Fibonacci sequences of various sizes, demonstrating efficient sparse matrix construction. Still needs some more work to be standarized.

**What you'll see:**
- Sparse CCS constraint system generation
- Witness generation and validation
- Nightstream SNARK proof generation with detailed timing breakdown
- Proof verification
- Performance metrics and proof size

### Fibonacci with IVC Folding
```bash
cargo run -p neo --example fib_folding_nivc --release
```

Demonstrates incrementally verifiable computation (IVC) with Fibonacci steps, using Nova-style folding.

### Incrementer Example
```bash
cargo run -p neo --example incrementer_folding --release
```

Simple state machine example showing NIVC (non-uniform IVC) API: state transitions with proof-carrying computation.

### NIVC Multi-Lane Demo
```bash
cargo run -p neo --example nivc_demo --release
```

Shows heterogeneous step types in non-uniform IVC, demonstrating multiple computation types in a single proof chain.

---

## Security & Correctness

### ✅ Implemented Safeguards
* **Parameter validation**: Enforces RLC soundness inequality `(k+1)·T·(b-1) < B` before proving
* **Fail-fast CCS checks**: Early witness validation with clear error reporting
* **Transcript binding**: Anti-replay protection via canonical public-IO headers
* **Constant-time verification**: Prevents timing side-channels in proof validation
* **Cryptographically secure RNG**: Production builds use `OsRng`; debug builds use deterministic seeds for reproducibility

### ⚠️ Current Limitations
* WIP

### 🔬 Security Posture
> **Research software warning**: This implementation demonstrates the Neo protocol but requires independent security review before production deployment. Do not use it.

---

## Performance Profile

| Metric | Current Implementation | Target (Post-Optimization) |
|--------|----------------------|---------------------------|
| **Proof Size** | ~500kb | Unoptimized |
| **Prover Time** | ~25ms (per folding) | TBD |
| **Verifier Time** | ~7ms | ~1-10ms |
| **Memory Usage** | TBD | TBD |

### Optimization Roadmap
- [ ] **Sparse weight vectors** in bridge (currently quadratic in `d·m`)
- [x] **Compact public IO** encoding (canonical, padded-before-digest)
- [x] **Parallel proving** optimizations (CSR SpMV, Ajtai rows)
- [ ] **Memory efficiency** improvements (ongoing)

---

## Development & Testing

### Running Tests
```bash
# All tests with extra guards and verbose debugging
NEO_PI_CCS_PREFLIGHT=1 NEO_SELF_VERIFY=1 NEO_STRICT_IO_PARITY=1 cargo test --release --workspace --features "testing,neo_dev_only,debug-logs,fs-guard,redteam,quickcheck"

# Core functionality
cargo test -p neo-fold -- --nocapture
cargo test -p neo-spartan-bridge -- --nocapture
cargo test -p neo-ccs -- --nocapture

# Security validation  
cargo test -p neo-ajtai red_team -- --nocapture
cargo test -p neo-fold security_validation -- --nocapture

# End-to-end integration
cargo test --workspace
```

### Parameter Validation (Optional)
```bash
# Validate lattice security parameters
sage sage_params.sage
```

---

## Roadmap

### Near Term (Next Release)
- [ ] **Explicit parameter threading** (remove global PP state) — *Still uses `OnceLock` registry; concurrent proving limited*
- [~] **Sparse bridge representation** (reduce memory footprint) — *CSR SpMV implemented; weight vector optimization ongoing*
- [x] **Typed public-IO validation** (bind verifier parameters to proof) — *Context digest binding complete*
- [ ] **Comprehensive benchmarks** (end-to-end and per-phase) — *Only example benchmarks; no criterion suite yet*
- [ ] **Lookup arguments** (table constraints for range checks, bitwise ops) — *Planned for CCS extension*

### Medium Term  
- [x] **Recursive proof composition** (proof-carrying state) — *IVC/NIVC fully functional*
- [~] **Additional circuit examples** (Merkle trees, VDF gadgets) — *Have: Fibonacci, incrementer; Need: Merkle, VDF*
- [~] **Performance optimizations** (parallel operations, memory efficiency) — *Rayon parallelization done; memory work ongoing*
- [ ] **GPU acceleration exploration** (Mojo for matmul/MSM) — *Exploratory: GPU-accelerated matrix operations*
- [~] **Security audit preparation** (hardened parameter sets) — *Track B security tests in progress; not audit-ready*

### Long Term
- [ ] **Production deployment tools** (parameter generation, key management)
- [ ] **Higher-level circuit DSL** (user-friendly constraint specification)
- [ ] **Integration libraries** (blockchain, application frameworks)

---

## Contributing

We welcome contributions to Nightstream! Please:

* **Add tests** for behavioral changes
* **Run formatting**: `cargo fmt` and `cargo clippy`  
* **Keep logs informative** but concise (parameter hashes, guard values, sizes)
* **Update documentation** for API changes

### Development Setup
```bash
git clone <repo-url>
cd halo3
cargo build --workspace
cargo test --workspace
```

---

## References & Background

### Academic Foundation
* **Neo Protocol**: Wilson Nguyen & Srinath Setty, "[Neo: Lattice-based folding scheme for CCS over small fields and pay-per-bit commitments](https://eprint.iacr.org/2025/294)" (ePrint 2025/294)
  - Introduces the pay-per-bit Ajtai commitment scheme this implementation uses
  - Adapts HyperNova folding to lattice setting with Goldilocks field support
  - Post-quantum secure using lattices
* **Spartan**: Srinath Setty, "Spartan: Efficient and general-purpose zkSNARKs without trusted setup" (CRYPTO 2020)
  - Sum-check based zkSNARK providing our final proof compression layer
  - Linear-time polynomial IOP with succinct verification
* **Nova**: Recursive arguments from folding schemes ([project page](https://github.com/Microsoft/Nova))
* **HyperNova**: CCS extensions and improved ergonomics
* **Plonky3**: Modular STARK/PLONK framework ([Succinct Labs](https://github.com/succinctlabs/plonky3))
  - Provides Goldilocks field implementation with SIMD optimizations (AVX2, AVX-512, NEON)
  - Poseidon2 hash function for Fiat-Shamir transcripts
  - Challenger/symmetric crypto primitives used throughout Nightstream
* **Sum-check Protocol**: Multilinear evaluation verification ([primer](https://xn--2-umb.com/24/sumcheck/))
* **Sum-check Optimizations**: Bagad et al., "[Speeding Up Sum-Check Proving](https://eprint.iacr.org/2025/1117)" (ePrint 2025/1117)

### Related Work & Implementation
* **[Spartan2](https://github.com/microsoft/Spartan2)**: Generic Spartan implementation supporting multiple polynomial commitment schemes (elliptic curve, hash-based, and lattice-based including Greyhound). Nightstream uses Spartan2's Hash-MLE backend with P3 integration for the final proof compression.
* **[Plonky3](https://github.com/succinctlabs/plonky3)**: Nightstream builds on Plonky3's field arithmetic (`p3-goldilocks`), Poseidon2 implementation (`p3-poseidon2`, `p3-symmetric`), and challenger/transcript primitives (`p3-challenger`) for consistent cryptographic operations across the stack.
* **[Merlin](https://github.com/dalek-cryptography/merlin)**: Composable proof transcripts for public-coin arguments. Nightstream's transcript API design (`neo-transcript`) draws inspiration from Merlin's domain separation, message framing, and challenge generation patterns, adapted for Poseidon2-based Fiat-Shamir transforms.
* **Understanding folding SNARKs**: Start with [Nova's overview](https://github.com/Microsoft/Nova) for conceptual background on folding schemes, then review the [Neo paper](https://eprint.iacr.org/2025/294) for lattice-specific adaptations.

---

## Governance & Policies

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Maintainers](MAINTAINERS.md)
- [Adopters](ADOPTERS.md) — please add verifiable references when your usage becomes public.

> **Note:** Update the OpenSSF Best Practices badge once a project ID is issued (replace `00000` with the assigned identifier).

---

## License

Licensed under the terms of the [Apache License, Version 2.0](LICENSE).
