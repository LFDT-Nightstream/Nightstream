# Neo — Lattice‑based Folding + Hash‑MLE SNARK

Neo is an end‑to‑end **post‑quantum** proving system that couples a lattice‑based folding scheme with a **Hash‑MLE** polynomial commitment SNARK. It targets circuits expressed as **Customizable Constraint Systems (CCS)** over Goldilocks field, supports efficient recursion, and avoids elliptic curves, pairings, and FRI. The design is informed by recent folding systems (e.g., Nova/HyperNova) and adapts them to a lattice setting with a practical Hash‑MLE backend.

> **🚧 Status**: Research prototype with working end‑to‑end **prove/verify** for small CCS programs. Security checks and transcript binding are implemented. Currently optimizing proof size and memory usage.

---

## Why Neo?

* **🔒 Post‑quantum security**: **Hash‑MLE** polynomial commitment scheme using only hash functions and multilinear extension evaluations
* **⚡ Optimized for modern fields**: CCS over Goldilocks provides excellent prover performance with **degree‑2 extension field** for sum‑check soundness
* **🎯 Simple API**: Clean two‑function interface (`neo::prove` and `neo::verify`) hides complexity while maintaining full functionality  
* **🔐 Cryptographic hygiene**: Unified **Poseidon2** transcript across folding and Hash‑MLE phases with anti‑replay protection

---

## Quick Start

### Prerequisites
* **Rust** ≥ 1.88 (MSRV; CI uses stable channel)
* Sufficient RAM for demo proofs (~2GB recommended)
* `git` and `clang` (for optimized builds)

### One-line demo
```bash
cargo run -p neo --example fib --release
```

### Build & test everything
```bash
cargo build --release                    # Build all crates
cargo test --workspace                   # Run comprehensive test suite
cargo run -p neo --example fib --release # Demo end-to-end Fibonacci proof
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
    let proof = prove(ProveInput {
        params: &params,
        ccs: &ccs,
        public_input: &public_input,
        witness: &witness,
    })?;
    
    println!("✅ Proof generated! Size: {} bytes", proof.size());

    // 3) Verify proof  
    let is_valid = verify(&ccs, &public_input, &proof)?;
    println!("🔍 Verification result: {}", if is_valid { "PASSED" } else { "FAILED" });
    
    Ok(())
}
```

---

## Architecture & Pipeline

Neo implements a four-stage proving pipeline:

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
  neo/                  # 🎯 Main API: prove() and verify() functions
  neo-ajtai/            # 🔐 Lattice (Ajtai) commitments over module-SIS  
  neo-ccs/              # ⚙️  Customizable Constraint Systems, matrices, utilities
  neo-fold/             # 🔄 Folding pipeline: CCS→RLC→DEC reductions + transcripts
  neo-spartan-bridge/   # 🌉 ME → Spartan2 R1CS conversion with Hash-MLE PCS
  neo-math/             # 🧮 Field arithmetic, rings, polynomial operations
  neo-challenge/        # 🎲 Challenge generation and strong sets
  neo-params/           # ⚙️  Parameter management and security validation

crates/neo/examples/
  fib.rs                # 📚 Complete Fibonacci sequence proof demo
```

---

## Examples

### Fibonacci Demo
```bash
cargo run -p neo --example fib --release
```

This demonstrates proving correctness of a Fibonacci computation (`z[i+2] = z[i+1] + z[i]`) as a CCS program, running the complete folding pipeline, and verifying the final SNARK.

**What you'll see:**
- CCS constraint system generation (8 Fibonacci steps)
- Witness generation and local verification  
- Neo SNARK proof generation with timing
- Succinct proof verification 
- Performance metrics and proof size

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

| Metric | Small Circuits | Current Status |
|--------|---------------|----------------|
| **Proof Size** | ~1-10KB | Unoptimized (verbose public IO) |
| **Prover Time** | ~10-100ms | Prototype performance |
| **Verifier Time** | ~1-10ms | Constant (SNARK property) |
| **Memory Usage** | ~100MB | Materializes one-hot vectors |

### Optimization Roadmap
- [ ] **Sparse weight vectors** in bridge (currently quadratic in `d·m`)
- [ ] **Compact public IO** encoding (currently verbose for debugging)
- [ ] **Parallel proving** optimizations
- [ ] **Memory efficiency** improvements

---

## Development & Testing

### Running Tests
```bash
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

### Benchmarking
```bash
# Install criterion if needed
cargo install cargo-criterion

# Run benchmarks
cargo criterion --bench folding --warm-up-time 10 --measurement-time 30
```

### Parameter Validation (Optional)
```bash
# Validate lattice security parameters
sage sage_params.sage
```

---

## Roadmap

### Near Term (Next Release)
- [ ] **Explicit parameter threading** (remove global PP state)
- [ ] **Sparse bridge representation** (reduce memory footprint)
- [ ] **Typed public-IO validation** (bind verifier parameters to proof)
- [ ] **Comprehensive benchmarks** (end-to-end and per-phase)

### Medium Term  
- [ ] **Recursive proof composition** (proof-carrying state)
- [ ] **Additional circuit examples** (Merkle trees, VDF gadgets)
- [ ] **Performance optimizations** (parallel operations, memory efficiency)
- [ ] **Security audit preparation** (hardened parameter sets)

### Long Term
- [ ] **Production deployment tools** (parameter generation, key management)
- [ ] **Higher-level circuit DSL** (user-friendly constraint specification)
- [ ] **Integration libraries** (blockchain, application frameworks)

---

## Platform Support

### Native ARM64 Compatibility ✅
Neo fully supports **ARM64 architectures** including Apple Silicon (M1/M2/M3/M4) and ARM64 Linux:

```bash
# Optimal ARM64 performance
export RUSTFLAGS="-C target-cpu=native"
cargo build --release
```

**Benefits of field-native design:**
- No problematic elliptic curve dependencies
- Direct Goldilocks operations with NEON optimizations  
- Simplified cross-platform deployment

---

## Contributing

We welcome contributions! Please:

* **Add tests** for behavioral changes
* **Run formatting**: `cargo fmt` and `cargo clippy`  
* **Keep logs informative** but concise (parameter hashes, guard values, sizes)
* **Update documentation** for API changes

### Development Setup
```bash
git clone <repo-url>
cd neo
cargo build --workspace
cargo test --workspace
```

---

## References & Background

### Academic Foundation
* **Neo Paper**: "Lattice-based folding scheme for CCS over small fields" (Nguyen & Setty, 2025)
* **Nova**: Recursive arguments from folding schemes ([project page](https://github.com/Microsoft/Nova))
* **HyperNova**: CCS extensions and improved ergonomics
* **Poseidon2**: Sponge-friendly permutation for transcript hashing
* **Sum-check Protocol**: Multilinear evaluation verification ([primer](https://xn--2-umb.com/24/sumcheck/))

### Related Work
Understanding folding SNARKs? Start with [Nova's overview](https://github.com/Microsoft/Nova) for conceptual background, then see [Poseidon2 documentation](https://dev.ingonyama.com/icicle/primitives/poseidon2) for transcript primitives.

---

## License

Licensed under your choice of:
* **MIT License** ([LICENSE-MIT](LICENSE-MIT))
