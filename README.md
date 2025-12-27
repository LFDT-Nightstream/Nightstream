# Nightstream — Lattice‑based Folding with Twist/Shout Memory

[![GitHub License](https://img.shields.io/github/license/nicarq/nightstream)](LICENSE)

Nightstream is a **post‑quantum** proving system built around a lattice-based folding scheme for **CCS** plus sum-check–based memory arguments (Twist/Shout). It supports shard-level folding and emits **final ME obligations** (main + value lanes) intended to be checked by an outer finalization layer.

- **Twist** for read/write memory
- **Shout** for read‑only lookups

It targets CCS over the **Goldilocks** field; uses a degree‑2 extension for sum-check soundness where required, and is designed for zkVM-style workloads via **shard-level folding**.

Nightstream implements the protocol from the Neo paper "Lattice‑based folding scheme for CCS over small fields" (Nguyen & Setty, 2025), extended with Twist & Shout memory arguments.

> **🚧 Status**: Research prototype. Shard folding loop and Twist/Shout integration (including two-lane obligations) are implemented. Verifier returns `ShardObligations { main, val }`; both must be enforced by the final layer. Not production-ready.

---

## What Works Today

- ✅ Shard prove/verify loop with shared transcript binding
- ✅ Twist/Shout integrated per chunk, including two-lane obligations
- ✅ End-to-end integration tests proving and verifying shards
- ⚠️ Final obligation finalizer (outer SNARK / recursion layer) is WIP
- ⚠️ No audit; research-grade performance/side-channel posture

**Obligations** = ME (single-point evaluation) claims emitted by shard verification that must be checked by the final proof layer.

---

## Quick Start

### Prerequisites
* **Rust** (stable, or use `rust-toolchain.toml` if present)
* `git`
* C compiler (only if enabling allocators like mimalloc)

### Build & Smoke Test

```bash
cargo build --release

# Canonical smoke test: shard prove/verify with Twist/Shout
cargo test -p neo-fold full_folding_integration --release -- --nocapture

# Full workspace tests
cargo test --workspace --release
```

### Focused Test Runs

```bash
cargo test -p neo-fold --release
cargo test -p neo-memory --release
cargo test -p neo-reductions --release
```

### Paper-exact Reference Mode

```bash
cargo test -p neo-reductions --features paper-exact --release
```

---

## Architecture Overview

Nightstream implements **shard-level folding** where each step processes one CCS chunk together with its matching Twist/Shout instances, all sharing sum-check challenges.

### Per-Step Folding Flow

**Diagram key:**
- `r_time`: shared evaluation point used to batch/align per-step checks across CCS/Twist/Shout
- `r_val`: Twist-only point used by the value-evaluation subprotocol; tracked separately

```
┌───────────────────────────────────────────────────────────────────────────┐
│                                  Step i                                   │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌─────────────────┐                                                     │
│   │  k running ME   │  ◄── Accumulator carried from step i-1              │
│   └────────┬────────┘                                                     │
│            │                                                              │
│            │      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│            │      │    Π_CCS     │  │   Π_Twist    │  │   Π_Shout    │    │
│            │      │  (CPU chunk) │  │ (R/W memory) │  │  (lookups)   │    │
│            │      └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│            │             │                 │                 │            │
│            │             └────────────┬────┴─────────────────┘            │
│            │                          │                                   │
│            │                          ▼                                   │
│            │             ┌────────────────────────┐                       │
│            │             │  Batched sum-check     │                       │
│            │             │  (shared r_time)       │                       │
│            │             └────────────┬───────────┘                       │
│            │                          │                                   │
│            │                          ▼                                   │
│            │             ┌────────────────────────┐                       │
│            │             │   Fresh ME claims      │                       │
│            │             │ (CCS+Twist+Shout+IDX)  │                       │
│            │             └────────────┬───────────┘                       │
│            │                          │                                   │
│            └──────────────────────────┤                                   │
│                                       ▼                                   │
│                    ┌──────────────────────────────────┐                   │
│                    │ Main lane: Π_RLC → Π_DEC         │                   │
│                    │ fold all ME@r_time → k children  │                   │
│                    └─────────────────┬────────────────┘                   │
│                                      │                                    │
│            ┌─────────────────────────┴─────────────────────────┐          │
│            │                                                   │          │
│            ▼                                                   ▼          │
│   ┌─────────────────┐                                ┌─────────────────┐  │
│   │  k ME children  │                                │   Value lane    │  │
│   │  (to step i+1)  │                                │ (Twist @ r_val) │  │
│   └────────┬────────┘                                └────────┬────────┘  │
│            │                                                  │           │
└────────────┼──────────────────────────────────────────────────┼───────────┘
             │                                                  │
             ▼                                                  ▼
    (next step i+1)                                   ┌─────────────────────┐
                                                      │  value-lane         │
                                                      │  obligations        │
                                                      │  must be enforced   │
                                                      └─────────────────────┘
```

### Unified Folding Interface

All arguments reduce to the same **ME(b, L)** relation:

```
Π_CCS   : MCS(b, L)  ⟿  ME(b, L)^t_ccs
Π_Twist : TWI(b, L)  ⟿  ME(b, L)^t_twi
Π_Shout : SHO(b, L)  ⟿  ME(b, L)^t_sho
```

At each step:
```
(k running ME + fresh CCS ME + Twist ME + Shout ME) → Π_RLC → ME^agg → Π_DEC → ME(b, L)^k
```

---

## Two-Lane Folding

Twist's val-eval subprotocol requires a separate evaluation point `r_val`, creating two parallel folding lanes:

| Lane | Evaluation Point | Contents |
|------|-----------------|----------|
| **Main** | `r_time` | CCS + Shout + Twist read/write checks |
| **Value** | `r_val` | Twist value-evaluation claims |

Both lanes produce ME obligations that must be verified by the final proof layer.

### Why Two Lanes?

- Most claims are enforced at a single shared evaluation point `r_time` (sampled once per step via Fiat–Shamir)
- Twist also needs a separate evaluation point `r_val` for its value-reconstruction subprotocol (fresh sum-check challenges)
- Because Neo's ME is a single-point evaluation relation, `ME@r_time` and `ME@r_val` cannot be mixed in the same `Π_RLC` call

**Result**: each step can emit:
- **Main obligations**: ME children at `r_time` (carried to the next step)
- **Value-lane obligations**: ME children at `r_val` (must be carried forward to the final checker)

---

## Developer Onboarding

### 1. Read the Protocol + Implementation Overview

| Doc | Purpose |
|-----|---------|
| [`docs/neo-ai-summary.md`](docs/neo-ai-summary.md) | Developer-grade Neo protocol overview |
| [`docs/system-architecture.md`](docs/system-architecture.md) | IVC architecture + emission policies |
| [`docs/neo-with-twist-and-shout/integration-summary.md`](docs/neo-with-twist-and-shout/integration-summary.md) | Twist/Shout integration strategy (why two lanes) |

### 2. Run Tests

```bash
cargo test --workspace --release

# See full shard folding with Twist/Shout in action:
cargo test -p neo-fold full_folding_integration --release -- --nocapture

# Shared CPU-bus linkage + adversarial checks:
cargo test -p neo-fold shared_cpu_bus_linkage --release -- --nocapture
```

### 3. Where to Start in the Code

**Shard folding loop** — [`crates/neo-fold/src/shard.rs`](crates/neo-fold/src/shard.rs)
- Look for `fold_shard_prove_impl(...)` and `fold_shard_verify(...)`
- This is where:
  - Per-step inputs are bound into the transcript
  - Π_CCS is executed
  - Twist/Shout proofs are produced/checked
  - Π_RLC → Π_DEC runs for the main lane, and (when needed) for the value lane

**Memory sidecar (Twist/Shout integration)** — [`crates/neo-fold/src/memory_sidecar/memory.rs`](crates/neo-fold/src/memory_sidecar/memory.rs)
- Bridge layer that:
  - Runs the memory/lookup sum-checks
  - Emits ME claims/witnesses at `r_time`
  - Runs Twist's value-eval sum-check and emits value-lane ME claims at `r_val`

**Trace → per-step witnesses** — [`crates/neo-memory/src/builder.rs`](crates/neo-memory/src/builder.rs)
- `build_shard_witness_shared_cpu_bus(...)` builds per-step bundles for **shared CPU-bus** mode:
  - CPU MCS witnesses (via the CPU arithmetization)
  - Metadata-only Twist/Shout instances (no separate commitments)

**Shared CPU-bus layout and constraints**
- [`crates/neo-memory/src/cpu/bus_layout.rs`](crates/neo-memory/src/cpu/bus_layout.rs) — canonical bus layout (single source of truth)
- [`crates/neo-memory/src/cpu/constraints.rs`](crates/neo-memory/src/cpu/constraints.rs) — CPU↔bus binding + padding-to-zero constraints
- [`crates/neo-fold/src/memory_sidecar/cpu_bus.rs`](crates/neo-fold/src/memory_sidecar/cpu_bus.rs) — guardrails + bus copyouts
- [`crates/neo-memory/src/twist_oracle.rs`](crates/neo-memory/src/twist_oracle.rs) — sum-check oracles

---

## Core Concepts (Paper → Code)

| Concept | Meaning | Code Entry Points |
|---------|---------|-------------------|
| **Shard** | Trace segment processed chunk-by-chunk | [`crates/neo-fold/src/shard.rs`](crates/neo-fold/src/shard.rs) |
| **Folding step** | Unit consumed per iteration of the loop | `StepWitnessBundle` in [`neo_memory::witness`](crates/neo-memory/src/witness.rs) |
| **CCS** | Customizable Constraint System | `neo_ccs::relations::CcsStructure` |
| **MCS** | Matrix Constraint System (CCS + commitment) | `neo_ccs::relations::{McsInstance, McsWitness}` |
| **ME** | Universal foldable claim (single-point eval) | `neo_ccs::relations::MeInstance` |
| **Π_CCS** | CCS/MCS → ME claims via sum-check | [`neo_reductions::engines::*`](crates/neo-reductions/src/engines/) |
| **Π_RLC / Π_DEC** | Aggregate then decompose (norm control) | [`crates/neo-fold/src/shard.rs`](crates/neo-fold/src/shard.rs) |
| **Twist** | R/W memory argument (sparse increments) | [`crates/neo-memory/src/twist.rs`](crates/neo-memory/src/twist.rs), [`twist_oracle.rs`](crates/neo-memory/src/twist_oracle.rs) |
| **Shout** | Read-only lookup argument | [`crates/neo-memory/src/shout.rs`](crates/neo-memory/src/shout.rs) |
| **IDX** | Index-to-virtual-one-hot adapter (proved via sum-check over bit-columns) | `IndexAdapterOracle` in [`twist_oracle.rs`](crates/neo-memory/src/twist_oracle.rs) |
| **Two-lane folding** | Needed for Twist's second eval point `r_val` | `val_fold` in [`shard.rs`](crates/neo-fold/src/shard.rs) |

### Key Structs

```rust
// One folding chunk worth of witness:
neo_memory::witness::StepWitnessBundle {
    mcs: (McsInstance, McsWitness),      // CPU chunk
    lut_instances: Vec<(LutInstance, LutWitness)>,  // Shout per chunk
    mem_instances: Vec<(MemInstance, MemWitness)>,  // Twist per chunk
}

// Final obligations after shard verification:
neo_fold::shard::ShardObligations {
    main: Vec<MeInstance>,  // ME claims at r_time
    val: Vec<MeInstance>,   // ME claims at r_val (Twist only)
}
```

---

## Glossary

| Term | Definition |
|------|------------|
| **CCS** | Customizable Constraint System — generalized arithmetization |
| **MCS** | Matrix Constraint System — CCS with commitment columns |
| **ME** | Matrix Evaluation — universal foldable single-point claim |
| **MLE** | Multilinear Extension — polynomial representation of vectors |
| **Π_RLC** | Random Linear Combination protocol — aggregates multiple ME claims |
| **Π_DEC** | Decomposition protocol — splits aggregated ME back into children (norm control) |
| **Obligation** | ME claim emitted by shard verification that must be enforced by the final layer |
| `r_time` | Shared evaluation point for main-lane claims (CCS + Shout + Twist read/write) |
| `r_val` | Separate evaluation point for Twist's value-eval subprotocol |
| **Main lane** | Folding lane for claims at `r_time` |
| **Value lane** | Folding lane for Twist value-eval claims at `r_val` |
| **Hash-MLE** | Merkle-tree based polynomial commitment used for final Spartan layer (no trusted setup) |

---

## Repository Structure

```
crates/
  neo-ajtai/           # Ajtai (lattice) commitments; module-SIS binding
  neo-ccs/             # CCS/MCS/ME relations, matrices, arithmetization
  neo-fold/            # Shard folding loop, proof types, transcript plumbing
  neo-reductions/      # Π_CCS / Π_RLC / Π_DEC engines (optimized + paper-exact)
  neo-memory/          # Twist/Shout traces, encoding, MLE utilities, oracles
  neo-vm-trace/        # VM tracing traits (CPU, Twist, Shout) + trace capture
  neo-spartan-bridge/  # ME → Spartan2-style R1CS bridge using hash-MLE (WIP)
  neo-math/            # Field/ring utilities, extension field, norms
  neo-params/          # Parameter bundles + Poseidon2 config
  neo-transcript/      # Poseidon2 transcript (Fiat–Shamir)

docs/
  neo-paper/                       # Paper text (reference)
  neo-with-twist-and-shout/        # Twist/Shout integration docs
  neo-ai-summary.md                # Implementation-facing overview
  system-architecture.md           # End-to-end architecture notes
```

---

## End-to-End: Trace → Witness → Fold

### Step 1: Build Per-Chunk Witnesses

Use the shared CPU-bus witness builder in `neo-memory`. The following is **pseudocode** illustrating the pattern; see the [actual test code](crates/neo-fold/tests/full_folding_integration.rs) for working examples:

```rust
// Pseudocode — see full_folding_integration.rs for working code
use neo_memory::builder::build_shard_witness_shared_cpu_bus;

let steps = build_shard_witness_shared_cpu_bus(
    vm, twist, shout,
    max_steps,
    1, // chunk_size == 1
    &mem_layouts,
    &lut_tables,
    &initial_mem,
    &cpu_arith, // must be configured to write the shared bus into the CPU witness
)?;
```

**Reference test**: [`crates/neo-fold/tests/full_folding_integration.rs`](crates/neo-fold/tests/full_folding_integration.rs) — `full_folding_integration_single_chunk`

### Step 2: Prove a Shard

```rust
// Pseudocode — see full_folding_integration.rs for working code
use neo_fold::shard::{fold_shard_prove, fold_shard_verify, ShardObligations};
use neo_transcript::Poseidon2Transcript;

let mut tr = Poseidon2Transcript::new(b"nightstream/shard");
let proof = fold_shard_prove(
    FoldingMode::Optimized,
    &mut tr,
    &params,
    &ccs,
    &steps,           // Vec<StepWitnessBundle>
    &acc_init,        // Initial ME accumulator
    &acc_wit_init,    // Initial witnesses
    &l,               // Commitment scheme
    mixers,
)?;
```

### Step 3: Verify and Handle Obligations

```rust
// Pseudocode — see full_folding_integration.rs for working code
let mut tr_v = Poseidon2Transcript::new(b"nightstream/shard");
let outputs = fold_shard_verify(
    FoldingMode::Optimized,
    &mut tr_v,
    &params,
    &ccs,
    &step_instances,
    &acc_init,
    &proof,
    mixers,
)?;

// Handle both obligation lanes:
let main_obligations: &[MeInstance] = &outputs.obligations.main;
let val_obligations: &[MeInstance] = &outputs.obligations.val;

// Pass to final SNARK layer or ObligationFinalizer
```

---

## Memory Arguments: Twist & Shout

### Twist (Read/Write Memory)

Twist models memory as a recurrence via sparse updates:
```
Val_{t+1} = Val_t + Inc_t
```

**Shared-bus mode:** Twist does not have its own witness/commitments. Instead it consumes bus fields
opened from the CPU commitment (the tail of the CPU witness `z`):
- `ra_bits`, `wa_bits`
- `has_read`, `has_write`
- `rv`, `wv`
- `inc_at_write_addr`

**What stays virtual:**
- Full memory vector `Val_t` (never committed, computed via sum-check)

**Code:**
- Bus layout: [`crates/neo-memory/src/cpu/bus_layout.rs`](crates/neo-memory/src/cpu/bus_layout.rs)
- CPU↔bus constraints: [`crates/neo-memory/src/cpu/constraints.rs`](crates/neo-memory/src/cpu/constraints.rs)
- Oracles: [`neo_memory::twist_oracle.rs`](crates/neo-memory/src/twist_oracle.rs)

### Shout (Read-Only Lookups)

Shout proves that when `has_lookup[t] = 1`, the committed `val[t]` matches `table[addr[t]]`.

**Shared-bus mode:** Shout consumes bus fields opened from the CPU commitment:
- `addr_bits`
- `has_lookup`
- `val`

**Code:**
- Oracles: [`neo_memory::shout.rs`](crates/neo-memory/src/shout.rs)
 - Bus layout: [`crates/neo-memory/src/cpu/bus_layout.rs`](crates/neo-memory/src/cpu/bus_layout.rs)

### Address Encoding & IDX Adapter

Addresses use compact **bit-decomposition** instead of one-hot vectors:
- Each address is `d` components in base `n_side`
- Each component commits `ell = ceil(log2(n_side))` bit-columns
- Address width: `d * ell` columns instead of `d * n_side` (~32× reduction)

The **IDX adapter** implements an index-to-virtual-one-hot bridge: it provides a **virtual one-hot oracle** backed by committed bit-columns. Twist/Shout protocols query conceptual one-hot MLE evaluations, and the adapter proves these are consistent with the compact index-bit representation via sum-check. This shifts work from commitments to foldable sum-check proofs.

**Code:** `IndexAdapterOracle` in [`twist_oracle.rs`](crates/neo-memory/src/twist_oracle.rs), bit-address validation in [`crates/neo-memory/src/addr.rs`](crates/neo-memory/src/addr.rs)

---

## Development Notes

### Folding Engines

| Mode | Description |
|------|-------------|
| `FoldingMode::Optimized` | Optimized implementation (default in tests/integration) |
| `FoldingMode::PaperExact` | Reference implementation (feature-gated) |
| `FoldingMode::OptimizedWithCrosscheck` | Debug comparison mode |

### Debugging Tips

- Start with `chunk_size = 1` to shrink state
- If RLC alignment errors occur, check:
  - `validate_me_batch_invariants` in `shard.rs`
  - The `r` points in emitted ME claims (must match per lane)

### Key Tests

```bash
# Full shard prove/verify with Twist/Shout:
cargo test -p neo-fold full_folding_integration --release -- --nocapture

# Twist/Shout witness building:
cargo test -p neo-fold twist_shout_trace_to_witness_smoke --release -- --nocapture

# Session API (IVC-style):
cargo test -p neo-fold test_session_multifold --release -- --nocapture

# Shard folding integration:
cargo test -p neo-fold shard_folding_integration --release -- --nocapture

# Twist/Shout soundness tests:
cargo test -p neo-fold twist_shout_soundness --release -- --nocapture
```

---

## Security & Correctness

### ✅ Implemented Safeguards

* **Parameter validation**: Validates the RLC soundness bound for the chosen parameters
* **Transcript binding**: Poseidon2 domain separation across all phases
* **ME claim alignment**: Validates `r`-point consistency before Π_RLC
* **Two-lane obligation tracking**: Value-lane ME claims tracked in `ShardObligations`

### 🔬 Security Posture

> **Research software warning**: This demonstrates the protocol and transcript-binding structure but has not undergone independent review. Do not deploy without a full audit and complete final verification layer.

**Specific caveats:**

* No audit / formal verification
* Potential side-channel issues (Rust big-int / norm computations, etc.)
* Parameter selection not hardened for production
* Transcript domain separation is implemented but still research-grade
* Final obligation verification layer is WIP

---

## Roadmap

### Near Term
- [ ] Complete Spartan2 final SNARK layer using hash-MLE (obligation finalization)
- [ ] Add criterion benchmarks
- [ ] Sparse weight optimizations in bridge

### Medium Term
- [ ] GPU acceleration exploration
- [ ] Security audit preparation

### Long Term
- [ ] Production deployment tools
- [ ] zkVM integration (RISC-V/WASM)

---

## References

* **Neo**: Wilson Nguyen & Srinath Setty, "[Neo: Lattice-based folding scheme for CCS over small fields](https://eprint.iacr.org/2025/294)" (ePrint 2025/294)
* **Twist/Shout integration**: [`docs/neo-with-twist-and-shout/`](docs/neo-with-twist-and-shout/)
* **Spartan**: Srinath Setty, "Spartan: Efficient and general-purpose zkSNARKs without trusted setup" (CRYPTO 2020) — Nightstream uses hash-MLE for the final Spartan layer for the time being (expected to be upgraded soon)
* **Nova/HyperNova**: Recursive arguments from folding schemes
* **Plonky3**: Goldilocks field, Poseidon2 used by Nightstream

---

## Acknowledgements

### Jolt zkVM

The CPU-to-bus constraint binding system in the `cpu/constraints` module is adapted from the **Jolt** zkVM project.

We are grateful to the Jolt team for their open-source contribution to the zkVM ecosystem.

---

## Contributing

* **Add tests** for behavioral changes
* **Run formatting**: `cargo fmt` and `cargo clippy`
* **Update documentation** for API changes

---

## Governance & Policies

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Maintainers](MAINTAINERS.md)

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
