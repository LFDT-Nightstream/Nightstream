# Nightstream Lattice System Architecture

This document describes the high-level architecture of Nightstream's lattice SNARK pipeline, built around a **two-phase design**:

1. **Phase 1 (Shard Folding)**: Fast per-step accumulation via lattice-based folding with integrated Twist/Shout memory arguments. Produces ME (Matrix Evaluation) obligations.

2. **Phase 2 (Closure Proof)**: WHIR + Spartan2 SNARK that proves the ME obligations are satisfied, with digest-binding to ensure the obligations match Phase 1.

This separation allows **fast folding** (no per-step SNARK overhead) while still producing a **succinct final proof**.

---

## System Flow (Conceptual)

```
+-----------------------------+
| High-Level Input            |
| (RISC-V / WASM / Circuit)   |
| + Public Inputs x           |
| + Private Witness w         |
+-----------------------------+
            |
            v  (External to Nightstream)
+-----------------------------+
| Trace → Per-Step Witness    |
| - Build StepWitnessBundle   |
| - CPU MCS (CCS + commitment)|
| - Twist instances (R/W mem) |
| - Shout instances (lookups) |
+-----------------------------+
            |
            v  (Shard folding loop)
+-----------------------------+
| Step i Processing           |
| ┌─────────────────────────┐ |
| │ k running ME claims     │ |
| │ (carried from step i-1) │ |
| └───────────┬─────────────┘ |
|             │               |
|   ┌─────────┼─────────┐     |
|   │    Batched time   │     |
|   │    sum-check at   │     |
|   │    shared r_time  │     |
|   └─────────┬─────────┘     |
|             │               |
|   ┌─────────┴─────────┐     |
|   │  Π_CCS  Π_Twist   │     |
|   │  Π_Shout  IDX     │     |
|   └─────────┬─────────┘     |
|             │               |
|   ┌─────────┴─────────┐     |
|   │ Fresh ME claims   │     |
|   │ at r_time         │     |
|   └─────────┬─────────┘     |
|             │               |
|   ┌─────────┴─────────┐     |
|   │ Π_RLC → Π_DEC     │     |
|   │ (main lane)       │     |
|   └─────────┬─────────┘     |
|             │               |
|   ┌─────────┴─────────┐     |
|   │ k ME children     │     |
|   │ (to step i+1)     │     |
|   └───────────────────┘     |
+-----------------------------+
            |
            v  (After all steps)
+-----------------------------+
| ShardObligations            |
| - main: ME claims @ r_time  |
| - val: ME claims @ r_val    |
|   (Twist value-eval lane)   |
+-----------------------------+
            |
            v  (Phase 2: Closure)
+-----------------------------+
| Closure Proof               |
| WHIR + Spartan2             |
| - Digest binding (R1CS)     |
| - Ajtai openings (batched)  |
| - ME consistency (sumcheck) |
| - Boundedness checks        |
+-----------------------------+
            |
            v
+-----------------------------+
| Output: ClosureProofV1      |
| - ClosureStatementV1        |
|   (context, pp, obligations |
|    digests)                 |
| - Opaque proof bytes        |
+-----------------------------+
```

---

## Core Architecture

### Shard-Level Folding

Nightstream implements **shard-level folding** where each step processes one CCS chunk together with its matching Twist/Shout instances, all sharing sum-check challenges.

**Key concepts:**
- **Shard**: A trace segment represented as a collection of folding chunks
- **Folding step/chunk**: The unit processed by one iteration of the folding loop
- **StepWitnessBundle**: One MCS (CPU chunk) plus matching Twist/Shout instances for that chunk

### Per-Step Processing Flow

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
                                                      │  (must be enforced) │
                                                      └─────────────────────┘
```

---

## Unified Folding Interface

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

### Why Two Lanes?

- Most claims are enforced at a single shared evaluation point `r_time` (sampled once per step via Fiat–Shamir)
- Twist also needs a separate evaluation point `r_val` for its value-reconstruction subprotocol (fresh sum-check challenges)
- Because Neo's ME is a single-point evaluation relation, `ME@r_time` and `ME@r_val` cannot be mixed in the same `Π_RLC` call

**Result**: Each step can emit:
- **Main obligations**: ME children at `r_time` (carried to the next step)
- **Value-lane obligations**: ME children at `r_val` (must be carried forward to the final checker)

---

## Core Stages

### Stage 1: Trace → Per-Step Witnesses

**Entry point**: `neo_memory::builder::build_shard_witness_shared_cpu_bus`

- Execute VM and generate execution trace
- Build per-step `StepWitnessBundle` containing:
  - CPU MCS (CCS witness + Ajtai commitment)
  - Twist instances (R/W memory, metadata-only in shared-bus mode)
  - Shout instances (lookup checks, metadata-only in shared-bus mode)

### Stage 2: Π_CCS (CCS Reduction with Sumcheck)

- Encode the CCS into a **Q polynomial** and prove its correct evaluation via **sum-check** over the hypercube
- Runs as part of batched time sum-check with shared `r_time`
- Outputs **ME claims** (multilinear evaluation claims)

### Stage 3: Memory Sidecar (Twist/Shout)

**Twist (R/W Memory)**:
- Models memory via recurrence: `Val_{t+1} = Val_t + Inc_t`
- Full memory vector `Val_t` is never committed; computed via sum-check
- Produces ME claims at `r_time` (read/write checks) and `r_val` (value-eval)

**Shout (Read-Only Lookups)**:
- Proves that when `has_lookup[t] = 1`, the committed `val[t]` matches `table[addr[t]]`
- Produces ME claims at `r_time`

**IDX Adapter**:
- Implements index-to-virtual-one-hot bridge
- Addresses use compact bit-decomposition instead of one-hot vectors
- ~32× reduction in committed address width

### Stage 4: Π_RLC (Aggregation)

- Combine all ME claims at the same evaluation point into **one** using random linear combination
- Main lane: combines claims at `r_time`
- Value lane: combines claims at `r_val` (when Twist is active)

### Stage 5: Π_DEC (Decomposition)

- Split the high-norm aggregated object into `k` low-norm parts
- Yields **k ME claims** that feed the next iteration (main lane) or become obligations (value lane)

---

## Phase 2: Closure Proof (Spartan2 + WHIR)

After shard folding produces `ShardObligations`, the **closure proof** (Phase 2) converts those ME obligations into a succinct SNARK. This is where the IVC-style verification happens.

### Two-Phase Architecture

| Phase | What it does | Output |
|-------|-------------|--------|
| **Phase 1** | Shard folding via Π_CCS/Π_RLC/Π_DEC | `ShardObligations` (ME claims) |
| **Phase 2** | Closure proof (WHIR + Spartan2) | Succinct proof |

### Closure Statement

The closure proof binds to a public statement:

```rust
pub struct ClosureStatementV1 {
    pub context_digest: [u8; 32],    // Binds to CCS/params
    pub pp_id_digest: [u8; 32],      // Public parameters ID
    pub obligations_digest: [u8; 32], // Hash of ME obligations
}
```

### What Phase 2 Proves

The WHIR-based closure backend proves:

1. **Ajtai commitment openings** (batched)
2. **Boundedness** of the witness matrices `Z`
3. **ME consistency** (and bus openings when `BusLayout` is provided)
4. **Digest binding**: private obligations → `obligations_digest` (via Poseidon2)

### Digest Binding Proof (IVC-like Component)

The **digest-binding proof** uses **Spartan2 with Bellpepper R1CS** to prove that the private obligations hash to the public `obligations_digest`. This is the IVC-style embedded verifier component:

```
Private Obligations → Poseidon2 Hash → obligations_digest (public)
```

This is proven via a Spartan2 R1CS circuit that:
- Takes private obligations as witness
- Computes the Poseidon2/Goldilocks digest
- Constrains the output to match `stmt.obligations_digest`

### Current Status (Phase 2)

| Backend | ID | Status |
|---------|-----|--------|
| WHIR full closure (dev) | `5` | ✅ Working, but obligations are public in payload |
| WHIR private closure | `6` | ⚠️ Exists, but not production-audit-ready |

**Remaining work for production-ready Phase 2** (see `docs/spartan-compression-phase2-obligations-private.md`):
- Prove that committed weights/claims are derived from the same private obligations
- Remove obligations from proof payload while maintaining soundness

---

## Shared CPU-Bus Architecture

In shared-bus mode, Twist and Shout do not have their own commitments. Instead, they consume bus fields opened from the CPU commitment:

**Twist bus fields** (from CPU witness tail):
- `ra_bits`, `wa_bits` (read/write address bits)
- `has_read`, `has_write` (operation flags)
- `rv`, `wv` (read/write values)
- `inc_at_write_addr` (increment at write address)

**Shout bus fields**:
- `addr_bits` (lookup address bits)
- `has_lookup` (lookup flag)
- `val` (lookup value)

**Key files**:
- Bus layout: `crates/neo-memory/src/cpu/bus_layout.rs`
- CPU↔bus constraints: `crates/neo-memory/src/cpu/constraints.rs`
- Bus guardrails: `crates/neo-fold/src/memory_sidecar/cpu_bus.rs`

---

## Shard Obligations

After shard verification, the verifier receives `ShardObligations`:

```rust
pub struct ShardObligations<C, FF, KK> {
    pub main: Vec<MeInstance<C, FF, KK>>,  // ME claims at r_time
    pub val: Vec<MeInstance<C, FF, KK>>,   // ME claims at r_val (Twist only)
}
```

**Both lanes must be enforced by the final proof layer.** It is not sufficient to only check sum-check transcripts and folding algebra.

---

## Security Invariants

- **Fiat–Shamir soundness**: All challenges derived via Poseidon2 transcript bound to public data
- **Transcript binding**: Domain separation across all phases
- **ME claim alignment**: Validates `r`-point consistency before Π_RLC
- **Two-lane obligation tracking**: Value-lane ME claims tracked separately and must be finalized
- **Post-quantum assumptions**: Lattice commitments (Ajtai) and hash-based transcripts

---

## Architecture Comparison: Phase 1 vs IVC Designs

Nightstream uses a **two-phase design** that differs from pure IVC/Nova-style designs:

### Phase 1 (Shard Folding) — No Embedded Verifier

1. **No per-step in-circuit verifier**: The folding loop runs natively without proving a verifier circuit per step
2. **Shard-level batching**: Multiple steps are folded together before any SNARK compression
3. **Two-lane obligations**: Twist's value-eval creates a separate obligation stream

### Phase 2 (Closure Proof) — IVC-Style Verification

1. **Digest-binding circuit**: Uses Spartan2 + Bellpepper R1CS to prove obligations → digest binding
2. **WHIR for ME closure**: Proves the ME evaluation claims via FRI-based polynomial commitments
3. **Single compression**: One closure proof covers all obligations from Phase 1

### Why This Design?

- **Phase 1 is fast**: No per-step SNARK overhead; just algebraic folding
- **Phase 2 amortizes**: One expensive proof covers the entire shard
- **Flexible batching**: Choose shard size based on latency/throughput tradeoffs

---

## Code Entry Points

### Phase 1 (Shard Folding)

| Component | Location |
|-----------|----------|
| Shard folding loop | `crates/neo-fold/src/shard.rs` |
| Memory sidecar | `crates/neo-fold/src/memory_sidecar/memory.rs` |
| Twist oracles | `crates/neo-memory/src/twist_oracle.rs` |
| Shout oracles | `crates/neo-memory/src/shout.rs` |
| CPU bus layout | `crates/neo-memory/src/cpu/bus_layout.rs` |
| Witness building | `crates/neo-memory/src/builder.rs` |
| Proof types | `crates/neo-fold/src/shard_proof_types.rs` |

### Phase 2 (Closure Proof)

| Component | Location |
|-----------|----------|
| Closure proof container | `crates/neo-closure-proof/src/lib.rs` |
| Digest-binding (Spartan2 R1CS) | `crates/neo-closure-proof/src/digest_binding.rs` |
| WHIR backend (dev) | `crates/neo-closure-proof/src/whir_p3_backend.rs` |
| WHIR private backend | `crates/neo-closure-proof/src/whir_p3_private_backend.rs` |
| Bridge digests | `crates/neo-fold/src/bridge_digests.rs` |

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
| **Twist** | R/W memory argument via sparse increments |
| **Shout** | Read-only lookup argument |
| **IDX** | Index-to-virtual-one-hot adapter |
| **hash-MLE** | Merkle-tree based polynomial commitment (no trusted setup) |
| **WHIR** | FRI-based polynomial commitment scheme (Plonky3) |
| **Closure proof** | Phase 2 SNARK that proves ME obligations are satisfied |
| **Digest binding** | Proof that private obligations hash to public `obligations_digest` |

---

## Current Status

### Phase 1 (Shard Folding)
- ✅ Shard prove/verify loop with shared transcript binding
- ✅ Twist/Shout integrated per chunk, including two-lane obligations
- ✅ End-to-end integration tests proving and verifying shards

### Phase 2 (Closure Proof)
- ✅ WHIR full closure backend (dev, backend id `5`) — working
- ✅ Digest-binding proof via Spartan2 R1CS
- ⚠️ WHIR private closure (backend id `6`) — exists but not production-audit-ready
- ⚠️ Missing: obligations→(weights, claimed_sum) binding for full privacy

### Overall
- ⚠️ No audit; research-grade performance/side-channel posture

---

## References

- **Neo**: Wilson Nguyen & Srinath Setty, "[Neo: Lattice-based folding scheme for CCS over small fields](https://eprint.iacr.org/2025/294)" (ePrint 2025/294)
- **Twist/Shout integration**: `docs/neo-with-twist-and-shout/`
- **Spartan**: Srinath Setty, "Spartan: Efficient and general-purpose zkSNARKs without trusted setup" (CRYPTO 2020)
