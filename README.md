# Nightstream: Lattice-based Folding with Twist/Shout Memory

[![GitHub License](https://img.shields.io/github/license/nicarq/nightstream)](LICENSE)

Nightstream is a **post-quantum** proving system built around a lattice-based folding scheme for **CCS** plus sum-check-based memory arguments (Twist/Shout). The active proving path targets CCS over the **Goldilocks** field with a degree-2 extension for sum-check soundness, and exposes a **compact published proof boundary** for RV64IM and CHIP-8 closed out with a Spartan2 final decider.

- **Twist** for read/write memory (register and RAM timelines)
- **Shout** for read-only lookups (bytecode fetch, decode, ALU tables)

Nightstream implements the protocol from the Neo paper "Lattice-based folding scheme for CCS over small fields" (Nguyen & Setty, 2025/294), extended with Twist/Shout memory arguments and a Spartan2 outer layer.

> **Status**: Research prototype. The single active proving path (`neo-fold-next`) proves and verifies full RV64IM and CHIP-8 programs end-to-end and publishes a compact Nightstream statement/proof pair. Chain-facing deployment wiring and independent audit are still unfinished. Not production-ready.

---

## What Works Today

- Full RV64IM and CHIP-8 public-proof → accepted-proof → final-relation → Spartan2 decider → published Nightstream pipeline
- Twist/Shout integrated as register, RAM, and TwistLink events inside Stage 2 semantics (no separate sidecar crate)
- Side-claim / side-opening / side-terminal relations wrap the RV64IM value-lane content into publishable artifacts
- End-to-end integration tests (`crates/neo-fold-next/tests/nightstream.rs`, `tests/chip8_nightstream.rs`) proving and verifying both ISAs
- Optional Midnight outer-compression bridge (`crates/nstream-midnight-bridge`) for theorem-facing proof exports

### Published Nightstream Boundary

The compact published boundary lives in `crates/neo-fold-next/src/nightstream/` and is published as a `NightstreamStatement` + ISA-specific nightstream proof:

- RV64IM carried Nightstream artifact: ~524 bytes
- CHIP-8 carried Nightstream artifact: ~548 bytes

These numbers refer to the *carried* published boundary, not the larger internal Spartan2 proof exchanged below the seam. The Spartan2 decider proof is backend-accounted and verifier-relevant, but it is not carried inside the published Nightstream artifact itself.

Measured via the perf snapshots:

```bash
# RV64IM (NS_DEBUG_N controls instruction count)
NS_DEBUG_N=1000 cargo test -p neo-fold-next --release --test perf -- \
  --ignored --nocapture rv64im_mixed_opcode_perf_snapshot

# CHIP-8
cargo test -p neo-fold-next --release --test perf -- \
  --ignored --nocapture chip8_nightstream_perf_snapshot
```

---

## Quick Start

### Prerequisites
- **Rust** stable (`rust-toolchain.toml` at repo root)
- `git`
- C compiler (only if enabling allocators like mimalloc)

### Build & Smoke Tests

```bash
cargo build --release

# Full workspace tests
cargo test --workspace --release

# Canonical RV64IM round-trip (prove + verify via nightstream seam)
cargo test -p neo-fold-next --release --test nightstream -- --nocapture

# Canonical CHIP-8 round-trip
cargo test -p neo-fold-next --release --test chip8_nightstream -- --nocapture
```

### WASM Demo (Browser)

See `demos/wasm-demo/README.md` for the full walkthrough. Quick build+serve:

```bash
./demos/wasm-demo/build_wasm.sh
./demos/wasm-demo/serve.sh
```

### iOS Native (XCFramework)

Build a native iOS static library packaged as an XCFramework (for Swift/Xcode integration):

```bash
./scripts/build_ios_xcframework.sh
```

See `demos/ios-demo/README.md` and `demos/android-demo/README.md` for the native demo apps.

### Paper-exact Reference Mode

Most tests use `FoldingMode::Optimized`. The `FoldingMode::PaperExact` engine is an O(2^ℓ) brute-force reference for cross-checking only; it is not used in normal test runs.

---

## Architecture Overview

`neo-fold-next` is the single active proving path. It has three layers: a generic SuperNeo shard spine, two ISA frontends (RV64IM and CHIP-8), and a compact publication boundary.

```
┌──────────────────────────────────────────────────────────────────┐
│  nightstream/               Published proof boundary (compact)   │  ← chain-facing
│    mod.rs                   NightstreamStatement + proof binding │
│    rv64im.rs                Rv64imNightstreamProof               │
│      side_claim / side_opening / side_terminal / side_eval_claim │
│      opening_artifact / side_bridges                             │
│    chip8.rs                 Chip8NightstreamProof                │
├──────────────────────────────────────────────────────────────────┤
│  decider/spartan2/          Generic Spartan2 final decider       │
├──────────────────────────────────────────────────────────────────┤
│  rv64im/        chip8/      ISA frontends                        │
│    isa / execute / lower    trace capture and expansion          │
│    builder / tables         parity manifests, ISA tables         │
│    stage1 / stage2 / stage3 row binding / temporal / continuity  │
│    kernel/                  three-stage kernel prover+verifier   │
│    ccs / layout             root CCS and column layout           │
│    final_relation           replay stages as folding chunks      │
│    decider_relation         wrap folded statement for Spartan2   │
│    trace_expand/            RV64IM only: MUL/DIV lowering        │
├──────────────────────────────────────────────────────────────────┤
│  Generic spine (ISA-agnostic)                                    │
│    proof.rs          StepInput, ChunkInput, RunProof, Carry,     │
│                      FoldSchedule, PackagedProof                 │
│    run.rs            prove_chunks / verify_chunks drivers        │
│    prover.rs         ShardProver::prove_chunk                    │
│    verifier.rs       ShardVerifier::verify_chunk                 │
│    chunk_relation.rs Π_CCS → Π_RLC → Π_DEC per chunk             │
│    finalize.rs       PackagedProof digest footer                 │
│    opening.rs        opening-claim / time-opening surfaces       │
│    time_opening.rs   grouped opening reduction / unification     │
│    step_build.rs     frontend StepBuild + extension records      │
│    witness_layout.rs packed-witness helpers                      │
├──────────────────────────────────────────────────────────────────┤
│  vm/                 Static VM contracts (VmSpec trait)          │
└──────────────────────────────────────────────────────────────────┘
```

### Per-chunk Folding Flow

At each chunk of the session, `chunk_relation::compute_chunk_relation_with_perf` runs the folding triple:

```
incoming main Carry + fresh CCS step claims
            │
            ▼
    ┌──────────────────┐
    │      Π_CCS       │  sum-check reduction over the CCS structure
    └────────┬─────────┘
             │   k fresh ME claims
             ▼
    ┌──────────────────┐
    │      Π_RLC       │  aggregate carry + fresh ME into one high-norm ME
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │      Π_DEC       │  decompose into k-1 low-norm ME children
    └────────┬─────────┘
             │
             ▼
      next main Carry  →  carried to the next chunk
```

All Fiat-Shamir challenges are sampled from a Poseidon2 transcript bound to `neo.fold.next/session`.

### RV64IM Side Lane

The RV64IM frontend also emits stage-level eval, opening, and terminal claims that cannot be folded together with the main-lane `Carry` (they are not single-point ME claims at the same point). These are published alongside the main proof as **side artifacts** under `nightstream/rv64im/side_*.rs`:

| Side artifact              | Content                                                         |
|----------------------------|-----------------------------------------------------------------|
| `side_claim_relation`      | Stage-claim bundle consistency                                  |
| `side_eval_claim_relation` | Phase-0 opened objects + stage proof bindings + eval claims     |
| `side_opening_relation`    | Stage selected rows vs carried opening claims                   |
| `side_terminal_relation`   | Witness artifact for the terminal side decider                  |
| `side_terminal_decider`    | Spartan2-backed publication shells (binding, target, relation)  |
| `opening_artifact`         | Phase-0/1/2 opening convergence artifact                        |
| `side_bridges`             | Projects accepted kernel artifacts into side relation witnesses |

CHIP-8 does not need these: its side/opening/linkage digests in `NightstreamStatement` are filled with fixed "absent" tags (see `chip8_absent_*_artifact_digest()` in `nightstream/chip8.rs`).

---

## Developer Onboarding

### 1. Read the Protocol + Implementation Overview

| Doc                                                                                                   | Purpose                                                                 |
|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`docs/superneo-paper/`](docs/superneo-paper/)                                                        | Neo paper text (source of truth for the folding protocol)               |
| [`docs/twist-and-shout-paper/`](docs/twist-and-shout-paper/)                                          | Twist / Shout paper text (source of truth for memory arguments)         |
| [`docs/jolt-paper/`](docs/jolt-paper/)                                                                | Jolt paper text (source for RV64IM lowering and lookup tables)          |
| [`docs/system-architecture.md`](docs/system-architecture.md)                                          | IVC architecture + emission policies                                    |
| [`docs/glossary.md`](docs/glossary.md)                                                                | Protocol terminology                                                    |
| [`docs/assurance-strategy.md`](docs/assurance-strategy.md)                                            | Testing / soundness assurance plan                                      |
| [`docs/explanations/zkvm-main-lane-vs-twist-shout.md`](docs/explanations/zkvm-main-lane-vs-twist-shout.md) | Main-lane vs. Twist/Shout split                                    |
| [`crates/neo-fold-next/specs/riscv-kernel.md`](crates/neo-fold-next/specs/riscv-kernel.md)            | RV64IM kernel spec (staging, parity, Jolt-inspired lowering)            |
| [`crates/neo-fold-next/specs/chip8-kernel.md`](crates/neo-fold-next/specs/chip8-kernel.md)            | CHIP-8 kernel spec                                                      |
| [`formal/superneo-lean/README.md`](formal/superneo-lean/README.md)                                    | Lean proof-facing model and dependency graph                            |

### 2. Run Tests

```bash
cargo test --workspace --release

# End-to-end nightstream round-trip for RV64IM
cargo test -p neo-fold-next --release --test nightstream -- --nocapture

# End-to-end nightstream round-trip for CHIP-8
cargo test -p neo-fold-next --release --test chip8_nightstream -- --nocapture

# Generic spine Π_CCS → Π_RLC → Π_DEC prove/verify
cargo test -p neo-fold-next --release --test prover_pipeline -- --nocapture
cargo test -p neo-fold-next --release --test finalized_proof  -- --nocapture

# Per-stage RV64IM suites
cargo test -p neo-fold-next --release --test rv64im_stage1 -- --nocapture
cargo test -p neo-fold-next --release --test rv64im_stage2 -- --nocapture
cargo test -p neo-fold-next --release --test rv64im_stage3 -- --nocapture
```

### 3. Where to Start in the Code

**Generic SuperNeo spine** in [`crates/neo-fold-next/src/`](crates/neo-fold-next/src/):

- [`proof.rs`](crates/neo-fold-next/src/proof.rs) defines the session types (`StepInput`, `ChunkInput`, `RunProof`, `Carry`, `FoldSchedule`, `PackagedProof`).
- [`run.rs`](crates/neo-fold-next/src/run.rs) hosts the `prove_chunks*`, `verify_chunks*`, `prove_and_package`, and `verify_packaged` drivers.
- [`prover.rs`](crates/neo-fold-next/src/prover.rs) holds the `ShardProver::prove_chunk` script.
- [`verifier.rs`](crates/neo-fold-next/src/verifier.rs) holds `ShardVerifier::verify_chunk`.
- [`chunk_relation.rs`](crates/neo-fold-next/src/chunk_relation.rs) sequences Π_CCS → Π_RLC → Π_DEC explicitly and defines `CommitmentMixers`.
- [`finalize.rs`](crates/neo-fold-next/src/finalize.rs) packages `PackagedProof` and emits the digest footer.

**RV64IM frontend** in [`crates/neo-fold-next/src/rv64im/`](crates/neo-fold-next/src/rv64im/):

- `isa.rs`, `execute.rs`, `lower.rs`, `builder.rs`, `tables.rs` cover ISA semantics, tracing, and lowering to `Rv64ExpandedRow`.
- `trace_expand/` contains the RV64IM-only multi-cycle MUL/DIV expansion (`mul/`, `divrem/`).
- `stage1/`, `stage2/`, `stage3/` cover row binding, the register/RAM/Twist-link timeline, and continuity.
- `kernel/` holds the three-stage kernel prover/verifier (`kernel/stages/`, `kernel/main_lane/`, `kernel/openings/`, `kernel/proof/`, `kernel/parity/`).
- `final_relation.rs`, `decider.rs`, `decider_relation.rs` replay stages as folding chunks and wrap them for Spartan2.

**CHIP-8 frontend** in [`crates/neo-fold-next/src/chip8/`](crates/neo-fold-next/src/chip8/):

- `spec.rs`, `isa.rs`, `execute.rs`, `lower.rs`, `builder.rs`, `tables.rs`, `poly.rs`, `trace.rs` cover CHIP-8 ISA semantics and trace capture.
- `stage1/`, `stage2/`, `stage3/`, `kernel/` mirror the three-stage structure from RV64IM.
- `ccs.rs`, `layout.rs`, `chunk_relation.rs`, `final_relation.rs`, `decider.rs` cover CCS definition through the decider.

**Published Nightstream boundary** in [`crates/neo-fold-next/src/nightstream/`](crates/neo-fold-next/src/nightstream/):

- `mod.rs` defines `NightstreamStatement`, `NightstreamProofBindingInputs`, and the core digest helpers.
- `rv64im.rs` and `rv64im/side_*.rs` build the RV64IM `Rv64imNightstreamProof` with its side-lane artifacts.
- `chip8.rs` builds the CHIP-8 `Chip8NightstreamProof`.

---

## Core Concepts (Paper → Code)

| Concept            | Meaning                                                                 | Code entry points                                                                              |
|--------------------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| **CCS**            | Customizable Constraint System                                          | `neo_ccs::CcsStructure`                                                                        |
| **MCS / CcsClaim** | CCS + commitment                                                        | `neo_ccs::CcsClaim`                                                                            |
| **ME / CeClaim**   | Universal foldable claim (single-point matrix evaluation)               | `neo_ccs::CeClaim`                                                                             |
| **Π_CCS**          | CCS/MCS → ME claims via sum-check                                       | [`neo_reductions::api::PiCcsProof`], sequenced in `neo-fold-next/src/chunk_relation.rs`        |
| **Π_RLC / Π_DEC**  | Aggregate then decompose (norm control)                                 | `neo-fold-next/src/chunk_relation.rs`; artifacts `PiRlcArtifact`, `PiDecArtifact` in `proof.rs` |
| **Chunk**          | Unit of per-chunk folding (one or more CCS rows)                        | `neo_fold_next::proof::ChunkInput` + `FoldSchedule`                                            |
| **Carry**          | Running main-lane ME claims + witnesses across chunks                   | `neo_fold_next::proof::Carry`                                                                  |
| **Run / Session**  | A full prove session over a sequence of chunks                          | `neo_fold_next::run::{prove_chunks, verify_chunks, prove_and_package}`                         |
| **PackagedProof**  | Final packaged proof + public statement                                 | `neo_fold_next::proof::PackagedProof`; built by `neo_fold_next::finalize`                      |
| **Twist**          | R/W memory argument (register + RAM timelines via sparse increments)    | `neo_fold_next::rv64im::stage2::semantics`: `RegisterWriteEvent`, `RamEvent`, `TwistLinkEvent`  |
| **Shout**          | Read-only lookup argument (bytecode fetch / decode / ALU / tables)      | `neo_fold_next::rv64im::stage1` / `chip8::stage1` row-binding families                         |
| **Nightstream**    | Published compact statement + proof boundary                            | `neo_fold_next::nightstream::{NightstreamStatement, rv64im, chip8}`                            |
| **Spartan2 decider** | Final decider over the folded statement                               | `neo_fold_next::decider::spartan2`                                                             |

### Key Types

```rust
// Generic session spine:
neo_fold_next::proof::StepInput {
    label: String,
    mcs: CcsClaim<Commitment, F>,
    witness: CcsWitness<F>,
}

neo_fold_next::proof::RunProof {
    fold_schedule: FoldSchedule,
    chunks: Vec<ChunkProof>,
    final_main_claims: Vec<CeClaim<Commitment, F, K>>,
}

neo_fold_next::proof::PackagedProof {
    statement: PublicStatement,
    proof: FinalProof,
}

// Published Nightstream boundary:
neo_fold_next::nightstream::NightstreamStatement {
    public_io_digest: [u8; 32],
    verifier_context_digest: [u8; 32],
    fold_schedule: FoldSchedule,
    semantic_step_count: u64,
    chunk_summaries: Vec<FixedShapeChunkSummary>,
    linkage_root: [u8; 32],
    proof_binding_root: [u8; 32],
}
```

---

## End-to-End: RV64IM

The following is pseudocode matching the flow in [`crates/neo-fold-next/tests/nightstream.rs`](crates/neo-fold-next/tests/nightstream.rs) (see `external_fixture` and `verify_fixture`).

```rust
use neo_fold_next::rv64im::{
    prove_rv64im_public_proof,
    setup_rv64im_spartan2_decider_from_public_proof,
    prove_rv64im_spartan2_decider_from_public_proof,
    Rv64imProofInput,
};
use neo_fold_next::nightstream::rv64im::{
    build_rv64im_nightstream_from_public_proof,
    verify_rv64im_nightstream,
};

// 1. Produce the public RV64IM proof (stages 1/2/3 + kernel + accepted artifact)
let proof_input: Rv64imProofInput = /* program_words + max_steps */;
let public_proof = prove_rv64im_public_proof(&proof_input)?;

// 2. Set up and run the Spartan2 decider over the folded statement
let (decider_pk, decider_vk) =
    setup_rv64im_spartan2_decider_from_public_proof(&public_proof)?;
let decider_proof =
    prove_rv64im_spartan2_decider_from_public_proof(&decider_pk, &public_proof)?;

// 3. Build the published Nightstream statement + proof
let (statement, nightstream_proof) =
    build_rv64im_nightstream_from_public_proof(&public_proof)?;

// 4. Verify the full chain
verify_rv64im_nightstream(
    &statement,
    &nightstream_proof,
    public_proof.statement.root_params_id,
    &decider_vk,
    &decider_proof,
)?;
```

CHIP-8 follows the same shape via `neo_fold_next::chip8::proof::prove_recursive` and `neo_fold_next::nightstream::chip8::{build_chip8_nightstream_from_recursive_proof, verify_chip8_nightstream_from_recursive_proof}`.

For direct use of the generic spine (bring your own CCS and `StepInput`s), use `neo_fold_next::run::{prove_and_package, verify_packaged}`.

---

## Memory Arguments: Twist & Shout

In `neo-fold-next`, Twist and Shout live directly inside the ISA frontends as deterministic row-binding and temporal-event semantics.

### Twist (Register and RAM R/W)

Stage 2 of each ISA captures the register and RAM timelines as events against the expanded trace:

- `RegisterReadEvent`: rs1/rs2 reads tagged by role
- `RegisterWriteEvent`: rd writes
- `RamEvent`: RAM reads and writes at canonical addresses
- `TwistLinkEvent`: link rows connecting the memory timeline to the committed execution

Stage 2 reduces these into canonical-family digests and summaries in `stage2/proof.rs`; Stage 2 sum-check logic and `r_twist_cycle` point derivation are specified in `crates/neo-fold-next/specs/riscv-kernel.md`.

**Code:** [`crates/neo-fold-next/src/rv64im/stage2/`](crates/neo-fold-next/src/rv64im/stage2/) and [`crates/neo-fold-next/src/chip8/stage2/`](crates/neo-fold-next/src/chip8/stage2/)

### Shout (Read-only Lookups)

Stage 1 of each ISA binds each executed row against read-only lookup families: bytecode fetch, decode, ALU tables, and (RV64IM) branch/address families. Each family produces a row-binding proof and digest that is carried through the kernel transcript and opened at phase-0 points by the openings subsystem.

**Code:** [`crates/neo-fold-next/src/rv64im/stage1/`](crates/neo-fold-next/src/rv64im/stage1/), [`crates/neo-fold-next/src/chip8/stage1/`](crates/neo-fold-next/src/chip8/stage1/), [`crates/neo-fold-next/src/rv64im/tables.rs`](crates/neo-fold-next/src/rv64im/tables.rs)

### Opening Convergence

Stage / main-lane / side-lane claims converge through a three-phase opening pipeline before being published:

- **Phase 0**: opened object bundles, stage proof bindings, eval claim bundles
- **Phase 1**: bucketed reduction of eval claims
- **Phase 2**: collapse to final openings

**Code:** [`crates/neo-fold-next/src/rv64im/kernel/openings/`](crates/neo-fold-next/src/rv64im/kernel/openings/), `crates/neo-fold-next/src/nightstream/rv64im/opening_artifact.rs`, `crates/neo-fold-next/src/opening.rs`, `crates/neo-fold-next/src/time_opening.rs`

---

## Development Notes

### Folding Engines

| Mode                                  | Description                                          |
|---------------------------------------|------------------------------------------------------|
| `FoldingMode::Optimized`              | Default; used in all normal tests and integration   |
| `FoldingMode::PaperExact`             | O(2^ℓ) reference engine, cross-check only            |
| `FoldingMode::OptimizedWithCrosscheck`| Debug comparison mode                                |

Per project policy in [`CLAUDE.md`](CLAUDE.md), tests always use `FoldingMode::Optimized` unless the paper-exact engine is explicitly requested.

### Debugging and Profiling

Constraint / perf dumps live in [`crates/neo-fold-next/tests/perf.rs`](crates/neo-fold-next/tests/perf.rs). All perf snapshots are `--ignored` by default.

```bash
# Full constraint + timing snapshot for RV64IM (N = instructions + 1 halt)
NS_DEBUG_N=10000 cargo test -p neo-fold-next --release --test perf -- \
  --ignored --nocapture rv64im_mixed_opcode_perf_snapshot

# CHIP-8 perf snapshot
cargo test -p neo-fold-next --release --test perf -- \
  --ignored --nocapture chip8_nightstream_perf_snapshot
```

For CPU/memory profiling see [`scripts/profile_for_ai.sh`](scripts/profile_for_ai.sh), [`scripts/profile_xctrace.sh`](scripts/profile_xctrace.sh), and [`scripts/profile_memory_deep.sh`](scripts/profile_memory_deep.sh). Usage is documented in [`CLAUDE.md`](CLAUDE.md).

### Formal (Lean)

Four Lean subprojects hold proof-facing models:

| Subproject                                                                | Purpose                                                       |
|---------------------------------------------------------------------------|---------------------------------------------------------------|
| [`formal/superneo-lean/`](formal/superneo-lean/)                          | Main SuperNeo theorem-facing model and dependency graph       |
| [`formal/nightstream-lean/`](formal/nightstream-lean/)                    | Published Nightstream boundary model                          |
| [`formal/twist-shout-lean/`](formal/twist-shout-lean/)                    | Twist/Shout memory-argument model                             |
| [`formal/opening-convergence-lean/`](formal/opening-convergence-lean/)    | Opening convergence pipeline model                            |

See [`CLAUDE.md`](CLAUDE.md) for the spec/interface/implementation layout and closure standard.

---

## Security & Correctness

### Implemented Safeguards

- **Parameter validation** for the RLC soundness bound
- **Transcript binding** via Poseidon2 domain separation across every phase (protocol-binding paths are Poseidon2-only)
- **ME claim alignment** checks before Π_RLC
- **Side-lane artifact digests** bound into the `NightstreamProofBindingInputs` root
- **Red-team test suite** ([`crates/neo-fold-next/tests/rv64im_redteam.rs`](crates/neo-fold-next/tests/rv64im_redteam.rs)) for tamper resistance on the RV64IM path

### Security Posture

> **Research software warning**: This repository demonstrates the protocol and transcript-binding structure but has not undergone independent review. Do not deploy without a full audit.

Specific caveats:

- No independent audit or formal verification of the Rust implementation
- Potential side-channel issues (Rust big-int / norm computations, etc.)
- Parameter selection not hardened for production
- Transcript domain separation is implemented but still research-grade
- Chain-facing deployment wiring for the published Nightstream boundary is still in progress

---

## Workspace Layout

```
crates/
  neo-params/             Parameter bundles + Poseidon2 config
  neo-math/               Field/ring utilities, extension field, norms
  spartan2/               Vendored Spartan2 used by the final decider
  neo-transcript/         Poseidon2 transcript (Fiat-Shamir)
  neo-ajtai/              Ajtai (lattice) commitments; module-SIS binding
  neo-ccs/                CCS/MCS/ME relations, matrices, arithmetization
  neo-reductions/         Π_CCS / Π_RLC / Π_DEC engines (optimized + paper-exact)
  neo-fold-next/          Active proving path: spine + RV64IM + CHIP-8 + nightstream
  nstream-midnight-bridge/ Midnight outer-compression bridge for proof exports

docs/
  superneo-paper/         Neo paper text
  twist-and-shout-paper/  Twist/Shout paper text
  jolt-paper/             Jolt paper text
  system-architecture.md  IVC architecture + emission policies
  glossary.md             Protocol terminology
  assurance-strategy.md   Assurance / testing plan
  rust-code-quality.md    Rust code-quality guidelines
  explanations/           Targeted explainers
  plans/                  Design and implementation plans
  soundness-specs/        Soundness requirement specs

formal/
  superneo-lean/          Main Lean model + dependency graph
  nightstream-lean/       Published boundary Lean model
  twist-shout-lean/       Twist/Shout Lean model
  opening-convergence-lean/ Opening convergence Lean model
```

---

## Roadmap

### Near Term
- Finish the chain-facing verifier/deployment story for the published Nightstream boundary
- Decide which backend-accounted proof material must become explicitly carried for chain verification
- Add criterion benchmarks
- Sparse weight optimizations along the opening convergence path

### Medium Term
- GPU acceleration exploration
- Security audit preparation

### Long Term
- Production deployment tools
- Broader zkVM coverage

See [`TODO.md`](TODO.md) for in-flight work.

---

## References

- **Neo**: Wilson Nguyen & Srinath Setty, "[Neo: Lattice-based folding scheme for CCS over small fields](https://eprint.iacr.org/2025/294)" (ePrint 2025/294). Local text: [`docs/superneo-paper/`](docs/superneo-paper/).
- **Twist / Shout**: Local text in [`docs/twist-and-shout-paper/`](docs/twist-and-shout-paper/).
- **Jolt**: Local text in [`docs/jolt-paper/`](docs/jolt-paper/); source for RV64IM instruction lowering, virtual composition, and lookup-table structure.
- **Spartan2**: Srinath Setty, "Spartan: Efficient and general-purpose zkSNARKs without trusted setup" (CRYPTO 2020). Vendored in [`crates/spartan2`](crates/spartan2).
- **Plonky3**: Goldilocks field and Poseidon2 primitives used by Nightstream.

---

## Acknowledgements

### Jolt zkVM

The RV64IM frontend's virtual-composition lowering for MUL/MULH/MULHSU/MULW and DIV/DIVU/DIVW/REM/REMU/REMW families, the bitmask+apply pattern for shift instructions, the virtual assertion pattern for control/assertion instructions, and the dense slot manifest structure for Stage 1 row binding are all Jolt-inspired. The detailed mapping is documented in [`crates/neo-fold-next/specs/riscv-kernel.md`](crates/neo-fold-next/specs/riscv-kernel.md).

Thanks to the Jolt team for releasing their zkVM work as open source.

---

## Contributing

- **Add tests** for behavioural changes
- **Run formatting**: `cargo fmt --all` and `cargo clippy` before pushing
- **Update documentation** for API changes
- **DCO sign-off** is required on every commit (see [`CLAUDE.md`](CLAUDE.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md))

---

## Governance & Policies

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Maintainers](MAINTAINERS.md)

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
