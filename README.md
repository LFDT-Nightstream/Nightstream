# Nightstream: Lattice-based Folding for CCS

[![GitHub License](https://img.shields.io/github/license/nicarq/nightstream)](LICENSE)

Nightstream is a **post-quantum** proving system built around a lattice-based folding scheme for **CCS** (SuperNeo, building on Neo, ePrint 2025/294) with a HyperNova-style recursive IVC layer. The active proving path targets CCS over the **Goldilocks** field with a degree-2 extension for sum-check soundness, Ajtai (module-SIS) commitments, and a Poseidon2-only transcript.

> **Status**: Research software under active development. `neo-fold-clean` is the main proving crate: it owns the lifecycle API, the F′ recursive-step circuit, and the decider. The earlier `neo-fold-prototype` sandbox (RV32IM/CHIP-8 end-to-end pipelines) has been removed from the tree. Chain-facing deployment wiring and independent audit are still unfinished. Not production-ready.

---

## What Works Today

- SuperNeo folding pipeline Π_CCS → Π_RLC → Π_DEC (`neo-reductions`, optimized + paper-exact engines)
- IVC lifecycle in `neo-fold-clean`: `prove` / `extend` a chain of CCS step instances, `finish_uncompressed` or `compress`, then `verify`
- F′ recursive-step shell: a low-norm bit-image layout, mixed-gate CCS structure, encoder, and R1CS compiler for stateful step functions
- Decider statement synthesis and terminal-CE relation checks; compact backend wiring is pending
- Red-team suites (`crates/neo-fold-clean/tests/system/lifecycle_redteam.rs` and friends) for tamper resistance on the lifecycle path

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

# Canonical end-to-end chain (encode F' steps, fold, finalize, verify)
cargo test -p neo-fold-clean --release --test system_fibonacci_bits_e2e -- --nocapture
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

`neo-fold-clean` is the main proving crate. Its module map:

```
crates/neo-fold-clean/src/
  lifecycle/        Public chain API: preprocess, prove, extend, compress,
                    finish_uncompressed, verify, verify_uncompressed(_audit)
  paper/            Paper-faithful protocol layer:
    construction2/  HyperNova-style IVC state transition + finalization
    reductions/     Π_CCS → Π_RLC → Π_DEC sequencing
    f_prime/        F′ step relation, R1CS lowering, Poseidon traces
    terminal_ce/    Terminal CE relation circuit + digest contract
    decider.rs      Decider-side x_out / state replay checks
    digest.rs       Poseidon2 structure/params/relation digests
  frontends/        F′ source-image frontends:
    f_prime/        Bit-image layout, mixed-gate structure, encoder,
                    stateful R1CS step compiler, recursive plan
    r1cs_f_prime/   Bellpepper-style R1CS → F′ instance builder
  engine/           Optimized execution: Π wrappers, CCS-native Poseidon2
                    gadgets, R1CS circuit builder, decider synthesis
```

### Per-chunk Folding Flow

Each lifecycle step folds fresh CCS claims into the running accumulator via the SuperNeo triple:

```
incoming running accumulator + fresh CCS step claims
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
   next running accumulator  →  carried to the next step
```

All Fiat-Shamir challenges are sampled from a Poseidon2 transcript; protocol-binding paths are Poseidon2-only.

### Lifecycle API

```rust
use neo_fold_clean::lifecycle;

let prep = lifecycle::preprocess(params, structure, public_input_len)?;

let mut audit = lifecycle::prove(&prep, [vec![step_0]])?;   // base step
audit = lifecycle::extend(&prep, audit, vec![step_1])?;     // fold one more step
lifecycle::verify_uncompressed_audit(&prep, &audit)?;       // replay-check the chain

let proof = lifecycle::finish_uncompressed(&prep, audit)?;  // close the chain
lifecycle::verify_uncompressed(&prep, &proof)?;
```

`lifecycle::compress` currently builds and validates the terminal statement, then
fails closed because no compact backend is connected. `toy-spartan` and its WHIR PCS
remain standalone by design.

---

## Developer Onboarding

### 1. Read the Protocol + Implementation Docs

| Doc                                                          | Purpose                                                       |
|---------------------------------------------------------------|---------------------------------------------------------------|
| [`docs/hypernova-paper/`](docs/hypernova-paper/)              | HyperNova paper text (source for the IVC layer)               |
| [`docs/architecture/`](docs/architecture/)                    | Design notes: terminal-CE proof, accumulator openings, perf   |
| [`docs/audits/`](docs/audits/)                                | Internal soundness-audit reports                              |
| [`docs/plans/`](docs/plans/)                                  | Design and implementation plans                               |
| [`formal/nightstream-lean/README.md`](formal/nightstream-lean/README.md) | Active assurance-first Lean model and evidence boundaries |

### 2. Run Tests

```bash
cargo test --workspace --release

# End-to-end F' chain: encode, fold, finalize, verify
cargo test -p neo-fold-clean --release --test system_fibonacci_bits_e2e -- --nocapture

# Lifecycle red-team (tamper-rejection) suite
cargo test -p neo-fold-clean --release --test system_lifecycle_redteam

# Reduction engines
cargo test -p neo-reductions --release
```

### 3. Where to Start in the Code

- [`crates/neo-fold-clean/src/lifecycle/`](crates/neo-fold-clean/src/lifecycle/) — the public chain API (`preprocess`, `prove`, `extend`, `compress`, `verify*`).
- [`crates/neo-fold-clean/src/paper/construction2/`](crates/neo-fold-clean/src/paper/construction2/) — IVC state transition, x_out binding, finalization.
- [`crates/neo-fold-clean/src/frontends/f_prime/`](crates/neo-fold-clean/src/frontends/f_prime/) — the F′ source image, structure, encoder, and the stateful R1CS step compiler.
- [`crates/neo-reductions/src/api.rs`](crates/neo-reductions/src/api.rs) — `FoldingMode` and the Π_CCS / Π_RLC / Π_DEC engine entry points.

---

## Core Concepts (Paper → Code)

| Concept            | Meaning                                                    | Code entry points                                              |
|--------------------|-------------------------------------------------------------|----------------------------------------------------------------|
| **CCS**            | Customizable Constraint System                              | `neo_ccs::CcsStructure`                                        |
| **MCS / CcsClaim** | CCS + commitment                                            | `neo_ccs::CcsClaim`                                            |
| **ME / CeClaim**   | Universal foldable claim (single-point matrix evaluation)   | `neo_ccs::CeClaim`                                             |
| **Π_CCS**          | CCS/MCS → ME claims via sum-check                           | `neo_reductions::api`, wrapped by `neo-fold-clean/src/engine/` |
| **Π_RLC / Π_DEC**  | Aggregate then decompose (norm control)                     | `neo-fold-clean/src/paper/reductions/`                         |
| **F′**             | Augmented recursive-step relation (HyperNova §6)            | `neo-fold-clean/src/frontends/f_prime/`                        |
| **Construction 2** | IVC chain state + per-step fold proof                       | `neo-fold-clean/src/paper/construction2/`                      |
| **Decider**        | Terminal check of the folded accumulator                    | `neo-fold-clean/src/engine/decider.rs`, `paper/terminal_ce/`   |
| **Toy Spartan**    | Standalone Spartan/WHIR backend; not lifecycle-connected    | `crates/toy-spartan`                                           |

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

Perf snapshots live in [`crates/neo-fold-clean/tests/perf/`](crates/neo-fold-clean/tests/perf/) and are `--ignored` by default:

```bash
# Lifecycle fold/IVC perf snapshot
cargo test -p neo-fold-clean --release --test perf_fibonacci_bits -- --ignored --nocapture fibonacci_bits_perf_snapshot

# Full-history audit-circuit R1CS shape handed to the decider
cargo test -p neo-fold-clean --release --test perf_fibonacci_bits -- --ignored --nocapture fibonacci_decider_r1cs_shape_snapshot
```

For CPU/memory profiling see [`scripts/profile_for_ai.sh`](scripts/profile_for_ai.sh), [`scripts/profile_xctrace.sh`](scripts/profile_xctrace.sh), and [`scripts/profile_memory_deep.sh`](scripts/profile_memory_deep.sh). Usage is documented in [`CLAUDE.md`](CLAUDE.md).

### Formal (Lean)

[`formal/nightstream-lean/`](formal/nightstream-lean/) is the active formal
project. Its specification, assurance tiers, generated-artifact policy, and
validation commands are documented in that project's
[`README.md`](formal/nightstream-lean/README.md) and
[`AGENTS.md`](formal/nightstream-lean/AGENTS.md).

The sibling Lean packages under `formal/` are legacy reference material. The
active project imports no theorem from them; any useful lemma or
counterexample must be reviewed and re-established inside
`formal/nightstream-lean` before it contributes assurance. Those directories
are therefore not semantic authorities for Rust or R1CS changes.

---

## Security & Correctness

### Implemented Safeguards

- **Parameter validation** for the RLC soundness bound (`neo-params`, SuperNeo Appendix B.2 profile)
- **Transcript binding** via Poseidon2 domain separation across every phase (protocol-binding paths are Poseidon2-only)
- **Structure/params digests** recomputed from authoritative inputs, never trusted from the wire
- **Red-team test suites** (`crates/neo-fold-clean/tests/system/lifecycle_redteam.rs`, `r1cs_compiler_stateful.rs`) for tamper resistance

### Security Posture

> **Research software warning**: This repository demonstrates the protocol and transcript-binding structure but has not undergone independent review. Do not deploy without a full audit.

Specific caveats:

- No independent audit or formal verification of the Rust implementation (internal audit notes live in [`docs/audits/`](docs/audits/))
- Potential side-channel issues (Rust big-int / norm computations, etc.)
- Parameter selection not hardened for production
- Chain-facing deployment wiring is still in progress

---

## Workspace Layout

```
crates/
  neo-params/      Parameter bundles + Poseidon2 config
  neo-math/        Field/ring utilities, extension field, norms
  toy-spartan/     Standalone Spartan backend with WHIR PCS
  neo-transcript/  Poseidon2 transcript (Fiat-Shamir)
  neo-ajtai/       Ajtai (lattice) commitments; module-SIS binding
  neo-ccs/         CCS/MCS/ME relations, matrices, arithmetization
  neo-reductions/  Π_CCS / Π_RLC / Π_DEC engines (optimized + paper-exact)
  neo-fold-clean/  Main proving crate: lifecycle API, F′ recursive shell,
                   frontends, decider, terminal-CE relation

docs/
  hypernova-paper/ HyperNova paper text
  architecture/    Design notes (terminal CE, accumulator openings, perf)
  audits/          Internal soundness-audit reports
  plans/           Design and implementation plans

formal/
  superneo-lean/   Main Lean model (source of truth)
  direct-ccs-fprime-lean/ Direct-CCS F′ model
  twist-shout-lean/ Twist/Shout Lean model
```

---

## Roadmap

### Near Term
- Compact terminal-CE proof for the folded accumulator (see [`docs/architecture/compact-terminal-ce-proof-requirements.md`](docs/architecture/compact-terminal-ce-proof-requirements.md))
- Twist/Shout memory arguments as projection protocols feeding the main fold lane (see [`docs/plans/`](docs/plans/) and `TODO.md`)
- Criterion benchmarks

### Medium Term
- GPU acceleration exploration
- Security audit preparation

### Long Term
- Production deployment tools
- Broader zkVM coverage

See [`TODO.md`](TODO.md) for in-flight work.

---

## References

- **Neo**: Wilson Nguyen & Srinath Setty, "[Neo: Lattice-based folding scheme for CCS over small fields](https://eprint.iacr.org/2025/294)" (ePrint 2025/294).
- **HyperNova**: Abhiram Kothapalli & Srinath Setty, "HyperNova: Recursive arguments for customizable constraint systems". Local text: [`docs/hypernova-paper/`](docs/hypernova-paper/).
- **Toy Spartan**: fork of Srinath Setty's "Spartan: Efficient and general-purpose zkSNARKs without trusted setup" (CRYPTO 2020), with a standalone WHIR PCS. See [`crates/toy-spartan`](crates/toy-spartan).
- **Plonky3**: Goldilocks field and Poseidon2 primitives used by Nightstream.

---

## Acknowledgements

The earlier RV32IM prototype frontend drew on the Jolt zkVM's instruction lowering and lookup-table structure. Thanks to the Jolt team for releasing their work as open source.

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
