# Getting Started

## Prerequisites

- Rust stable (pinned by `rust-toolchain.toml` at the repo root)
- `git`; a C compiler only if enabling allocators like mimalloc

## Build and smoke-test

```bash
cargo build -p nightstream --release
timeout --signal=KILL 300 cargo test -p nightstream --release
```

Project test policies (enforced, see [CLAUDE.md](../CLAUDE.md)): every test invocation
gets a hard 5-minute timeout; tests always use `FoldingMode::Optimized` (never
`PaperExact` without explicit approval); tests live under `tests/`, never inline in
implementation files; run `cargo fmt --all` after modifying Rust code.

## Current lifecycle

Use [`nightstream`](../crates/nightstream/README.md) to compile an application,
save and load its circuit package, and select a prover and verifier. Both require
an explicit minimum statistical-security level from the caller.
The same page gives the current CPU and Metal benchmark commands.

## Legacy lifecycle reference

The retained `neo_fold_legacy::lifecycle` API uses a frontend.
The minimal frontend is direct-CCS: you supply an R1CS shape and satisfying
assignments, it hands back foldable instances.

```rust
use neo_fold_legacy::{
    frontends::direct_ccs, prove, extend, finish_uncompressed,
    verify_uncompressed, CcsInstance, FoldSchedule,
};

let prep = direct_ccs::preprocess_seeded(&r1cs, seed)?;

// One CCS instance per row of user computation. z = [x | w], length structure.m;
// z[..m_in] is public, z[m_in..] is private.
let rows: Vec<CcsInstance> = user_assignments.iter()
    .map(|z| direct_ccs::build_instance(&prep, &r1cs, z))
    .collect::<Result<Vec<_>, _>>()?;

// Batching policy: RowsPerStep(1) (default), RowsPerStep(n), or WholeRun.
let steps = FoldSchedule::RowsPerStep(4).partition(rows)?;

let mut audit = prove(&prep, steps)?;            // in-flight proof + audit trail
for step in FoldSchedule::RowsPerStep(4).partition(more_rows)? {
    audit = extend(&prep, audit, step)?;         // fold more batches later
}

let proof = finish_uncompressed(&prep, audit)?;  // flush trailing latest, drop audit trail
verify_uncompressed(&prep, &proof)?;             // terminal-only IVC verification
```

For diagnostics, the Spartan decider statement, or multi-chunk chains, keep the audit
trail: `finish_uncompressed_with_audit` + `verify_uncompressed_audit`. See
[Lifecycle API](architecture/lifecycle.md) for when each path applies.

## Where to start reading code

For current work, start with `crates/nightstream/src/circuit.rs` and
`crates/nightstream/src/folding/`. The following paths describe the legacy API:

1. `crates/neo-fold-legacy/src/lifecycle/` — the public chain API and its two
   verification paths. The module doc in `lifecycle/mod.rs` is the best single overview.
2. `crates/neo-fold-legacy/src/paper/mod.rs` — the paper-symbol → code glossary. Every
   identifier in the `paper/` layer is a paper symbol or maps to one.
3. `crates/neo-fold-legacy/src/paper/construction2/` — IVC state transition, x_out
   binding, finalization.
4. `crates/neo-fold-legacy/src/frontends/` — how user computation becomes foldable
   CCS instances, and the frontend soundness boundary.
5. `crates/neo-reductions/src/api.rs` — `FoldingMode` and the Π_CCS / Π_RLC / Π_DEC
   engine entry points.

## Demos

Browser (WASM), iOS, and Android demo apps live under `demos/` with their own READMEs
(`demos/wasm-demo/`, `demos/ios-demo/`, `demos/android-demo/`).
