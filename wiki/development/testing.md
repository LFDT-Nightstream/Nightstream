# Testing

The root [AGENTS.md](../../AGENTS.md) defines the test rules.

- Run Rust tests with `--release`.
- Give every non-Lean test command a timeout of at most five minutes.
- Use `FoldingMode::Optimized` in normal tests.
- Use PaperExact only for an explicitly approved reference check.
- Put integration tests under `tests/`, not in implementation files.
- A regression test must fail while the defect exists.

## Core checks

```sh
timeout --signal=KILL 300 cargo test -p nightstream --release
timeout --signal=KILL 300 cargo test -p neo-ccs --release --test packed_witness
timeout --signal=KILL 300 cargo test -p neo-reductions --release --test matrix_rows
timeout --signal=KILL 300 cargo test -p nightstream-fprime --release --lib application_records
timeout --signal=KILL 300 cargo test -p nightstream-fprime --release --lib package::native_application
```

CI runs these commands as separate steps. The legacy suite is not part of CI.
Until a Mac runner is available, run the Metal relation check locally on a
supported Mac:

```sh
timeout --signal=KILL 300 cargo test -p neo-prover-metal --release --no-default-features --features metal --lib session::joint::relation::tests
```

## neo-fold-legacy test areas

| Directory | Scope |
|---|---|
| `direct_ccs/` | Direct R1CS conversion and rejection checks |
| `f_prime/` | F' image, lowering, selective rows, and recursive relation |
| `nebula/` | Memory relation, segments, lane commitments, and lifecycle |
| `nifs/` | NIFS round trips, fixed adapters, and crosschecks |
| `reductions/` | PiCCS, PiRLC, PiDEC, and transcript binding |
| `gadgets/` | R1CS primitives and Poseidon2 transcript gadgets |
| `system/` | Lifecycle, decider, formal-conformance, and red-team checks |
| `perf/` | Ignored performance snapshots |

## Formal checks

Use only the validation wrapper in
`formal/nightstream-lean/scripts/validate.sh`. Lean commands have a
25-minute cap. Read
[formal/nightstream-lean/AGENTS.md](../../formal/nightstream-lean/AGENTS.md)
before a Lean change.
