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
timeout 300s cargo test -p neo-reductions --release
timeout 300s cargo test -p neo-fold-clean --release --test nifs_round_trip
timeout 300s cargo test -p neo-fold-clean --release --test f_prime_r1cs
timeout 300s cargo test -p neo-fold-clean --release --test nebula_f_prime
timeout 300s cargo test -p neo-fold-clean --release --test system_r1cs_ivc_terminal
timeout 300s cargo test -p wip-spartan --release
```

## neo-fold-clean test areas

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
