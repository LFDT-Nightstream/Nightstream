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

CI runs the selected Rust regression suites. Lean remains a local check.
Until a Mac runner is available, run the Metal relation check locally on a
supported Mac:

```sh
timeout --signal=KILL 300 cargo test -p neo-prover-metal --release --no-default-features --features metal --lib session::joint::relation::tests
```

## Formal and complete-fold checks

Use `formal/nightstream-fprime/scripts/validate.sh` for Lean. Each command
has a 25-minute cap. Read the active project's `AGENTS.md` before editing.
[Golden conformance](../../scripts/GOLDEN_CONFORMANCE.md) runs two fresh native
folds, Lean comparisons, and exact terminal rejection checks.
