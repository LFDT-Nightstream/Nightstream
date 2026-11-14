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
timeout --signal=KILL 300 cargo test -p ns-compress --release
timeout --signal=KILL 300 cargo test -p nightstream-fprime --release --lib application_records
timeout --signal=KILL 300 cargo test -p nightstream-fprime --release --lib bounded_depth::application
```

The tests that compress a step of the hash chain hold the compression setup
of 10 GB and need up to 30 GB, so they are ignored. The test of the large
pinned key needs 6 GB and is ignored too. CI runs the last three commands
alone, after the other tests, with the release profile of `Cargo.toml` (fat
LTO), which is the build that users run (`scripts/ci_rust_tests.sh`). The
other CI tests turn LTO off. Run the first command by hand:

```sh
timeout --signal=KILL 300 cargo test -p nightstream --release --lib lifecycle::bounded_depth::tests::compression -- --ignored --test-threads 1
timeout --signal=KILL 300 cargo test -p nightstream --release --test compression -- --ignored
timeout --signal=KILL 300 cargo test -p nightstream --release --test compression_at_cap -- --ignored
timeout --signal=KILL 300 cargo test -p ns-compress --release --lib -- --ignored key::tests::the_large_pinned_key_is_the_key_of_the_public_seeds
```

`cargo test -p ns-compress --release` runs the argument on small claims,
with a key polynomial of 512 blocks.
`scripts/akita_schedules.sh` makes the Akita schedule files of
`crates/ns-compress/artifacts` again.

CI runs the selected Rust regression suites. Lean remains a local check.
Until a Mac runner is available, run the Metal relation check locally on a
supported Mac:

```sh
timeout --signal=KILL 300 cargo test -p neo-prover-metal --release --no-default-features --features metal --lib session::joint::relation::tests
```

## Formal and cross-checks

Use `formal/nightstream-fprime/scripts/validate.sh` for Lean. Each command
has a 25-minute cap. Read the active project's `AGENTS.md` before editing.
