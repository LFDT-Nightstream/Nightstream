# Validation for the goal-split review

Source: `4fc02857c3aa4207f3290739cc62d794ba86f5f9`. The tracked working tree was clean at the start and remained unchanged. The review reports and retained logs are under the repository's ignored `docs/` directory.

## Package loader

From the repository root:

```sh
RUSTC_WRAPPER="" PATH="/opt/homebrew/bin:$PATH" timeout -s KILL 300 \
  cargo test --locked --offline -p nightstream-fprime --release \
  --test package_loader -- --nocapture
```

Exit 101. Build: 22.02 s. Tests: 0.01 s; 1 passed, 7 failed. The package-dependent tests fail because the checked-out circuit JSON is a Git LFS pointer. The single passing test rejects a malformed PiCCS evaluation shape without needing that artifact.

[Full log](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-goal-splits/logs/package-loader.log).

## Native fixed interface and isolated R1CS test

```sh
RUSTC_WRAPPER="" PATH="/opt/homebrew/bin:$PATH" timeout -s KILL 300 \
  cargo test --locked --offline -p neo-fold-clean --release \
  --test nifs_fixed --test nifs_round_trip --test nifs_r1cs_isolated -- --nocapture
```

Exit 101. Shared release build: 49.45 s. Cargo ran `nifs_fixed` first: 3 passed in 0.03 s. It then ran `nifs_r1cs_isolated`: its single test failed before folding because it supplies `RunningInstance::default()`, while the current digest-only verifier requires a running claim. Cargo did not run `nifs_round_trip` after that failure.

[Full log](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-goal-splits/logs/native.log).

## Native replay and mutations

```sh
RUSTC_WRAPPER="" PATH="/opt/homebrew/bin:$PATH" timeout -s KILL 300 \
  cargo test --locked --offline -p neo-fold-clean --release \
  --test nifs_round_trip -- --nocapture
```

Exit 0. Incremental build: 0.11 s. All 9 tests passed in 0.06 s. They cover native replay, the CPU adapter, PiCCS output/terminal mismatch, PiDEC child/count mismatch, and mutations to the carried parent/children. The positive toy fixture returns a zero assignment regardless of its seed. These passes do not establish production nonzero Lean/Rust conformance.

[Full log](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-goal-splits/logs/round-trip.log).

## Limits

The 300-second cap is the root project's mandatory non-Lean test limit. Commands ran sequentially. No timeout occurred. No test was repaired or newly added; the purpose was to assess the current split candidates.

The unchanged source's Lean library and registered axiom library were validated earlier in the conversation. The conditional review root and encoding example also passed then. Their records are in the [previous review validation](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-prove2/VALIDATION.md). No Lean build was repeated for this source review.

The full conformance archive is absent. The current package-dependent approved gates were not run. No production proof backend, hosted verifier, new approval record, or conformance status was created.

The source audits traced the proposal boundaries, exact theorem assumptions, caller paths, distinct registered checker implementations, and required artifacts. They are not a new independent proof of all cryptographic formulas or an exhaustive audit of every primitive in the workspace.
