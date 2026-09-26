# Manual golden conformance

The maintained run checks CPU–Lean conformance for **1→2→3**: the first fold,
one recursive fold, their caller inputs, and terminal checks at state 3.
Lean execution stays local. CI does not install or run Lean.

## Fresh CPU run and Lean verification

From the repository root, with Python 3.11 or later:

```sh
python3 -B scripts/golden_conformance_ci.py --directory /path/to/new-run
```

The command builds the current `nightstream` library test executable once.
It runs both native folds, terminal acceptance and rejection, and fresh Lean
checks of both proofs. It compares every returned C/R/D field, the complete
canonical proof bytes, and all caller words. The native checker also runs the
55 PiDEC mutation cases. A missing output or failed required command prevents
a successful completion record.

For comparison with saved native outputs from the **same package**, add
`--archives /path/to/archives`. The five archives named by
`restore_golden_inputs.py` must be present. Their recorded identities establish
file custody; the comparisons use complete values and bytes.

Lean checks the native PiCCS messages and PiDEC child commitments and
evaluations. This run does not generate independent Lean proofs or compare
complete physical witnesses. The retired independent replay coordinator and
its legacy native executable are removed.

The terminal rejection tests change two child evaluations while preserving
their weighted PiDEC sum. A preparation stage rebuilds and saves the complete
fresh witness and commitment. A separate stage checks terminal rejection.
Each stage has the repository's 300-second cap.
`opening-k` must reach the production `Eval_K` comparison;
`opening-a` must reach `Eval_A`. Rejection before that check fails the test.
To run those tests on an existing native output directory:

```sh
timeout --signal=KILL 300 python3 -B crates/nightstream/tests/run_recursive_phase.py --binary TEST_EXECUTABLE --directory CPU_RUN_DIRECTORY --phase opening-k-prepare
timeout --signal=KILL 300 python3 -B crates/nightstream/tests/run_recursive_phase.py --binary TEST_EXECUTABLE --directory CPU_RUN_DIRECTORY --phase opening-k
timeout --signal=KILL 300 python3 -B crates/nightstream/tests/run_recursive_phase.py --binary TEST_EXECUTABLE --directory CPU_RUN_DIRECTORY --phase opening-a-prepare
timeout --signal=KILL 300 python3 -B crates/nightstream/tests/run_recursive_phase.py --binary TEST_EXECUTABLE --directory CPU_RUN_DIRECTORY --phase opening-a
```

## Package and assignment conformance

Golden verifier comparisons do not replace exact matrix comparison or raw
assignment evaluation. Keep those separate gates when accepting a new package:

- `crates/nightstream-fprime/tests/per_application_logical_matrix_conformance.rs`
  compares every expanded matrix row with the independent Lean-row reader.
- `crates/nightstream-fprime/tests/base_step_assignment.rs` and
  `crates/nightstream-fprime/tests/per_application_assignment.rs` check raw
  assignments and rejection cases against those rows.
- `crates/nightstream/tests/assembly_internal/encoding.rs` compares every
  application row, recipe, and identity byte with the Lean application plan.

The candidate conformance driver is
`crates/nightstream-fprime/src/bin/check_package_conformance.rs`. It takes
explicit candidate files. Run the required candidate checks before changing
verifier pins.

## Limits and records

Each native or Python test child keeps the repository's 300-second cap. Each
Lean child uses `validate.sh` and the 1,500-second cap. Build processes run one
at a time in a worktree. The root three-round stop rule applies.

Keep the complete input and output directories and command receipts. The
[earlier result record](../docs/reviews/nightstream-fprime-requirements/golden-conformance-wide/README.md)
states the source and artifacts used for its checks; it does not certify a
changed package.

For a manual change plan, run
`python3 -B scripts/golden_conformance_changes.py --base BASE --head HEAD`.
Its flags request native checks, Lean artifact regeneration, or Metal checks.
They do not certify a result or enable Lean execution in CI.
