# Golden conformance

The maintained workflow runs two fresh native folds, from state 1 through
state 3, then checks both folds with Lean. It compares complete proof bytes,
caller values, and physical witnesses. Lean verifies the native PiCCS messages;
this workflow does not independently generate those proof messages in Lean.

```sh
python3.12 -B scripts/golden_conformance_ci.py --directory /path/to/new-run
```

The directory must not exist. The driver builds the current release test
binary and runs each producer phase under its own cap. No old-package archive
is needed. To compare with saved native outputs for the same package, also
supply `--archives /path/to/archives`.

Lean uses the repository toolchain by default. An owner-approved compatible
compiler can be selected for the whole command with `elan run TOOLCHAIN`;
all Lean children still run through `formal/nightstream-fprime/scripts/validate.sh`.

## Terminal rejection checks

The native run checks terminal acceptance at state 3 and a changed fresh
relation. It also changes two child evaluations while preserving their
weighted PiDEC sum. Each balanced change gets a complete new witness and
commitment. Verification must reach the production `Eval_K` or `Eval_A`
comparison and reject there. An earlier rejection fails the check.

Preparation and verification have separate process caps. To repeat a pair on
an existing native output directory, use the current built test executable:

```sh
timeout --signal=KILL 300 python3.12 -B crates/nightstream/tests/run_recursive_phase.py --binary TEST_EXECUTABLE --directory CPU_RUN_DIRECTORY --phase opening-k-prepare
timeout --signal=KILL 300 python3.12 -B crates/nightstream/tests/run_recursive_phase.py --binary TEST_EXECUTABLE --directory CPU_RUN_DIRECTORY --phase opening-k
timeout --signal=KILL 300 python3.12 -B crates/nightstream/tests/run_recursive_phase.py --binary TEST_EXECUTABLE --directory CPU_RUN_DIRECTORY --phase opening-a-prepare
timeout --signal=KILL 300 python3.12 -B crates/nightstream/tests/run_recursive_phase.py --binary TEST_EXECUTABLE --directory CPU_RUN_DIRECTORY --phase opening-a
```

The maintained library test
`lifecycle::tests::staged::fold::lean::golden::native_checker` reads JSON requests
on stdin. Its `compare`, `encode`, `ccs`, and `child-handoff` operations check
complete results, proof encoding, PiCCS mutations, and recomposition from all
17 input sources. Current requests are registered in
`lean_graph/obligations.json`.

## Saved fixtures

Refresh saved fixtures only after the fresh native run and both Lean comparisons
pass. The first checked NIFS result and caller replace the matching formal
artifacts; its native result and proof replace `stage1_actual_nifs` fixtures.
The maintained `golden-v1.zip` contains 19 interface files:

- For each fold 1 and 2: `native/fold-N/{pi_ccs_input.json,children.json,actual_result.json,proof.native,caller-inputs.json}`.
- For each state 1 and 2: `native/step-N/{envelope.json,fresh-claim.json}`.
- For each fold 1 and 2: `expected/step-N-{nifs,caller}.json`.
- The final `native/step-3/envelope.json`.

Do not put large private witness files in this archive. Archive replay checks
the interfaces. The fresh workflow separately requires complete physical
witness equality and retains those witnesses in its run directory.

## Limits and records

Each native or Python test child has the repository's 300-second cap. Each
Lean child has its 1,500-second cap and uses `validate.sh`. The staged native
runner keeps the owner-approved 16 GiB RSS guard. One Lean or Rust build/test
runs at a time. A timeout is a failed check, even if earlier phases passed.

Keep command receipts, complete inputs, and outputs with the run. The final
`cpu-result.json` exists only after all native and fresh Lean checks pass.
Digests in receipts identify files; they do not replace value or row checks.
Metal compatibility and independent review are separate acceptance checks.
