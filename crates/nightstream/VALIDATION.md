# Replacement crate validation

Work branch: `nico/nightstream-crate`, based on
`9787d8e77069246e3e2afc7dcfab755556fd5023`.
The unchanged `neo-fold-clean` source is the migration reference. Its only
dependency edge from this crate is for development comparisons.

The selected Nightstream Goldilocks profile remains `b = 2`, `k_rho = 16`,
`B = 65536`, one fresh claim and sixteen carried claims. The golden assembly
preserves the selected package, key and transcript identities.

## Checked results

Results below were measured on 2026-09-19 on an Apple M5 Max with 18 CPUs and
128 GiB of memory, using `rustc 1.94.1 (e408947bf 2026-03-25)`. Times exclude
compilation unless stated otherwise.

| Check | Result | Time |
| --- | --- | --- |
| Lean shared-formula comparisons and export theorem audit | Passed | 3 s |
| Lean formula export | Passed; 332,415 bytes | 1 s |
| Final Lean boundary checks and complete `tests.Axioms` build | Passed; only the allowed axioms | 78 s for the audit build |
| Compiled Lean shared-verifier export comparison | Passed for both existing application fixtures | 81.90 s |
| Lean shared-verifier export | Passed; 15,917,816 bytes | 36.53 s |
| Rust formula interpreter against existing matrix formulas | 13 tests passed | 0.38 s |
| Fixed-key prefix commitments, including suffix-zero correspondence | 4 tests passed | 0.02 s |
| Rust application, fixture and component checks | 10 tests passed | 0.04 s |
| Complete assembled package equality | Passed; all records, not only digests | 1.78 s |
| Golden package/key identities and boundary rows | Passed | 56.64 s |
| All 6,377,559 active rows across 14 matrices | Passed against the independent sealed reference | 28.16 s |
| Second Rust application assembly | Passed with its own binding | 37.54 s |
| Saved native proof and transcript through the copied verifier | Passed; changed child rejected | 37.92 s |
| Public Poseidon2 first step and terminal check | Passed; changed expected output rejected | 166.11 s |
| Second Rust application first step and terminal check | Passed | 165.96 s |
| Unchanged old lifecycle, same Poseidon2 inputs | Passed | 145.42 s |
| Complete base assignment and commitment against the reference, then terminal verification | Passed | 217.02 s |
| Ordinary `cargo test -p nightstream --release`, with Lean/Lake absent | 20 passed, 5 large checks ignored and reported separately | 41.89 s unit tests; 57.76 s assembly tests |

The public Poseidon2 run took 38.06 s to prepare, 50.75 s to prove the first
step, and 77.07 s to verify it. A first step constructs the base case; it does
not execute an active C/R/D fold. Full recursive execution is recorded
separately and must not be inferred from that result.

The old lifecycle took 19.00 s to prepare, 48.99 s to prove the first step,
and 77.39 s to verify it. Both use the shared native kernels and current shared
formula interpreter. The new generic preparation costs about 19 s more;
these measurements do not establish a speed improvement. The old source was
not changed for this comparison.

The second-application run used 37.04 s for preparation, 50.11 s for proving,
and 78.59 s for terminal verification. `/usr/bin/time -l` recorded
34,153,627,648 bytes of maximum resident memory for that Cargo test invocation,
including its child processes. This is not a same-application memory comparison.

The production library also built with Lean and Lake absent from `PATH`.
The full public Poseidon2 check passed in that environment in 166.30 s:
38.71 s preparation, 50.00 s proving, 77.39 s terminal verification.
Its test invocation reached 31,988,760,576 bytes of maximum resident memory;
the same-input old invocation reached 30,788,304,896 bytes. The production
dependency tree excludes `neo-fold-clean` and Lean runtime crates.
`cargo package --list -p nightstream --allow-dirty` also includes the required
blueprint, shared manifest and saved fixtures. All repository links resolve
to tracked files. Registry publication was not attempted.

## Reproduction

Run Nightstream build and test commands sequentially. The five-minute limit
for native tests and the twenty-five-minute limit for Lean commands come from
the repository's `AGENTS.md`. Other projects do not share this queue.
The complete profile and performance tests are ignored by default and run
individually below: the measured base and baseline tests alone would exceed
five minutes if combined in one Cargo invocation. An ignored result is not a pass.

The artifact READMEs give the separate maintainer export commands. Rust checks
use saved artifacts and do not run Lean. From the repository root:

```sh
timeout --signal=KILL 300 cargo test -p nightstream-fprime --release --lib matrix_program
timeout --signal=KILL 300 cargo test -p neo-ajtai --release --test nightstream_fprime_prefix_commitment --test nightstream_fprime_matrix_commitment
timeout --signal=KILL 300 cargo test -p nightstream --release --test component_forms --test application_builder --test application_poseidon2
timeout --signal=KILL 300 cargo test -p nightstream --release --lib assembly::encoding_tests
timeout --signal=KILL 300 cargo test -p nightstream --release --test assembly_circuits
timeout --signal=KILL 300 cargo test -p nightstream-fprime --release --test per_application_logical_matrix_conformance final_fourteen_matrices_equal_the_independent_sealed_interpretation -- --exact --ignored --nocapture
timeout --signal=KILL 300 cargo test -p nightstream --release --lib lifecycle::tests::saved_proof_and_transcript_match_lean -- --exact
timeout --signal=KILL 300 cargo test -p nightstream --release --test circuit_lifecycle poseidon_base_step_matches_lean_and_verifies -- --exact --ignored --nocapture
timeout --signal=KILL 300 cargo test -p nightstream --release --test circuit_lifecycle rust_addition_base_step_verifies -- --exact --ignored --nocapture
timeout --signal=KILL 300 cargo test -p nightstream --release --test lifecycle_baseline unchanged_old_poseidon_base_lifecycle -- --exact --ignored --nocapture
timeout --signal=KILL 300 cargo test -p nightstream --release --lib lifecycle::tests::base::base_extension_matches_full_lean_assignment_and_terminal -- --exact --ignored --nocapture
```

The complete recursive test is compiled but not yet executed:
`lifecycle::tests::recursive::fresh_recursive_producer_matches_golden_and_folds_successor`.
It constructs a new base, generates C/R/D, compares every first-fold proof byte,
transcript state and successor caller word, consumes that successor in a second
fold, and checks terminal acceptance and a rehashed false opening. Existing
full-producer evidence exceeds the five-minute cap. Its specific longer
invocation requires owner approval. The goal remains open until it passes.

The first fold has a saved golden reference. The later fold is new execution
evidence; this repository does not contain an independently recorded Lean
reference for all of that later fold's bytes. Passing its terminal checks must
not be described as later-fold byte equality with Lean.

## Scope

The shared constraints and their existing Lean proofs remain authoritative.
The Lean changes are limited to export code, export checks and an equality
theorem for the exported output ports. No protocol, circuit, layout or
application proof definition was changed.

Rust decoding, assembly, native arithmetic and terminal verification remain
implementation boundaries. Exact comparisons do not prove arbitrary Rust
execution or application intent. A Rust-defined application does not supply
the Lean `Program` and `Fits` hypotheses required by the earlier per-application
closure result.

This crate checks terminal openings. No production compression backend or
production SNARK run is claimed.
