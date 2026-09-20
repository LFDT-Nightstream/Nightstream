# Replacement crate validation

Work branch: `nico/nightstream-crate`, based on
`9787d8e77069246e3e2afc7dcfab755556fd5023`.
The unchanged `neo-fold-clean` source is the migration reference. Its only
dependency edge from this crate is for development comparisons.

The selected implementation goal is complete. The
[fresh replay receipt](tests/evidence/fresh-recursive-replay.json) records all
29 successful phases, both complete output comparisons, and the failed
successor attempt that led to the batch commitment change.

That result covers the CPU migration. The later Metal engine validation is
incomplete; see [the M1 Max results](#engine-validation-on-the-m1-max).

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
| Batch and existing fixed-key commitment checks | 7 passed; scalar parity, independent ring products and rejection checks | 0.05 s, excluding compilation |
| Fresh two-fold replay and terminal acceptance/rejection | 29 staged invocations passed; both complete output comparisons passed | 74.32 min summed successful phase time |
| Final ordinary `cargo test -p nightstream --release`, with Lean/Lake absent from `PATH` | 20 passed, 6 large checks ignored; separate runs reported here | 41.63 s unit tests; 57.11 s assembly tests |

The public Poseidon2 run took 38.06 s to prepare, 50.75 s to prove the first
step, and 77.07 s to verify it. A first step constructs the base case; it does
not execute an active C/R/D fold. Full recursive execution is recorded
separately and must not be inferred from that result.
These single-step comparisons preceded the batch commitment change.

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
The final production build at Rust source `096c94f0` passed in 7.36 s.
The full public Poseidon2 check passed in that environment in 166.30 s:
38.71 s preparation, 50.00 s proving, 77.39 s terminal verification.
Its test invocation reached 31,988,760,576 bytes of maximum resident memory;
the same-input old invocation reached 30,788,304,896 bytes. The production
dependency tree excludes `neo-fold-clean` and Lean runtime crates.
`cargo package --list -p nightstream --allow-dirty` also includes the required
blueprint, shared manifest and saved fixtures. All repository links resolve
to tracked files. Registry publication was not attempted.

## Engine validation on the M1 Max

On 2026-09-20, the engine checks ran on an Apple M1 Max with 10 CPUs and
64 GiB of memory, using Rust 1.94.1 and Apple Metal compiler 32023.921.
These are different hardware conditions from the migration measurements above.
The source was the uncommitted engine changes on `a76160b7d`, including the
Metal fixes described below. No Lean command was needed.

The production check uses the Rust Poseidon2 application, `b = 2`,
`k_rho = 16`, 6,377,559 constraint rows, 14 matrices, and the selected
28-round padded domain. The CPU and Metal PiCCS runs read the same newly
generated base envelope and witnesses. The CPU baseline preceded the fixes;
those fixes change only Metal code. Saved digests are not the acceptance test:
the PiCCS phase replays its proof through the CPU verifier and compares the
complete transcript state and cursor.

| Check | Result | Runtime | Peak resident memory |
| --- | --- | --- | --- |
| Generate the production base source | Passed | 175.23 s | 6,761,037,824 bytes |
| Production CPU PiCCS, including preparation | Passed | 202.89 s | 34,178,367,488 bytes |
| Production Metal PiCCS, before fixes | Stopped at the cap; no proof | 300 s cap | Not captured |
| Production Metal PiCCS, after the zero-carry fix | Stopped during application-table construction | 300 s cap | Not captured |
| Production Metal PiCCS, after both fixes | Reached the first SumCheck round, then stopped | 300 s cap | Not captured |
| Small CPU/Metal C/R/D parity, including zero carry | Three passed | 145.80 s together | Not measured |
| Empty-table and zero-carry buffer regression checks | Both passed in separate checks | Not benchmarked | Not measured |
| Public Metal `prove`, two `extend` calls, and terminal checks | Compiled; not run | — | — |

CPU preparation took 67.89 s, cache construction 38.54 s, and PiCCS proving
95.82 s. In the last Metal attempt, preparation took 67.61 s, cache
construction 38.98 s, matrix-plan construction 97.08 s, and application-table
construction 65.71 s. The first SumCheck round had not completed at the cap.
Matrix-plan construction had taken 32.55 s in the preceding attempt. The
small concurrent Metal tests also had long waits; an isolated zero-carry
check, before the empty-table fix, passed in 1.14 s. These observations do not establish stable
Metal performance or a speed improvement.

The checks found two invalid buffer reads. The empty-offset shader helper
read its dummy buffer as a normal row-offset array. The zero-carry path
allocated one value but reported the full table length to later kernels.
The fixes return offset zero for an empty table and initialize the zero
carry with its actual allocated length. The regression checks use poisoned
dummy offsets and check the advertised buffer bounds. Temporary diagnostic
printing was removed after recording the failed production attempts.

The [engine receipt](tests/evidence/metal-production-replay.json) records the
scope and failures. The run files remain at
`/tmp/nightstream-metal-production-20260920`. The outer timeout stopped the
phase driver with each failed test, so those failure records were recovered
from the command result and log; test-process peak memory was not recovered.
Three review rounds were used, as required by `AGENTS.md`.

Full production Metal parity, successive folds, terminal acceptance and
rejection, and complete lifecycle timings remain unverified. The new public
test is `poseidon_metal_recursive_lifecycle` in `tests/circuit_lifecycle.rs`.
It is ignored by default and must keep the five-minute cap unless the owner
approves a longer run for that specific invocation. CUDA remains unavailable.

The existing phase driver now accepts `--engine optimized` or `--engine metal`
for `ccs` and `child`. Use separate CPU and Metal output directories because
checkpoint files are written once. For example, after generating a base
source in `RUN_DIRECTORY`:

```sh
timeout --signal=KILL 300 cargo test -p nightstream --release --features metal --lib --test circuit_lifecycle --no-run
timeout --signal=KILL 300 python3 -B crates/nightstream/tests/run_recursive_phase.py --binary TEST_EXECUTABLE --directory RUN_DIRECTORY --phase ccs --step 1 --engine metal
timeout --signal=KILL 300 cargo test -p neo-prover-metal --release --no-default-features --features metal --lib session::joint::tests
```

## Independent Poseidon2 benchmark

The crate also provides `nightstream-poseidon2-bench`; see the
[benchmark commands](README.md#poseidon2-benchmark). It builds from normal
dependencies, without `neo-fold-clean`, and uses the public lifecycle with
fixed Poseidon2 inputs. It does not call the migration baseline test.

On the M1 Max above, the first `--engine optimized --steps 1` trial recorded
67.62 s for preparation and 105.36 s for base proving. Its state matched
native Poseidon2. Terminal verification was still running at the 300-second
cap, so the run failed that slice and emitted no `benchmark_finished` record.
The OS timer retained a peak resident size of 20,360,183,808 bytes through
termination. This is a partial base-step measurement, not a completed
verification or an active-fold timing. Logs are in
`/tmp/nightstream-poseidon2-benchmark-20260920/cpu-base.jsonl` and `cpu-base.time`.

The release build with Metal and the argument/error checks passed. The
benchmark has no default step count and does not run as part of `cargo test`.

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
timeout --signal=KILL 300 cargo test -p neo-ajtai --release --test nightstream_fprime_batch_commitment
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

## Fresh recursive replay

The run at `/tmp/nightstream-crate-fresh-55312916` passed all 29 phases. Source
`55312916` generated the base and first C/R/D results. Its first successor
attempt reached the 300-second cap and failed that invocation. Source
`096c94f0` adds batch key expansion with the same scalar commitment results.
The retry passed in 160.66 s, using the same newly generated fold outputs.
No old reference output was used as a producer input. The generated successor
then drove the second fold.

Both full output comparisons passed. The first covers 542,178,206 reference
bytes; the second covers 688,725,539. Each compares all sixteen complete child
matrices, the complete fresh witness and claim, and every semantic envelope
field. Both canonical proofs match their references exactly. The later proof
has 945,983 bytes; its outgoing transcript, cursor and applicable independent
Lean result fields also match. The comparison helper rejected a control case
with one changed witness coefficient.

| Second-fold check | Result | Phase time |
| --- | --- | --- |
| Source commitments | Passed | 156.15 s |
| PiCCS | Passed, all 28 rounds | 149.37 s |
| PiRLC | Passed | 52.63 s |
| Signed split and commitments | Passed | 121.91 s |
| Seven nonzero child openings | All passed; nine zero children also checked | 117.00–175.08 s each |
| Complete C/R/D proof | Passed | 118.55 s |
| Successor construction | Passed | 161.10 s |
| Terminal acceptance and wrong-state rejection | Passed | 240.25 s |
| Rehashed and recommitted false opening | Rejected for the required `Eval_K` error | 238.63 s |

The successful phase times sum to 74.32 minutes. They include repeated input
loading, circuit preparation and checks. They exclude builds, the failed
attempt, comparisons and gaps between commands. This is not an uninterrupted
production proving benchmark. The largest recorded phase used
43,514,134,528 bytes of maximum resident memory, during the second PiCCS check.
Every successful phase finished below the 300-second cap with a clean source
tree. The receipt retains both source revisions and the failed attempt.

The staged run uses the production C/R helpers, native split, commitment and
opening kernels, NIFS verifier, successor construction and terminal verifier.
One uninterrupted call to the public active `extend` method remains unexecuted.
The corresponding single-process test is compiled and ignored by default:
`lifecycle::tests::recursive::fresh_recursive_producer_matches_golden_and_folds_successor`.
Its separate longer invocation still needs owner approval under `AGENTS.md`.
The staged run completes the path required by this goal; it does not establish
a universal proof of file loading or orchestration.

To reproduce the staged run, build the ignored phase test, then pass the
reported executable to the phase driver. Each successful invocation records
its source revision, request, exit status, time and memory. The driver rejects
a filter that ran zero tests. The outer timeout also covers the Python driver;
if it stops the driver before a receipt is written, retain that failure log.

```sh
timeout --signal=KILL 300 cargo test -p nightstream --release --lib lifecycle::tests::staged::run_phase --no-run
timeout --signal=KILL 300 python3 crates/nightstream/tests/run_recursive_phase.py --binary TEST_EXECUTABLE --directory FRESH_RUN_DIRECTORY --phase base
```

For source iterations 1 and 2, run `sources`, `ccs`, `rlc`, and `split`, each
with `--step ITERATION`. Run `child --step ITERATION --child INDEX` for the true
entries of `fold-ITERATION/split.json`'s `nonzero` array. Then run `nifs` and
`successor` for that iteration. Finally run `terminal`, `mutation`, and `reject`
without a step argument. Use the same binary and run directory throughout.
The NIFS phase re-splits the actual parent and requires every active child's
opening, so changing saved activity flags cannot remove a check. Sources and
matrix caches are checked from their authoritative inputs; checkpoint digests
are not accepted as authority.

The first successor reference is in the tracked
`docs/reviews/nightstream-fprime-requirements/NATIVE_ENVELOPE_EVIDENCE.zip`.
Its child-witness manifest and `TERMINAL_REPLAY_INPUTS.json` identify the
external archive
`NATIVE_TERMINAL_CHILD_WITNESSES_EVIDENCE-933ef4b4e31310b59abf1ff07856efe308d39513a88c4af36966a68b83d3f10e.zip`.
Extract the first envelope, fresh claim and fresh witness into
`FIRST_REFERENCE`, and its sixteen digit files into `FIRST_REFERENCE/material`.

`NONZERO_EXECUTION_RELEASE.json` records the later archive
`STAGE1_NONZERO_EXECUTION_EVIDENCE-b8cb6550f209a0b43d7bd5c6913d045b81e757e1e724a63afa88e1c6c29a93c1.zip`.
It contains the complete native proof, independent Lean C/R/D result,
successor caller, fresh assignment and sixteen child matrices. Extract its
member paths unchanged into `LATER_REFERENCE`. The planned GitHub release was
unavailable. Both external archives were copied by read-only SSH from the
owner's recorded evidence directory. Their sizes and checksums match the
committed manifests. Those checks establish provenance; the comparisons use
the actual file contents.

```sh
timeout --signal=KILL 300 python3 crates/nightstream/tests/compare_recursive_outputs.py --directory FRESH_RUN_DIRECTORY --fold 1 --reference FIRST_REFERENCE
timeout --signal=KILL 300 python3 crates/nightstream/tests/compare_recursive_outputs.py --directory FRESH_RUN_DIRECTORY --fold 2 --reference LATER_REFERENCE
```

Successful comparisons write `comparison-fold-N.json` with exact byte counts
and scope. JSON formatting and equivalent Goldilocks representatives may
differ; canonical proof bytes must match exactly. Complete later assignment,
claim and output equality is not a direct comparison of every later caller
input word. The receipt states that limit. The first caller's private and
public arrays are compared directly inside the first successor phase.

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
