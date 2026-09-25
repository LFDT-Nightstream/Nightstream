# Replacement crate validation

## Wide-sampler selection

The local production package now selects the proved wide sampler from
`a0f4b5b41`. It has 137,341,872 committed coordinates, 3,248,956 logical rows,
and 2,607,606,765 normalized matrix entries. The native sampler, setup prefix,
binding, full matrix and assignment comparisons, and shared assembly checks
pass. Fresh recursive fixtures and final consumer checks are in progress.
The [integration record](../../tools/recursive-constraint-minimizer/experiments/wide-sampler-integration.md)
contains the current proof and execution scope.

The records below describe their named historical package versions. Their
counts and saved proof bytes do not describe the wide-sampler selection.

## Historical quotient integration

The quotient layout from `8b7c07d8` and the complete Lean direct-producer proof
at signed checkpoint `6bcdbb7c` are now connected to the selected Rust core
and lifecycle. Rust integration is still uncommitted at this update.
The installed package has SHA-256
`6216d1f62250a58d073ecf0a908bd074d3834957ec5be3361620bbdeb5a97642`.
The hash identifies these bytes; canonical Poseidon2 binding owns authority.

The selected profile stays Goldilocks, Poseidon2, `b = 2`, `k_rho = 16`, and
`B = 65536`. The package has **184,359,564 committed coordinates**,
**4,703,127 logical rows**, and **3,001,571,645 matrix nonzeros**. The
coordinate reduction is 27.1339%; matrix nonzeros increase by 28.5017% from
the original baseline. No overall speed or memory improvement is established.

The independent complete assignment check, direct/full result and error
agreement, changed-package fallback, identity, binding, pilot, and sparse
commitment checks passed. Fresh actual base C → R → D execution also passed:
the parent and canonical split were recomputed, six active child openings
were computed, all 43 NIFS mutations were rejected, and the complete
945,983-byte native proof matched the independent Lean result. The saved
comparison rejected all 55 PiDEC mutation cases. New native outputs and
independent Lean base-NIFS and recursive caller fixtures are published; see
[the current fixture record](../neo-fold-clean/tests/nifs/fixtures/stage1_actual_nifs/README.md)
and [the integration plan](../../tools/recursive-constraint-minimizer/experiments/PLAN.md).
The two small application reference files and the state/message request are
unchanged. The final Nightstream checks also passed:

| Check | Result | Time |
| --- | --- | ---: |
| Complete assembly equality and schema-2 rejection | Passed | 5.11 s tests; 112 s compilation |
| Saved native proof and transcript | Passed | 42.61 s |
| Exact recursive caller fields and detached-input rejection | Passed | 64.22 s |

All correctness and integration checks for this quotient checkpoint are
complete. The candidate lifecycle benchmark remains pending. These check
times do not establish an overall performance improvement.

## Historical migration and engine evidence

All sections below record earlier source and package runs. Their identities,
dimensions, timings, and completed-goal statements apply to those historical
runs and do not establish validation of the current quotient package.

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
those fixes change only Metal code. The historical PiCCS phase replayed its
proof through the CPU verifier and compared the transcript state and cursor.
That checks protocol replay, but does not independently establish correct
device arithmetic: the device also supplies the output openings. Production
Metal acceptance requires complete proof byte equality against CPU on the
same source. The current Metal PiCCS phase requires that reference and compares
the source files, SumCheck messages, and all output claims.

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
scope and failures. Raw logs and phase records are retained in
[the run archive](tests/evidence/metal-production-20260920).
The larger witness files were held at `/tmp/nightstream-metal-production-20260920`
and saved in a private recovery archive before the restart.
The outer timeout stopped the
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
timeout --signal=KILL 300 python3 -B crates/nightstream/tests/run_recursive_phase.py --binary TEST_EXECUTABLE --directory METAL_RUN_DIRECTORY --phase ccs --step 1 --engine metal --cpu-reference CPU_RUN_DIRECTORY
timeout --signal=KILL 300 cargo test -p neo-prover-metal --release --no-default-features --features metal --lib session::joint::tests
```

## Memory architecture investigation

The current export has 6,377,559 logical rows, of which 7,700 belong to the
application. The remaining 6,369,859 rows are fixed recursive-verifier work.
The shared manifest fixes a base logical width of 252,695,531 coordinates;
the application adds 41 coordinates for each of its 7,700 field slots. The
result is 253,011,231 logical coordinates, padded to 253,011,276 carrier
coefficients. The 41-coordinate encoding is balanced ternary with values in
`{-1, 0, 1}`; it is separate from the `b = 2`, `k_rho = 16` decomposition.
These dimensions describe the current circuit, not a necessary minimum for
the protocol.

One witness needs 74,966,304 bytes in the existing two-mask-per-ring-block
format. The old cache stored repeated matrix coefficient patterns separately.
The Metal transpose expanded those patterns a second time, and openings kept
forms for all matrices plus a complete padded equality tensor live together.
These storage choices explain the large memory use without changing the
required mathematical computation.

With coefficient patterns shared, a complete CPU PiCCS production replay
passed on the same source files and matched every saved CPU proof byte,
including output openings. Peak RSS fell from 34,178,367,488 to 4,327,669,760
bytes. The complete phase took 227.67 s, versus 202.89 s before. Cache building
took 55.01 s, versus 38.54 s; proving took 100.06 s, versus 95.82 s. The new
phase took longer. The host still held resources from a timed-out Metal job,
so these separate runs do not establish a controlled speed comparison. The
memory result covers the first
fold with zero carried witnesses, not a full lifecycle or the general 16 GB
requirement. The [raw record and log](tests/evidence/metal-production-20260920/cpu-shared-patterns)
identify the source and comparison inputs.

A later CPU change stores early signed-unit folds as indices into their
possible field values until those indices no longer fit `u16`. Its comparison
with ordinary dense folding passes through the transition and implicit zero
tail. The production memory measurement above predates that change.

The Metal rewrite now keeps references to shared coefficient patterns,
processes one matrix opening at a time, and factors the padded equality
weights. Its first device check on 2026-09-20 was blocked. The combined parity invocation
reached the 300 s cap after five cross-check tests passed; the first Metal
test did not finish. An isolated small test also waited at its first device
command and was stopped. The earlier production process remained in the
kernel exit state, while the GPU driver reported 52,719,140,864 bytes in use
and 100% utilization. No new production GPU run was started. The
[device snapshot and test log](tests/evidence/metal-production-20260920/compact-validation)
record this block. These failed attempts establish neither Metal parity nor
a speed result.

Before a restart was possible, process cleanup was attempted without a
restart or logout: the test-owned Metal compiler services received TERM and
KILL. A separate Objective-C program then submitted a four-byte buffer fill
without any Nightstream shaders. It reached the 300 s cap in committed state,
with no scheduled or completed callback. These actions did not recover the
device. No kernel settings, display service, or GPU driver were changed.
The same archive contains the probe source, log, and recovery result.

The checks after the CPU prefix change were:

| Check | Result |
| --- | --- |
| Nightstream reference/CPU C/R/D cross-checks, including the two-row exported polynomial | Six passed, 132.79 s |
| Saved Lean Poseidon2 application checks | Four passed |
| Public engine selection | Two passed |
| Reduction unit and row-source tests | Six passed; two external-source checks remained ignored |
| Metal crate test targets without the legacy adapter | Release build passed; device execution blocked |
| Nightstream all targets with `metal,cuda` | Release check passed |
| Clippy inspection of Nightstream, Metal, and reductions | No warnings on changed lines; existing warnings remain |
| Legacy `pi_ccs_v1_1_engine_parity` suite | Seven passed, eight failed at the unchanged missing-running-claim guard |

The legacy parity failures occur before the new prefix or matrix execution.
The fixtures call the digest-only prover with no running claim, while the
unchanged transcript owner requires one. That fixture repair is outside this
memory change. Clippy used `--cap-lints warn` to inspect all targets; it is not
a clean `-D warnings` result. Formatting and `git diff --check` passed.

The general memory requirement remains open. Metal still expands nonzero
norm sources into extension-field tables, and the carried calculation has
overlapping large temporary arrays. Commitments and terminal verification
still use the CPU. Those paths and complete lifecycle measurements must be
addressed before the 16 GB and 5× requirements can be accepted.

## GPU recovery and compact Metal checks

The owner restarted the M1 Max after the failed process cleanup. On
2026-09-21 UTC, the new boot time and absence of all four stuck test processes
confirmed the restart. The driver reported 424,837,120 bytes in use and no
busy work queues. No GPU or memory sysctl settings were changed.

| Check | Result | Time |
| --- | --- | --- |
| Independent four-byte Metal buffer fill | Correct value; scheduled and completed without an error | 0.33 s |
| Compact openings, empty offsets, and zero-carried buffer bounds | Three passed | 0.46 s test time |
| Complete small-circuit engine comparisons | Nine passed; two CUDA tests ignored because the kernel is absent | 129.35 s test time; 172.64 s including build |

The opening check crosses the parallel and tiled list thresholds and compares
all returned values with CPU results. The engine suite compares proof bytes,
claims, witnesses, and transcript state. It includes the two-row exported
polynomial, nonzero carried data, zero carry, and parallel PaperExact/Optimized
cross-checks with rejection cases. The logged verifier errors and worker panic
come from those deliberate rejection checks; the suite passed.

All invocations used the project's 300-second timeout. The engine invocation
recorded a maximum RSS of 4,068,556,800 bytes, including the build. This is a
small-circuit validation measurement, not a production memory result. After
the tests, driver memory returned to 424,837,120 bytes, there were no busy
work queues, and no Nightstream test process remained. Logs and the result
record are in [the recovery checks](tests/evidence/metal-recovery-20260921).

At this recovery checkpoint, the compact Metal rewrite had device parity
evidence. Production GPU runs were still stopped because the dense norm
buffers exceeded the owner's memory target with all witnesses active. The
next section records their replacement. The recovery checks did not establish
the 16 GB lifecycle requirement or the 5× speed target. No Lean command ran.

## Compact norms and production PiCCS

On 2026-09-21 UTC, the M1 Max ran the first production PiCCS fold through
both engines using the same stored source files. Both complete proofs match
the saved CPU proof, including every output opening. No Lean command ran.

| Measurement | Optimized CPU | Metal |
| --- | --- | --- |
| Complete PiCCS phase | 217.03 s | 132.40 s |
| Circuit preparation | 67.49 s | 66.87 s |
| Matrix cache construction | 54.97 s | 53.27 s |
| Proving, including Metal plan construction | 94.12 s | 11.32 s |
| Peak process RSS | 11,892,965,376 bytes | 12,220,039,168 bytes |

The CPU preparation was sampled for 10 s at 1 ms intervals, as in
`scripts/profile_for_ai.sh`. Release symbols were stripped, so that sample
does not identify named hot functions. Treat these phase timings as diagnostic
measurements. They show about 8.3× for proving and 1.6× for the complete PiCCS
phase. They are not full lifecycle timings or a pass of the 5× requirement.

The Metal norm prefix now uses shared field-value tables: 9 values after the
first fold, 81 after the second, and 6,561 after the third for signed-unit
witnesses. Positions use byte or `u16` indices. Later rounds use dense values,
with output allocated only for the next prefix and the old prefix released
after completion. For the selected width with all 17 sources active, the
calculated peak for norm-prefix buffers falls from 51.61 GB to 6.45 GB. This
calculation excludes masks, matrices, other tables, source witnesses, and
allocator overhead; it is not a process-memory measurement.

The carried projection combines one ring block in threadgroup memory and
produces its matrix and identity projections directly. It no longer stores
full-width real and imaginary intermediate planes. The common table also
shrinks each round instead of retaining its initial pair of buffers. Host
witness-mask copies are released after upload.

Two norm-prefix tests cover every fold, odd lengths, sparse source indices,
zero and one challenges, the dense transition, and zero witnesses. A separate
test compares the carried projection with CPU results while all 16 production
carried sources are nonzero. Three complete small CPU/Metal C/R/D comparisons
pass after both changes. An allocation test checks that an oversized request
fails before allocation or submission. The existing Metal adapter still builds.

Metal's allocator rejects requests that would exceed 16,000,000,000 bytes of
tracked device buffers. CPU allocations and driver overhead still require
separate measurement. The production result above uses zero carried witnesses;
later folds, terminal verification, and the general 16 GB requirement remain
open. In this run, the complete-phase bottleneck was the inherited CPU
preparation and cache construction. Cache construction still expanded the
compact verifier program before grouping repeated coefficients.

The [raw logs and result record](tests/evidence/metal-norm-20260921) include
both production phases, the tests, and the buffer-size calculation. All test
invocations used the 300-second cap. Clippy found no new warnings in this
change; existing dependency warnings and an unchanged Metal boolean-expression
warning remain. Clippy used `--cap-lints warn`, not a clean `-D warnings` check.

## Compact matrix runs

On 2026-09-21 UTC, formula substitution and cache construction stopped
expanding each retained field into 41 scalar coefficients. A retained field
now stays one geometric run through substitution and enters the existing CPU
and Metal run evaluators directly. Scalar rows still expand on request for
PaperExact and reference checks. No circuit formula, field encoding, profile,
identity, or proof format changed.

The same stored first-fold input was run again on the M1 Max. The optimized
CPU proof matches the saved proof from the expanded cache. The Metal proof
matches the new CPU proof. Both comparisons include every output opening and
check that all source files match. No Lean command ran.

| Measurement | Optimized CPU | Metal |
| --- | --- | --- |
| Complete PiCCS phase | 175.12 s | 89.58 s |
| Circuit preparation | 66.72 s | 66.71 s |
| Matrix cache construction | 5.30 s | 5.31 s |
| Proving, including Metal plan construction | 102.10 s | 17.09 s |
| Peak process RSS | 12,426,706,944 bytes | 14,220,787,712 bytes |

Cache construction is about 10× faster than the preceding expanded-cache
run. Proving became slower in both engines. The complete phase still improved:
CPU fell from 217.03 s to 175.12 s, and Metal from 132.40 s to 89.58 s. These
are diagnostic runs of the first PiCCS fold, with zero carried witnesses.
The Metal/CPU ratio for the complete phase is about 2×. This does not establish
the 5× full lifecycle target.

Memory also needs further work. Metal's retained device allocation after the
proof rose from 2,076,639,232 to 3,316,727,808 bytes. Its process peak rose from
12.22 GB to 14.22 GB. The run representation and geometric opening plan need
less metadata before later production folds can establish the 16 GB limit.
The 16 GB limit for every supported circuit remains unverified.

Fourteen formula tests passed, including every Poseidon2 and Phi81 row against
the independent expanded interpreter. Four cache tests compare scalar matrix
evaluation and openings, check overlapping runs and empty rows, and reject
invalid bounds, ordering, and incomplete coverage. The Metal carried
projection test uses all 16 nonzero carried sources and compares compact runs
with a separate scalar CPU cache.

The [raw logs and result record](tests/evidence/compact-matrix-runs-20260921)
contain both production runs and the focused tests. Every test invocation used
the repository's 300-second cap.

## Parallel preparation and streamed identity

On 2026-09-21 UTC, a 10-second sample of preparation with symbols showed
6,386 of 7,192 preparation samples in Poseidon2 hashing. The existing Cargo
profiling profile was used with the duration and interval from
`scripts/profile_for_ai.sh`. The test prepared the independent Rust addition
application. These samples cover the first reference check, not the complete
preparation interval.

Preparation now runs reference authorization and candidate construction in
parallel. It returns a circuit only after the reference passes its pinned
package and key checks and the candidate passes its own checks. A test changes
reference data that assembly replaces. The candidate remains byte-for-byte
equal to the original circuit, but preparation still rejects the changed
reference identity.

The geometric Metal opening plan now dispatches each matrix's run plan only
for that matrix. Its group records reuse the existing block coordinates and
the next group's offset. Tests compare multiple matrices with scalar CPU
openings, check sparse block gaps, and check the dispatch and storage counts.

Before the subsequent streaming hash change, the same first production PiCCS
input produced these results:

| Measurement | Optimized CPU | Metal |
| --- | --- | --- |
| Complete PiCCS phase | 141.93 s | 57.83 s |
| Circuit preparation | 34.75 s | 34.75 s |
| Matrix cache construction | 5.33 s | 5.50 s |
| Proving, including Metal plan construction | 101.39 s | 17.10 s |
| Peak process RSS | 13,772,636,160 bytes | 15,079,833,600 bytes |

Both complete proofs match the saved expanded-cache CPU proof. The Metal
dispatch count fell from 425 to 282. Retained device storage fell from
3,316,727,808 to 3,247,538,176 bytes. Proving time did not materially improve;
the phase speed gain came from parallel preparation. These first-fold runs
have zero carried witnesses and do not establish the 5× full lifecycle target.

Circuit identity hashing now consumes the same framed words incrementally.
It retains only the Poseidon2 state and rate position, instead of building a
full field-element input vector for each hash. The existing slice hash remains
the reference. Tests cover all update splits across empty, full, and partial
blocks, the exact nested numeric-array framing, and invalid input nodes.
No digest format, padding, circuit identity, or Lean artifact changed.

After streaming was added, the public optimized Poseidon2 base lifecycle
passed in 249.61 s with 13,163,626,496 bytes of peak process RSS. Preparation
took 34.66 s, proving 100.90 s, and terminal verification 113.43 s. The test
checked the saved Lean output and rejected a changed output. It used both the
repository's 300-second cap and the owner's 16,000,000,000-byte process-memory
cap. This is a base lifecycle check; it executes no active recursive fold.
The later terminal measurements below include streamed identity inputs.

The [logs and result record](tests/evidence/parallel-prepare-20260921) include
the profile excerpt, authorization test, opening tests, both PiCCS runs,
streaming hash checks, and public base lifecycle result. No Lean command ran.
The full recursive lifecycle, 5× speed target, and the
16 GB limit for every supported circuit remain open.

## Device terminal rows and zero openings

On 2026-09-21, terminal verification gained a Metal check of the actual matrix
rows and constraint polynomial. It does not use the prover's satisfied-row
substitution. Nonzero running openings also use the device. Statement checks,
norm checks, commitments, and transcript control remain shared host work.

Zero openings retain their shape checks, then return zero without allocating
matrix scratch. The allocation regression test failed before the change and
passed after it. CPU evaluation selects parallel work from the nonzero witness
count. Metal opening metadata is created only when an opening needs it.

| Public base lifecycle | Optimized CPU | Metal |
| --- | --- | --- |
| Total process time | 251.22 s | 245.96 s |
| Preparation | 34.93 s | 34.94 s |
| Proving | 100.70 s | 100.64 s |
| Terminal verification | 114.72 s | 110.20 s |
| Peak process RSS | 9,796,550,656 bytes | 11,410,685,952 bytes |

Both public tests matched the stored Lean application result, accepted the
base proof, and rejected a changed expected output. A separate test passed a
CPU-produced proof to Metal verification. It then recomputed the commitment of
a false witness; CPU and Metal both rejected its first unsatisfied row. This
check took 259.73 s and used 11,870,699,520 bytes of peak process RSS. Eight
focused Metal tests passed, including independent scalar row evaluation,
partial ring blocks, multiple threadgroups, and zero-opening allocation checks.

These runs used the repository's 300-second test cap. The public and adversarial
tests also used the owner's 16,000,000,000-byte process-memory cap. No Lean
command ran. [Logs and result records](tests/evidence/terminal-rows-20260921)
retain the measurements and failing allocation test. The base lifecycle has no
active C/R/D fold. Full recursive device parity, 5× lifecycle speed, and the
memory bound for every supported circuit remain unproved.

## Device fixed-key commitments

The CPU base completion timers measured 1.11 s in witness execution, 2.65 s in
packing, and 97.42 s in the fresh commitment. The witness has 2,459,565 nonzero
ring blocks. The selected key requires 2,921,963,220 coefficients for those
blocks across its 22 rows.

Metal now generates the exact `nightstream-ajtai-chacha20-wide256-v1`
coefficients in threadgroup tiles. Each tile is shared by the active witnesses,
then discarded. Signed convolution and Phi81 reduction stay on the device.
The existing CPU validator checks every input coordinate first. Zero witnesses
require no device work. This preserves the selected key, seed, coefficient
addresses, field reduction, ring, and commitment format.

The full fresh commitment matched the saved CPU commitment in 3.08 s. All
source files from the timed CPU base run equal the saved source byte for byte.
The Metal call allocated 98,999,236 bytes of explicit device buffers, uploaded
49,192,660 bytes, and downloaded 9,504 bytes. Its 22 command buffers contained
396 dispatches. This is about 31.65× faster for the commitment operation; it is
not a full lifecycle speed claim.

Three device tests compare mixed signs, dense and packed representations,
virtual zeros, partial tiles, odd reduction levels, and the first and last
production key addresses with CPU results. Invalid shapes and values fail
before a device allocation or dispatch. The shared key validator is used by
both backends. [Logs and the source comparison](tests/evidence/terminal-rows-20260921)
record the timings and complete commitment equality.

With device commitments enabled in proving and verification, the public Metal
base lifecycle passed in 52.11 s, including process start and cleanup. It used
11,458,199,552 bytes of peak process RSS. Preparation took 35.13 s, proving
6.83 s, and terminal verification 9.57 s. The saved Lean output matched and a
changed expected output was rejected. The same CPU path measured above took
251.22 s, so this base lifecycle result is 4.82× faster. It does not meet the
5× target and contains no active recursive fold.

Nine engine tests also passed with device commitments, including complete
CPU/Metal C/R/D proof equality, nonzero carried data, the selected two-row
polynomial, and the runtime PaperExact/Optimized cross-check. The two CUDA
checks remain ignored because the canonical CUDA kernel is absent. All test
invocations stayed within the repository's 300-second cap; the public base
test also used the owner's 16 GB process-memory cap. The CPU-proof terminal test also passed with device commitments enabled. It
accepted the original proof, then rejected the recommitted false witness at
the same row as CPU. It used 11,921,293,312 bytes of peak process RSS. General
bounded memory and complete production recursive device parity remain open.

The current first production PiCCS phase also matched every saved CPU proof
byte and all source files. It took 55.29 s: 34.80 s preparation, 4.99 s cache
construction, and 15.04 s proving. Peak process RSS was 15,179,464,704 bytes.
This phase has zero carried witnesses. It does not establish the memory cost
of a later fold with nonzero carried witnesses.

The default build passes. The Metal and CUDA build and legacy Metal adapter
type checks also pass. Strict Clippy stops at inherited errors in unchanged
dependency files. An audit with `--no-deps --cap-lints warn` covered the changed
crates and found one new test-only clone warning, which was corrected. The
final Metal audit retains only inherited warnings; this is not a strict
Clippy pass. Formatting and `git diff --check` pass. The production dependency
tree with `metal,cuda` contains no `neo-fold-clean`.

## Application and parent storage

The row kernel now reads signed witness masks directly. It no longer expands
the current carrier into a 2,024,090,208-byte array of field elements. Application
tables retain only the current prefix and allocate the next half when a fold
needs it. The initial reservation of 1,071,430,080 bytes for future prefixes is
gone. Sumcheck partials are local to each round. Completed command buffers are
released before the next norm prefix is allocated.

Two allocation tests failed before these changes and pass after them. One
checks that initial application storage grows only with current rows. The
other checks that row scratch does not grow with unused carrier columns.
Ten Metal joint tests pass, including independent CPU row and opening results.

On the same first-fold source, the complete PiCCS proof and source files still
match the saved CPU results. The phase took 55.79 s: preparation 34.62 s, cache
construction 4.99 s, and proving 15.28 s. Peak process RSS was 15,440,166,912
bytes. Thus these local allocation reductions did **not** reduce the measured
first-fold process peak. The opening evaluator still reserves 4,048,180,416
bytes for a full-carrier form buffer. At that point, later folds also allocated
a separate full-carrier projection buffer for the carried table.

The public base lifecycle passed in 52.46 s with 10,810,359,808 bytes of peak
RSS, down from 11,458,199,552 bytes. Preparation took 35.14 s, proving 6.82 s,
and terminal verification 9.66 s. It matched the saved Lean output and rejected
a changed expected output. Relative to the unchanged 251.22 s CPU path, this
run is 4.79× faster. The 5× requirement remains unmet.

CPU PiDEC now takes ownership of its parent witness and drops it after the
signed split. Metal also drops the parent at that point. Commitments, openings,
and public recomposition use the owned digits and parent claim. Nine engine
tests pass after this ownership change; two CUDA checks remain ignored because
the canonical kernel is absent. No production run with nonzero carried
witnesses is claimed here.

[Logs and the storage model](tests/evidence/compact-application-storage-20260921)
record the failing allocation tests, checks, proof comparison, and measurements.
Every test invocation used the repository's 300-second cap. The production
PiCCS and public base runs also used the owner's 16,000,000,000-byte process
memory cap. No Lean command ran. General bounded storage and complete recursive
device validation remain open.

## Carried projection, zero masks, and packed PiRLC

Carried-table construction now uses its output for both projection passes.
Only the actual row sums need separate storage: 102,040,944 bytes for this
profile, instead of the old 4,048,180,416-byte projection buffer. The device
checks match CPU formulas with all sixteen carried sources active, distinct
matrix weights, retained-field runs, and more rows than carrier coefficients.
The allocation regression failed on the old storage design and now passes.

Device masks omit trailing zero witnesses. Source counts and gamma exponents
keep their logical indices. Tests cover a zero prefix before an active source,
a zero suffix, and an all-zero family. Thirteen joint tests pass. The shared
seeded legacy adapter also passed after the carried-buffer, zero-mask, and
PiRLC changes.

The public recursive run first stopped at 16,600,924,160 bytes. After zero-mask
trimming, it stopped at 16,391,487,488 bytes. Both used the original 16 GB
process guard. The owner accepted the latter observed peak temporarily.

Packed PiRLC now reads positive and negative column masks directly. It no
longer creates a column/value pair for every nonzero coefficient. The fresh
production witness has 81,343,582 such coefficients, so those lists held at
least 1,301,497,312 bytes before capacity overhead. Older row-packed inputs
use two column-mask arrays instead. Dense inputs are unchanged.

The allocation test failed before this change and passes after it: a
column-packed input allocates only its output. Direct matrix multiplication
agrees for mixed signs, non-diagonal challenges, multiple sources, both packed
formats, and serial and parallel execution. Nine engine parity tests pass;
the two CUDA tests remain ignored.

Both implementations then ran the production PiRLC phase separately on the
same saved sources and CPU PiCCS proof. Peak RSS fell from 9,195,765,760 to
7,865,794,560 bytes, a reduction of 1,329,971,200 bytes. Total phase time fell
from 45.54 s to 42.23 s. These times include preparation and JSON input/output.
All 253,011,276 parent witness coefficients match as field elements, including
the completion tail. Goldilocks permits different integer representatives in
its JSON, so the witness files differ in bytes. The complete parent claim,
PiCCS proof, identities, and transcript record match byte for byte.

The next public recursive attempt used the accepted 16,391,487,488-byte guard.
It stopped after 62.82 s with 16,577,855,488 bytes observed. PiCCS reported
2.529 s for oracle construction, 5.707 s for rounds, and 6.611 s for outputs.
PiRLC and terminal verification did not finish. The mask change therefore
does not establish a lower full-process peak. A small buffer-lifetime probe
did not reproduce a completed-command leak; no ownership change was made
from that hypothesis.

[Logs and records](tests/evidence/carried-buffer-reuse-20260921) retain the
allocation failures, checks, and stopped production runs. Each test kept the
300-second cap. These three storage changes preserve the circuit, profile,
key and transcript. Formatting and the release all-targets Clippy audit with
warnings capped passed; existing warnings remain, so this is not a strict
`-D warnings` result. Complete recursive parity, the 5× lifecycle target, and
bounded storage for all supported circuits remain open.

## Complete first production Metal fold

The crate's complete Metal producer passed on the saved first-fold inputs.
Every one of its 945,983 canonical proof bytes matches the saved CPU proof.
This covers PiCCS, PiRLC, and PiDEC, including all child commitments and
openings. The incoming carried witnesses are zero; six outgoing digit
witnesses are nonzero. Public verifier replay also checks the returned claims and parent
authority. The phase saves all sixteen child witnesses for successor checks.

The test took 78.87 s, or 79.96 s including the external runner. Preparation
took 34.81 s and the complete C/R/D call took 42.99 s. Peak RSS was
16,733,356,032 bytes. Physical footprint peaked at 15,801,577,736 bytes.
This is a saved-input fold check with checkpoint input/output, not a full
lifecycle benchmark or a 5× result.

The generated child witnesses then passed the successor check on CPU. All
sixteen recomputed commitments matched. The complete caller's private/public
arrays and output digest matched the saved Lean fixture. The new step-2
envelope was saved for the next fold. This check passed in 273.84 s with
6,931,087,360 bytes peak RSS. Child commitments took 110.57 s and the fresh
commitment took 122.60 s. This is an independent CPU integration check, not a
Metal lifecycle timing.

The owner confirmed RSS as the acceptance measure, accepted approximately
16 GB for this pass, and deferred further memory tuning. This slice used a
16 GiB RSS guard (17,179,869,184 bytes). The uninterrupted recursive test
reached 17,252,679,680 bytes and stopped after witness splitting. No complete
recursive lifecycle is claimed. A memory map showed large empty allocator
regions, but a trial call to release unused allocator pages did not reduce
RSS. That experiment and its diagnostic code were removed.

[The logs and records](tests/evidence/allocator-pages-20260921) contain both
memory measures, the stopped runs, and the complete proof comparison. Each
test kept the 300-second cap. No Lean command ran.

The phase driver can repeat the complete producer and proof comparison:

```sh
timeout --signal=KILL 300 python3 -B crates/nightstream/tests/run_recursive_phase.py --binary TEST_EXECUTABLE --directory RUN_DIRECTORY --phase prove --step 1 --engine metal --reference-proof CPU_PROOF_FILE
```

The input directory must contain the complete `step-1` source. The reference
proof is used only for the final comparison and never as a producer input.
The later fold and terminal checks are recorded below. Uninterrupted full
lifecycle timing remains open.

## Nonzero-carried production fold

The second fold consumed the successor generated from the first Metal fold.
Its six nonzero carried witnesses, fresh witness, and claims were identical
between the new CPU and Metal runs. No Lean command ran. The production
arithmetic was unchanged in this validation slice; test phases gained shared
CPU opening-cache use and Metal successor selection.

The first CPU PiCCS attempt stopped at the normal RSS guard with
18,136,563,712 bytes observed. The owner then approved one CPU reference run
above the approximate 16 GB limit, retaining the 300-second timeout. That run
passed with 21,273,313,280 bytes peak RSS. It provides correctness evidence and
does not pass the production memory requirement. The exception was used only
for that run; all later phases used the 16 GiB RSS guard.

| Phase | Result | Runner time | Peak RSS, bytes |
| --- | --- | --- | --- |
| CPU PiCCS reference, approved memory exception | Passed | 187.87 s | 21,273,313,280 |
| Metal PiCCS | All CPU proof bytes and source files match | 71.72 s | 14,697,201,664 |
| CPU PiRLC | Passed | 46.45 s | 8,314,503,168 |
| CPU split and all child commitments | Passed | 189.99 s | 8,244,707,328 |
| CPU openings, one shared cache | All sixteen checked | 183.78 s | 12,068,405,248 |
| CPU complete proof assembly and replay | Passed | 210.11 s | 8,336,834,560 |
| Complete Metal C/R/D producer | All 945,983 CPU proof bytes match | 103.02 s | 16,655,368,192 |
| Metal successor construction | Passed | 54.54 s | 7,101,726,720 |
| Metal terminal acceptance and wrong-state rejection | Passed | 74.71 s | 14,822,801,408 |
| CPU construction of the false opening, rehash, and fresh commitment | Passed | 171.62 s | 5,979,226,112 |
| Metal rejection of the rehashed and recommitted false opening | Required `Eval_K` error | 72.79 s | 15,901,687,808 |

The batch CPU opening phase recomputes the signed split and compares every
saved child matrix. The final NIFS phase independently repeats the split,
recomputes all commitments, checks every saved child, and replays the complete
proof. Saved commitments are phase data, not authority.

The complete Metal fold calls the real producer, then compares its canonical
proof bytes with the newly generated CPU proof. Its production C/R/D call took
65.09 s; preparation took 35.00 s. A separate comparison checked all sixteen
returned matrices, all claims and openings, parent, transcript, identities,
and proof bytes against CPU: 585,569,271 reference bytes in total. Goldilocks
representatives are normalized for typed JSON comparison. All child matrix
files match exactly apart from permitted final line endings.

The generated Metal successor drove terminal verification with seven nonzero
running witnesses. Terminal checks made 959 device dispatches and accepted the
expected state; a changed expected state was rejected. A separate CPU phase
changed the first claimed `Eval_K`, recomputed the state hash, changed the
fresh public witness, and recomputed its commitment. Metal rejected this input
with `Eval_K differs from the complete witness opening`, as required.

[Logs, requests, and comparison records](tests/evidence/nonzero-fold-20260921)
retain these results and the failed CPU memory attempt. Each invocation kept
the repository's 300-second test cap. Preparation and file input/output repeat
across phases, so their sum is not a lifecycle benchmark. The uninterrupted
recursive lifecycle, 5× speed target, and general memory bound remain open.

For the CPU reference, `openings --step 2` can replace the individual `child`
invocations. It requires the completed `split` phase and must be followed by
`nifs`. `successor --step 2 --engine metal` uses device commitments. These are
test-driver choices, not new Cargo features or production APIs.

## Full lifecycle with direct mask upload

The device path previously packed all logical witnesses into a host mask
vector, then copied its nonzero prefix into Metal storage. PiCCS reserved
1,274,427,168 bytes for this vector on the selected profile; PiDEC reserved
1,199,460,864 bytes. A zero suffix still occupied the host vector. Encoding
also revisited every scalar coefficient even when signed masks already existed.

PiCCS and PiDEC now write signed masks directly into the final shared Metal
buffer. Only the prefix through the last nonzero witness occupies that buffer;
logical source counts and indices are unchanged. General digit inputs retain
one source's temporary encoding. This change adds no circuit-specific branch,
new feature, environment variable, or memory threshold. The block-count
accessor lets the device preparation validate all source widths before use.

The host allocation regression failed on the old path: one nonzero source
allocated 80 Rust heap bytes, while the same source with sixteen zero witnesses
allocated 1,104. It now requires and gets equal allocation for both families.
Thirteen joint device checks and nine engine parity checks pass; the two CUDA
checks remain ignored because their kernel is absent. The Clippy audit reports
only the existing boolean-expression warning, with no new warning in this change.

The uninterrupted public Metal lifecycle passed in 183.62 s, or 184.80 s with
the external runner. Peak RSS was 16,167,714,816 bytes. It includes preparation,
base proving, both active `extend` calls, terminal acceptance, and wrong-state
rejection. This replaces the earlier stopped public-lifecycle result.

The changed complete second-fold producer also passed exact CPU proof and
output comparison. Every one of its 945,983 canonical proof bytes and all
sixteen returned matrices match the existing CPU reference. The phase took
92.91 s with the runner and 15,983,771,648 bytes peak RSS; C/R/D itself took
54.53 s. The CPU reference was not rerun, and no Lean command ran.

The standalone production benchmark then ran on the same M1 Max, using the
fixed initial state `[202, 203, 204, 205]` and message `[7, 11, 13, 17]`:

| Phase | Metal time |
| --- | --- |
| Preparation | 34.75 s |
| Base proof | 6.81 s |
| First active extension | 50.23 s |
| Second active extension | 66.14 s |
| Terminal verification | 29.23 s |
| Complete lifecycle | **187.16 s** |

The benchmark emitted `benchmark_finished` with `verified: true`, the expected
final state, and the selected circuit identity. Peak RSS was **16,594,255,872
bytes**; the external runner took 187.83 s. These are complete process
measurements under the owner's approximate-16-GB allowance and the stated
16 GiB RSS guard. Physical footprint is also recorded as diagnostic data.

[Logs and records](tests/evidence/direct-witness-masks-20260921) retain the
allocation failure and pass, build and parity checks, proof comparison, public
lifecycle result, and production benchmark. Every native test and benchmark
used the 300-second cap. Full CPU timing is not available; the 5× comparison
still requires that run. This selected-circuit result does not establish a
memory bound for all supported circuits, and the earlier 21.27 GB CPU reference
still exceeds the production target.

## CPU carried storage and dense folding

The CPU carried-table builder previously made a full scalar combination of
the running witnesses, copied it into real and imaginary ring planes, and
built separate matrix and Pad projections. The selected carrier has
253,011,276 coefficients. A scalar extension-field vector occupies
4,048,180,416 bytes; the two dense real-field planes together occupy the same
space.

The final path combines only the ring block being projected. It evaluates
matrix rows from that projection, then reuses the same buffer for Pad. Logical
source positions and gamma exponents are unchanged. The output buffer also
covers relations with more rows than witness coordinates.

Dense SumCheck tables now fold inside their current allocations. Each worker
handles an even chunk; after all reads finish, the results are moved together.
Chunk sizes come from the active worker count. The completed SumCheck tables
are released before witness openings. No circuit, profile, transcript, feature,
or environment variable changed.

The allocation regression failed on the old carried path. The final path has
one full projection allocation and no full combined-witness planes. A separate
regression failed on the old dense fold's additional half-table allocation and
passes with in-place folding. Values match independent scalar calculations;
checks include zero sources, reused zero blocks, odd lengths, parallel chunk
boundaries, and rows beyond the witness carrier. The row-prover check and all
nine supported Nightstream engine parity checks pass. Two CUDA checks remain
ignored. The original complex-cache equivalence suite also passed 25 checks
while evaluating the first storage change.

All three production attempts kept the normal 16 GiB RSS guard and the
300-second cap:

| Attempt | Last completed work | Stop time | Peak RSS, bytes |
| --- | --- | --- | --- |
| Construct full ring planes directly | Preparation and cache | 50.61 s | 19,280,347,136 |
| Combine per block and reuse the projection | Oracle setup | 68.75 s | 18,899,304,448 |
| Also fold dense tables in place | Oracle setup and SumCheck | 77.99 s | 20,684,931,072 |

These stopped at different phases. Their peaks do not establish a reduction
in complete-run memory. The final attempt reported 15.37 s for oracle setup
and 19.87 s for rounds, then hit the memory guard during output openings. It
did not complete the saved production proof comparison. No memory exception
was used, and no Lean command ran. The three-round project fuse ended this
storage slice with the opening-phase memory requirement still open.

An additional `neo-reductions` parity suite had seven passes and eight failures.
The failing fixtures supply no running claim and stop at the unchanged
digest-only transcript guard, before the changed evaluator is created. The
guard and fixtures match `HEAD`; they were not repaired in this storage change.
This is not a workspace-wide green test result.

[Logs and checks](tests/evidence/cpu-carried-blocks-20260921) retain the failed
allocation checks, passes, every stopped production attempt, and those fixture
failures. The intermediate full-ring-plane approach was replaced; it is not
the final implementation. The next storage change below closes the production
PiCCS proof comparison. Full CPU lifecycle measurement, the 5× ratio, and the
general memory bound remain open. The earlier Metal benchmark remains a record
of its measured revision; measure both engines from one final build for the ratio.

## CPU opening buffer reuse

The completed carried SumCheck vector now becomes the opening scratch buffer.
Real and imaginary coefficients share its existing `Vec<K>` allocation. This
avoids two new coefficient planes totalling 4,048,180,416 bytes on the selected
carrier. The implementation uses safe Rust and keeps matrix evaluation order,
completion tails, and the protocol unchanged. Encoded witness folds also reuse
their code buffers through the same in-place pair-fold operation.

Both allocation regressions failed before their fixes and pass afterward.
The 23 applicable cache-equivalence checks, three row/opening checks, eight
CPU unit checks, and nine supported Nightstream engine comparisons pass. Two
unit checks requiring separate local captures and the two CUDA comparisons
remain ignored.

The first opening-reuse production attempt stopped before openings at
17,478,352,896 bytes RSS. With encoded-code reuse included, the complete second
production PiCCS phase passed in **182.43 s** (183.03 s including the runner).
It matched every saved CPU proof byte and output opening on identical sources.
Peak RSS was **17,103,896,576 bytes** (17.10 GB; 15.93 GiB), below the working
16 GiB guard of 17,179,869,184 bytes. Physical footprint was 13,033,157,640 bytes;
RSS remains the acceptance measure. Oracle setup took 16.30 s, rounds 19.69 s,
and output openings 105.01 s. No memory exception or Lean run was needed.

The changed decomposition opening path also passed on all sixteen saved
children. The seven nonzero child records match the CPU reference byte for
byte, including commitments, parent, and transcript; the nine zero children
pass their zero-opening checks. This phase took 182.02 s with 14,638,448,640
bytes peak RSS. The parent and digit witnesses were reused from the saved
reference; this run did not regenerate a complete C/R/D proof.

A fresh standalone benchmark from this final build passed the full three-step
Metal lifecycle in **179.514135833 s**, with **16,784,228,352 bytes peak RSS**.
Preparation took 34.67 s, base proving 6.81 s, the two extends 49.75 s and
64.25 s, and terminal verification 24.04 s. Inputs, circuit identity, and final
state equal the earlier Metal run. The exact executable is retained for the
pending full CPU measurement; no speed ratio is inferred from the PiCCS-only
CPU timings. The full CPU run still needs the separately requested timeout
and memory exception; the earlier reference exception was already used.

[Logs and comparisons](tests/evidence/cpu-opening-reuse-20260921) retain the
failed allocation checks, the stopped production attempt, passing runs, and
exact comparisons. The CPU measurements cover production phases. They do not
establish the full CPU lifecycle memory bound or the 5× lifecycle ratio.
Further memory tuning below the owner's approximate limit remains a later pass.

## Parallel PaperExact/Optimized cross-check

On 2026-09-20, five cross-check tests and two public engine-selection tests
passed in release mode. Each test invocation used the five-minute cap from
`AGENTS.md`. No Lean command or artifact generation was needed.

`Engine::Crosscheck` runs the reference prover on a separate thread while the
calling thread runs Optimized. Both receive copies of the same inputs and
transcript. The agreement test checks the returned proof bytes, accumulator,
witness values, and transcript against a direct optimized run. It also verifies
the proof and confirms that the reference read the original rows on another
thread. The fixture uses the Nightstream Goldilocks `b = 2`, `k_rho = 16`
profile, with nonzero carried data and a non-aligned logical width.

The rejection tests cover changed proof fields, child witnesses, claims,
parent authority, transcript state, and transcript cursor. They also exercise
different reference rows, a reference prover error, and a worker panic. Failed
comparisons and these failures leave the caller's transcript unchanged.

These are small-circuit checks. A full production cross-check was not run;
PaperExact has exponential cost. The full Metal and CUDA gaps above remain.

```sh
timeout --signal=KILL 300 cargo test -p nightstream --release --lib engine::parity::crosscheck
timeout --signal=KILL 300 cargo test -p nightstream --release --test engine_selection
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
