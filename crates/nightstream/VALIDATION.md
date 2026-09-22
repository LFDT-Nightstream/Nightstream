# Replacement crate validation

Work branch: `nico/nightstream-crate`, based on
`9787d8e77069246e3e2afc7dcfab755556fd5023`.
The unchanged `neo-fold-clean` source is the migration reference. Its only
dependency edge from this crate is for development comparisons.

The selected implementation goal is complete. The
[fresh replay receipt](tests/evidence/fresh-recursive-replay.json) records all
29 successful phases, both complete output comparisons, and the failed
successor attempt that led to the batch commitment change.

That result covers the CPU migration. The later engine work has a completed
[prepared-package CPU/Metal comparison](#final-prepared-package-cpumetal-comparison).
A universal process-RSS guarantee remains unproven.

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

## Metal GPU profiling

Instruments now has the owner's 30-minute cap in `AGENTS.md`; other non-Lean
tests keep the five-minute cap. A native three-step capture verified in
190.29 s under profiling and saved successfully. Its uninstrumented baseline
took 179.51 s; these are different measurement conditions.

The native GPU timeline records 119.98 s of active work. Commitment sequences
account for 60.78 s, separately recorded opening product/bar-transform intervals
for 22.70 s, and geometric opening forms for 13.72 s. Shader sampling attributes
48.97 s to the primary commitment kernel and only 0.32 s to commitment reduction.
The chunk-sum loop is not a major measured cost. Timing data can combine
encoders, and shader samples do not account for every GPU-active interval.

Encoder labels were added to identify these stages. Arithmetic and protocol
are unchanged. The capture uses the same inputs, circuit identity, and final
state as the saved benchmark. Limiter exports did not yield usable values;
no occupancy, bandwidth, ALU-utilization, or spill-rate result is claimed.
[The profile record](tests/evidence/metal-gpu-profile-20260921) contains the
measurements, native pipeline limits, capture issues, and the Apple/CUDA
references used to guide the next optimization. No algorithm optimization was
made in this profiling slice.

## Profiled Metal opening changes

The profile led to two arithmetic-preserving changes. Geometric forms now
compute each coefficient once for both extension components. Sparse ring
products skip multiplication by one; other magnitudes keep their multiplication.
Both changes use the existing buffers and support the same matrix inputs.

The retained standalone three-step benchmark verifies in **167.804142542 s**
with **16,858,628,096 bytes peak RSS** (16.86 GB; 15.70 GiB). Preparation takes
34.55 s, base proving 7.10 s, the two extends 46.01 s and 58.27 s, and terminal
verification 21.87 s. These times include the full lifecycle on the saved inputs.
The saved baseline took 179.51 s; a repeated baseline with encoder labels and
unchanged arithmetic took 193.31 s. The retained result uses 6.5% and 13.2% less
time, respectively. These are individual runs with visible timing variation;
the smaller measured difference is used in the summary.

The two opening tests compare exact CPU results, including overlapping
geometric runs, zero and negative ratios, both extension components, reused
form storage, and partial blocks. A fresh nonzero-carried production fold
matches all **945,983 CPU proof bytes**, every claim and opening, the transcript,
and all **16 returned witness matrices**. That check takes 86.42 s, including
preparation and file work, with 16,322,904,064 bytes peak RSS. No Lean run was
needed. The stored CPU proof and inputs remain the reference.

A center-out commitment thread order passed correctness checks but took
171.18 s for the lifecycle. It was removed because it showed no gain.

[Evidence](tests/evidence/profiled-metal-openings-20260921) contains the staged
measurements, CPU comparisons, and rejected candidate result. These are Metal
improvements; the full CPU timing and 5× ratio remain open. RSS stayed below the
working 16 GiB guard in these runs. This does not establish a bound for every
supported circuit.

## CPU lifecycle and terminal profile

The full Optimized Time Profiler run completed preparation, base proving, and
both recursive folds. It reached the owner-approved 30-minute Instruments
deadline during terminal verification. The partial trace could not be exported.
This run does not provide a full lifecycle time or a 5× CPU/Metal comparison.

A separate CPU terminal phase passed on the stored Metal step-3 envelope. It
accepted all commitments and openings and rejected a wrong final state. All
result fields except the engine name equal the Metal result, including all
16 running claims. The 19 input files are byte-identical. The phase took
448.09 s, including 34.85 s of cold preparation, JSON loading, a new CPU cache,
and terminal checks. This is not the warm terminal time in the Metal benchmark.
The observed kernel peak RSS was 13,893,877,760 bytes. The peak could not be
read after exit, so the final unsampled interval is not proved.

The saved trace contains 2,723,146 CPU samples. Release Rust symbols were
stripped, and a rebuild has different code bytes; a global address mapping
is not justified. Exact, unique function-byte matches identify 14 sampled
functions. Random-block generation accounts for 39.46% of all self samples,
and 128-bit division accounts for 29.85%. These are CPU weights across threads,
not wall times. Unmatched functions remain unnamed.

Maintenance Sleep occurred during both CPU captures. Keep
the original times; do not subtract sleep intervals to estimate a corrected
ratio. Future measurements use a command-scoped `caffeinate -is` assertion
while on AC power.

The memory review found a specific gap in the accepted row bounds. A circuit
with 239,217,427 application rows has 245,587,286 logical rows. The current
one-witness, 14-matrix application table would request 27,505,776,032 bytes.
The device guard rejects that allocation. This source-derived result shows
why bounded row evaluation is still needed; no large circuit was generated.
Other CPU and device allocations also remain part of the total RSS requirement.

[Evidence](tests/evidence/cpu-lifecycle-profile-20260921) contains run logs,
RSS records, the exact terminal comparison, clock evidence, and the allocation
derivation. No Lean command ran. No full CPU time or universal memory bound
is claimed.

## CPU indexed-key reduction

The streamed key loader now uses the Goldilocks identities `x^2 = x - 1` and
`x^6 = 1`, for `x = 2^32`, to replace eight 128-bit remainders per coefficient.
The scalar setup function retains the division formula as an independent
reference. Key bytes, addresses, and commitment semantics remain unchanged.

Serial production Sources checks on identical inputs reduce the fresh
commitment time from **95.144954 s to 58.379275583 s**, or **38.6% less time**.
The whole phase falls from 130.10 s to 93.32 s; preparation takes 34.63 s in
both runs. Peak process RSS is 6.07 GB before and 6.12 GB after. Each run
recomputes all saved commitments. This phase has one fresh witness and sixteen
zero running witnesses; it is not an active fold or full lifecycle benchmark.
Both commands use `caffeinate -is` on AC power, the 300-second cap, and the
current 16 GiB RSS guard.

Twelve Ajtai tests, three Metal commitment tests, and all nine supported engine
comparisons pass. The checks include stored Lean values, generated addresses,
the maximum nonce words, independent ring products, complete small C/R/D
proofs, and parallel PaperExact/Optimized cross-checking. CUDA remains two
ignored comparisons. Formatting and whitespace checks pass. No Lean command
ran, and no feature or environment variable was added.

[Evidence](tests/evidence/cpu-key-reduction-20260921) contains the before/after
logs, RSS records, exact input comparison, and test results. Full CPU lifecycle
time, the 5× ratio, and bounded storage for all supported circuits remain open.

## Bounded Metal application tables

Application values now use a resident prefix when it fits the available device
workspace, or bounded replay of original row windows through all preceding
challenges. Window evaluation, local folds, and weighted accumulation run on
Metal. The first F-to-K fold reuses each input pair's 16 bytes; later folds
release the old table before proceeding. Round kernels preserve global
equality, assignment, and carried indexes. Terminal checks also use windows
and report global failure rows.

Forced seven-, thirteen-, and seventeen-row tests match complete Optimized
proof bytes, every round and challenge, transcript state, and all output
openings. The derived application payload limits are 672 and 1,792 bytes.
The tests cover multiple prior challenges, larger source windows, transition
to resident storage, an odd last row, and a terminal failure in the last
window. All 22 Metal library tests and nine supported engine comparisons pass;
CUDA remains two ignored comparisons.

A fresh production second fold matches all **945,983 CPU proof bytes**, all
sixteen returned matrices, and complete claims, openings, parent, identities,
and transcript. The check takes 85.86 s, including preparation and file work,
with **16,384,065,536 bytes peak RSS**. The C/R/D phase takes 47.99 s. All runs
use the ordinary 300-second cap and the current 16 GiB RSS guard where measured.
No Lean command ran. No feature, environment variable, or production threshold
was added.

The application payload bound removes the need for the previously identified
27.51 GB table and 55.01 GB first-fold overlap. Other host/device caches and
indexes, witness storage, and generic seeded partials remain outside this
bound. No total-RSS result for every circuit or 5× full lifecycle ratio is
claimed. [Evidence](tests/evidence/bounded-application-20260921) contains tests,
exact production comparisons, resource scope, and build records.

The complete three-step Metal benchmark then passes in **164.661508625 s**,
with **16,813,703,168 bytes peak RSS** (16.81 GB). Preparation takes 34.50 s,
base proving 6.80 s, the two extends 45.61 s and 57.81 s, and terminal
verification 19.94 s. Starting inputs, profile, circuit identity, and final
state match the prior Metal run. The exact release executable retains function
symbols and uses command-scoped sleep prevention. This is one measurement;
no speed gain over earlier builds is inferred. The full CPU comparison must
use this saved executable.

## Matched full lifecycle profiles

These results use the saved build from before bounded matrix-row generation.
They do not establish the speed of the newer implementation below.

Serial Time Profiler captures on the same saved release executable measure
**7.62× Metal speed over Optimized** for preparation, base proving, two active
folds, and terminal verification. Both runs finish successfully and verify the
same final state. All starting fields except the engine and all final fields
except elapsed time match, including profile and circuit identity.

| Phase | Optimized CPU (s) | Metal (s) |
| --- | ---: | ---: |
| Preparation | 34.524239208 | 34.526846625 |
| Base proving | 59.835702167 | 6.804702334 |
| Extend 2 | 396.958310208 | 45.611830292 |
| Extend 3 | 496.363860083 | 57.861076000 |
| Terminal verification | 268.338163042 | 19.932587750 |
| **Full lifecycle** | **1,256.020431625** | **164.737161375** |

This exceeds the 5× target under matching Time Profiler instrumentation. It is
not a measurement of the CPU run without Instruments. Approval for one such
30-minute CPU invocation is pending; that command has not run. The saved
executable has UUID `D66506D5-8D84-3D01-8EEB-E21332765BD7` and retains symbols.
Optimization settings are unchanged. Both captures use the same one-millisecond
CPU sampling settings, `caffeinate -is` on AC power, and the existing RSS guard.
Instruments reports nominal thermal state throughout both captures.

The observed kernel lifetime RSS peaks are **16,133,701,632 bytes on CPU** and
**16,628,383,744 bytes on Metal**. Post-exit peak reads are unavailable, so the
final unsampled intervals are not proved. Recorder and target cleanup completes
normally. Including trace finalization, the controllers take 1,388.24 s and
171.20 s, within the 1,800-second Instruments cap in `AGENTS.md`. No timeout,
sleep interval, or preparation time is subtracted from the benchmark result.

The separate production comparison above establishes complete CPU/Metal proof
and output equality for the current implementation. These lifecycle logs
establish matching inputs and accepted final states; they do not contain proof
bytes. [Evidence](tests/evidence/lifecycle-time-profile-20260921) contains both
native logs, capture controls, resource records, and the exact comparison.

The complete CPU sample export assigns 54.07% of backtrace weight to random-block
generation. In the Metal run, shared Poseidon2 and signed-column PiRLC work use
51.89% and 22.74% of host CPU backtrace weight. These are sampled CPU weights
across threads, not wall or GPU times. Every native address resolves against the
saved executable's matching UUID and recorded load address. Complete compact
function totals retain unknown frames and all self/inclusive weights. The
separate device profile above records GPU kernel costs.

The general memory bound remains unmet. Repeating a three-term equality adds
application rows without variables and fits the current shape checks. At the
accepted row bound, one host matrix-run vector alone requires at least
**17,223,654,456 bytes**. Keeping its device copy doubles that payload before
row indexes, witnesses, or other caches are counted. This source-derived case
requires matrix windows to be generated before a full cache is built. Eager
application rows remain another preparation-memory gap. No large circuit or
Lean command ran for this review.

## Bounded matrix row windows

CPU and Metal now read original package rows into bounded local caches, with
global columns and equality weights. Full host/device matrix caches are no
longer a prerequisite. The initial production check reached the 300-second cap.
A stopped Time Profiler capture identified repeated row-window construction;
the range retry now makes geometric progress, and scalar entries retain the
existing parallel/tiled device opening path.

The corrected production second fold matches every saved CPU proof byte
(945,983 bytes), all sixteen complete returned matrices, parent, claims,
openings, transcript and identities. The nineteen source files also match by
complete byte comparison. Actual C/R/D takes 102.544524166 s; the test including
preparation and file I/O takes 139.837589042 s. Final native peak RSS is
13,756,530,688 bytes (13.76 GB), within the owner's approximate limit. This is
less memory and more time than the preceding saved implementation.

Validation passes: 15 CPU unit tests (two existing ignored), 11 row-source tests,
23 Metal unit tests, and nine supported engine comparisons. Two CUDA comparisons
remain ignored. The fixed-prefix scan checks 6,369,859 rows; its maximum encoded
one-row workspace is 13,942 bytes, below the selected minimum workspace allowance
of 4,750,236,214 bytes. These local payload bounds do not prove total process RSS.
Eager application/assembly storage remains open. The full Metal run completed
preparation (34.53 s), base proving (6.79 s), and both extensions (90.05 s and
119.10 s), then reached the 300-second cap during terminal verification. It is
a failed complete-run check, with no final native RSS result. A subsequent
change borrows row terms instead of allocating per-visit vectors. It passes the
same complete production proof and output comparison: C/R/D 98.162725292 s,
test 135.484281125 s, final native RSS 13,794,082,816 bytes (13.79 GB).
The same final source completes a real full lifecycle Time Profiler capture in
291.840242833 s: preparation 34.56 s, base 6.79 s, extensions 86.64 s and
115.30 s, terminal verification 48.55 s. Final state, circuit identity, profile
and all benchmark inputs match the prior saved run. Thermal state is Nominal
throughout. The observed kernel lifetime RSS peak is 13,880,573,952 bytes;
the post-exit peak is unavailable, so the last unsampled interval is not proved.
Recording and finalization finish in 299.35 s under the 1,800-second cap.

All 5,851 native sample addresses resolve against the saved executable UUID
`5841982D-F5C1-35FA-903D-66E7053B74CB`. Matrix-window construction accounts for
117.359 s of inclusive CPU sample weight out of 251.270 s with backtraces.
Template substitution accounts for 47.495 s within those stacks; these inclusive
values overlap and must not be added. Fourteen stored matrix/formula reference
tests also pass. The matched CPU Time Profiler run completes in
1,330.702711417 s, so the same-build full lifecycle ratio is **4.5597×**, below
the 5× target. Both runs use the same inputs and return the same final state,
circuit identity and profile. CPU preparation takes 34.54 s, base proving
60.09 s, extensions 420.10 s and 530.47 s, and terminal verification 285.50 s.
The observed CPU lifetime RSS peak is 16,790,093,824 bytes (16.79 GB); the final
post-exit peak is unavailable. Trace recording and finalization take 1,461.62 s,
within the approved 1,800-second Instruments cap.

The same saved executable also passes without Instruments under the ordinary
300-second cap: **290.798678833 s**, with final native peak RSS
**13,848,379,392 bytes (13.85 GB)**. Preparation takes 34.46 s, base proving
6.79 s, the extensions 86.07 s and 114.95 s, and terminal verification 48.52 s.
This closes the complete-run memory measurement for this benchmark. It does
not close the general application-storage bound or the new 5× comparison.
[Evidence](tests/evidence/matrix-row-windows-20260921)
contains the failure, diagnostic profile totals, corrections and completed checks.

## Sealed application records and reusable row buffers

The application builder writes original rows, exact recipe syntax, offsets and
recipe links to private files. Sealing closes write handles; a partial append
prevents sealing. The loader, witness executor and both canonical identity
occurrences share one immutable owner. Generated columns map by checked
arithmetic. No complete application row, recipe or hash-word vector is retained.

Template execution also reuses thirteen port buffers and lends one row's run
slices to each consumer. It no longer clones and merges each input form or
constructs every row in an invocation before visiting the first row. This keeps
the saved formulas, coefficients, order, and early-stop behavior.

Checks pass: eight application tests, five record-owner tests, five native-loader
tests, six identity tests, five assembler tests, two public assembly tests,
sixteen matrix formula tests, and nine supported engine comparisons. Two CUDA
checks remain ignored. Identity checks compare every word of the complete
saved envelope before checking its fixed identities. Workspace release checks
with all targets and Metal/CUDA, formatting, and whitespace checks pass.

The production second Metal fold matches all 945,983 saved CPU proof bytes,
all sixteen returned matrices, parent, claims, openings, transcript and identities.
All nineteen input files also match by complete byte comparison. C/R/D takes
83.36266325 s; the full test takes 120.504824708 s. Final native peak RSS is
13,193,953,280 bytes. The full public Metal lifecycle passes without Instruments
in 264.522361916 s with 12,748,128,256 bytes peak RSS. Preparation takes 34.38 s,
base proving 8.46 s, extensions 76.75 s and 103.16 s, and terminal verification
41.78 s. Complete start and finish records match the prior saved Metal run
except for elapsed time. Matching Time Profiler runs on the same saved image
(`8B18A367-06BA-3915-8254-40C42C9C0620`) take 1,308.784675084 s on CPU and
276.998745834 s on Metal: **4.724875815×**, below the 5× target. Both complete
start records match except for engine; finish records match except for time.
Observed lifetime RSS peaks are 15,786,639,360 bytes for CPU and 12,633,473,024
bytes for Metal. The final post-exit peaks are unavailable. Recording and trace
finalization finish in 1,435.69 s and 284.38 s, each within the 1,800-second cap.
Both traces report Nominal thermal state throughout. The Metal capture attributes
97.132 s of summed CPU sample weight to matrix-window construction: counting
59.258 s, filling 37.677 s, and other construction 0.197 s. The count/fill split
uses full sampled stacks and the exact frozen source/image. All 5,317 native
addresses resolve; 2,230 sentinel samples and 995 unresolved top samples remain
unassigned. These are CPU sample weights, not elapsed or GPU time.

The fixed production key bounds application variables to 7,709. A single native
row therefore cannot retain arbitrary raw duplicate terms in memory. This is
source-derived payload evidence, not a universal RSS result; source scratch,
allocator and driver residency remain outside parts of the workspace accounting.
No Lean command, new feature or environment variable was used.
[Evidence](tests/evidence/sealed-application-records-20260921)
contains exact comparisons, logs, the memory review and saved-image metadata.


## Incremental matrix-row counting

Matrix-window counting now keeps the accepted prefix separately from candidate
counts. Runs add their exact encoded payload cost, and each first offset family
charges all preceding rows. Row completion uses cumulative scalar totals instead
of recounting every matrix. Single-input template substitution also keeps the
existing canonical key order without sorting it again.

Twelve row-storage tests, sixteen saved formula tests, twenty-three Metal tests
and nine engine comparisons pass; two CUDA checks remain ignored. Exact-budget
and one-byte-short tests cover empty rows and offset families that appear late.
Workspace release checks with all targets and Metal/CUDA pass.

On saved image `89776242-C7BC-32FF-8174-47B117EBFF79`, the production second
fold matches all 945,983 CPU proof bytes and all sixteen complete child matrices,
parent, claims, openings, transcript and identities. All nineteen source files
also match exactly. C/R/D takes 77.438333583 s; final native peak RSS is
13,197,393,920 bytes. The complete raw Metal lifecycle takes 250.277271958 s with
12,681,920,512 bytes final native peak RSS. It verifies the same final state.
These measurements do not establish a matched ratio for this image.
[Evidence](tests/evidence/incremental-row-count-20260922) records the checks,
complete comparisons, image metadata and profile review.


## Recipe-stack batch reuse

Native witness execution now reuses one disk-backed continuation stack per
nonempty recipe batch. Each evaluation clears the logical stack first, including
after a prior lookup error. Records remain immutable and resident scratch stays
bounded. Nineteen application/native-loader checks and four saved Lean Poseidon2
checks pass, as do workspace release checks with all targets and Metal/CUDA.

Saved image `9F975E6F-3954-3D37-8C7D-97A6BBD37CDA` completes the raw Metal
lifecycle in 244.83093 s with 12,717,260,800 bytes final native peak RSS. Matched
Time Profiler runs take 1,300.842536625 s on CPU and 246.453710167 s on Metal:
**5.278242862×**. Complete start records match except engine and finish records
match except elapsed time. Device and recording settings match. Both captures
report Nominal thermal state and finish within the 1,800-second cap, including
trace finalization. Observed lifetime RSS peaks are 15,783,100,416 bytes on CPU
and 12,711,673,856 bytes on Metal; final post-exit peaks are unavailable.

These runs include compilation. The owner has now changed the target to package
loading, proving and terminal verification, with compilation measured separately.
The prepared-package image before the fixed-metadata node guard measured
5.9191× with compilation excluded. The final guarded image is recorded below.
[Evidence](tests/evidence/incremental-row-count-20260922)
contains the saved image, raw records, metadata and checked comparison.

## Final prepared-package CPU/Metal comparison

Final image `7DCAD994-817D-33E5-95BB-666FA9543762` measures **5.933208287×**
over package loading, proving and terminal verification. Compilation is excluded.
Both engines load the same caller-selected package. Loading validates execution
data without imposing authentication or repeating whole-circuit identity hashing.

| Engine | Matched Time Profiler total | Observed peak RSS |
| --- | ---: | ---: |
| Optimized CPU | 1,262.100768458 s | 14,448,033,792 bytes |
| Metal | 212.71809575 s | 11,474,714,624 bytes |

Both completed captures were serial, with the same image, package, full inputs,
final state and recording settings. Both reported Nominal thermal state.
An initial Metal recorder hung during trace finalization after its target
verified. The owned recorder was stopped; only the completed retry enters this ratio.

Raw Metal passes in **212.469842375 s** with **11,374,968,832 bytes** final
native peak RSS; loading and engine setup take 2.152479833 s. Compilation takes
34.857374625 s and saving 0.133257583 s. The 127,306,104-byte package matches
the prior package byte for byte. Nine prepared-package tests, one maximum/zero
compiler-shape regression and four public cache tests pass.

Both measured engines remain below the accepted 16 GiB RSS guard. The source
and tests establish bounded native storage; these fixture measurements do not
prove a universal process-RSS guarantee for every supported circuit. Complete
records, the compiler-derived fixed-metadata bound, and scope are in the
[prepared-package evidence](tests/evidence/prepared-package-20260922).

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
timeout --signal=KILL 300 cargo test -p neo-fold-legacy --release --test nightstream_baseline unchanged_old_poseidon_base_lifecycle -- --exact --ignored --nocapture
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
