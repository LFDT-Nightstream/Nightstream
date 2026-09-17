# Fresh recursive-loop replay

Status: implementation and preflight validation; full execution pending.

This record covers a stronger local replay than the selected staged result at
`6c3c8c0f`. Start with the original accepted iteration-2 state and its 17
witnesses. Generate the full iteration-3 successor, use its exact returned
state and witnesses as the next input, complete iteration 4, then run the
existing terminal verifier and rejection checks.

Both folds preserve the selected production package, key and transcript:
Goldilocks, `b = 2`, `k_rho = 16`, 16 children and 14 matrices. Nebula is
inactive. Rust results are comparison targets for Lean. The second Lean source
projection reads only the first Lean fresh witness, fresh claim, child claims
and complete private digit ranges.

## Checked composition

`LeanGraph.Targets.CheckedRecursiveReplay` is the literal registered target.
`CheckedReplayComposition.accepted_and_handoff` derives acceptance of the
actual successor and of its literal reconstruction as the next prior.

The premises are initial terminal acceptance, a nonwrapping iteration, actual
C/R sampling and public-parent results, returned R blocks, successful D split
and acceptance, successful canonical row checks, and exact source/evaluation/
carrier/input/context custody. Parent validity, child openings, row validity,
norm and successor acceptance are conclusions.

`FreshRowsCheck.checkBlock_of_workerRanges` proves complete coverage for the
hardware-count ceiling ranges used by the executable. Each Poseidon unit keeps
one complete 94-row invocation. The other units are scalar canonical rows.
Every block uses the same existing numeric matrix evaluator and fixed
production polynomial. Padding rows are zero by a checked theorem.
`FreshCommitmentFold.completeRow_value` connects complete block accumulation
to the semantic commitment of the same carrier.

Each successor command now retains Rust's physical assignment as headerless
little-endian `u64` words: private values, the constant one, then public values.
Rust executes its own encoded caller inputs; Lean's caller packet is used only
for comparison. The coordinator compares this complete file with Lean's
`physical.bin` before comparing the logical assignment. The runner regression
fails if the physical output or this comparison is removed. Passing that
regression does not substitute for the two actual physical comparisons.
The default release test build, the selected fixture-producer build, and the
ordered static/build/axiom gates pass for this change. Its before/after
regression and gate logs are retained in `physical-comparison-handoff/` in
the run directory. The active run paused after `norm-after2-check`; the next
command met the held queue before computation. No completed output was removed.

Initial acceptance uses the successful iteration-2 terminal record in
`NATIVE_TERMINAL_EVIDENCE.zip`. Its manifest and child-witness manifest match
this run's copied envelope, fresh claim, fresh witness and all 16 children.
The successful accepted invocation took 267.164 seconds; the separate failed
311-second attempt supplies no acceptance evidence.

File parsing, task execution, ABI transport and compiler/runtime execution
remain explicit implementation boundaries. A receipt digest detects changed
files; it does not establish a protocol value or semantic acceptance.

## Execution and reproduction

Use the recorded Linux host and the source checkpoint for this run. Keep one
build/test queue. Subagents must be idle during measured scans.

The new run directory is outside Git:

```text
/home/nicoarq/develop/nightstream-stage1-evidence/recursive-loop-6c3c8c0f.d84yc9ll
```

Its `original-sources/` contains the original envelope, fresh claim, fresh
witness, 16 digit witnesses and `next-message-input.json`.
`original-package.json` is the exact selected package.
`original-inputs.json` records the copied input bytes. The application message
for both steps is `[7, 11, 13, 17]`; the next state is derived from the generated
successor, never substituted from an old result.

From the repository root:

```sh
export LEAN_SYSROOT=/home/nicoarq/develop/nightstream-stage1-evidence/recursive-loop-6c3c8c0f.d84yc9ll/lean-4.30.0-runtime-a6f4723408
export PATH="$LEAN_SYSROOT/bin:$PATH"
export LAKE_ARTIFACT_CACHE=false
python3 -B formal/nightstream-fprime/scripts/replay_recursive_loop.py \
  /home/nicoarq/develop/nightstream-stage1-evidence/recursive-loop-6c3c8c0f.d84yc9ll \
  2 all
```

For a new reproduction directory, copy only the pinned original inputs and
package into the same relative locations. Do not copy computed intermediates.
The coordinator records source and input identities before running. It fails
on changed or failed checkpoints, missing outputs, changed bytes, or changed
directory members.

Earlier receipts include owner-approved deadline-free runs. Current commands
use the project caps: 1,500 seconds for Lean and 300 seconds for native tests.
If measured work is too slow, optimize that computation with exact output
equality before extending it. The shared guard and one-command queue remain
in force. The coordinator sequences these
commands outside the graph lock. It builds each Lean executable before the
measured stages. Individual checkpoints are `build`, `prepare`,
`native`, `ccs`, `reductions`, `successor` and `terminal`. The `all` command
completes both successors before final terminal verification.

## Preflight evidence

These measurements are on the same Linux host, with subagents idle:

- Removing the duplicate Rust matrix-capacity scan reduced the complete native
  C command from 240.62 to 202.43 seconds. All bytes of the proof, C input and
  phase output match. Peak memory was essentially unchanged.
- The initial row checker timed out at 1,500 seconds. Proved worker ranges
  completed the same full retained carrier in 496.49 seconds, with
  1,698,180 KiB peak RSS. All 6,377,559 active rows, 14 ports, 270 public fields
  and 45 zero tail coefficients passed. Block 10 used 434.84 seconds.
- Eight row-check rejection cases passed, including changed public/private
  values, malformed encodings, nonzero tail and existing output.
- The source projection and coordinator checks passed 24 contract tests.
  The graph suite has 86 passing tests, including owner-authorized deadline removal. Exact target closure and
  allowed-axiom checks passed.

The retained-carrier scan is a preflight, not fresh-loop evidence. Full C/R/D
generation, byte comparisons, both new carrier checks, exact feedback and final
terminal acceptance/rejections remain required before this record can close.

Protected external acceptance remains pending by owner choice. This record
does not claim a production SNARK backend `prove → verify` run or universal
proof of the filesystem, compiler or Rust runtime.

The deadline-control change preserves every Lean and Rust arithmetic source.
The run retains both source pins and an explicit transition record. Completed
checkpoints retain their original commands, deadlines and byte identities. One
producer completed while its active Bash wrapper was edited; the wrapper then
failed with exit 127. That attempt is retained separately and is regenerated
under the new source pin. It is not counted as a passed checkpoint.

When deadlines are disabled, the coordinator gives all original C matrix
requests to one existing multi-range invocation, and all D matrix requests
to another. This preserves request order, ranges and output files while
removing 11 C and 38 D repeated source loads per fold. Both executables already
process these requests sequentially and release range-local working data.
Arithmetic code and the complete Lean–Rust byte comparisons are unchanged.
The scheduling test checks that the complete request/output sequences match.

The initial Linux runtime used the existing fork commit
`14f2fed9c9782048e896c924f424624585acd772`. Only `mpz.cpp.o` in a separate
copy of the official Lean 4.30.0 `libleanrt.a` changes. All other archive
members match byte for byte; the compiler and proof checker are unchanged.
The build recipe, compiler/header identities and native conversion tests are
retained in the run directory. The coordinator also pins the selected runtime
archive, compiler and shared library, so a later runtime change fails resume.

A matched production-source matrix-prefix range `1518288..1534672` used all
16 workers, with all subagents idle. Command time fell from 52.85 to 47.43
seconds; compute time fell from 32.483 to 27.717 seconds. Peak RSS was
2,461,264 and 2,456,944 KiB. All 33 files, totaling 898,651 bytes, match exactly.
This is one range measurement, not a complete-loop speedup. The complete
new-loop comparisons and final verification remain required.

The subsequent native dot-product change is recorded in
`PICCS_NATIVE_DOT_PERFORMANCE.md`. Its equality theorem covers arbitrary
input vectors. The same production prefix takes 30.91 seconds versus
48.24 seconds with the saved baseline; all 33 files match byte for byte.
Static, build, axioms, identity, boundary and changed-coefficient checks pass.
The fresh replay retains its completed outputs across an explicit source
transition; this optimization does not close the full-loop record.

The next selected runtime is fork commit
`a6f47234088e5b35c7a5b1b336db6372f797d998`. Its persistent unary closure
fast path reduces the measured fresh-polynomial command time by 1.9–3.6%;
matrix computation time is effectively unchanged. Full package, binding,
fresh-polynomial and matrix-prefix output bytes match. The ownership and
existing closure tests pass. `LEAN_REPLAY_PERFORMANCE.md` and
`LEAN_RUNTIME_APPLY_PERFORMANCE.json` state the exact scope and measurements.
Resume preserves the completed stage outputs, checks their saved identities,
archives obsolete build receipts, and records the native-dot and runtime
source transition. Each new command uses the current project execution caps.
