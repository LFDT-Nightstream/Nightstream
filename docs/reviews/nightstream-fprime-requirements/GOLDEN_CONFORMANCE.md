# Selected golden-vector conformance

Status: active. This record belongs to `nico/golden-conformance`.

The goal is reproducible, required conformance checks for the existing
Nightstream Goldilocks profile (`b = 2`, `k_rho = 16`, `B = 65536`). The
engineering target is at least 9/10 confidence for the selected first-fold
and recursive cases. This rating is not a cryptographic security bound.

## Acceptance

- Restore the recorded archives into a clean directory and run the selected
  comparisons there. A local restore does not close protected external
  reproduction or backend approval.
- Connect the Lean-checked reference to the current CPU result used by every
  engine comparison. Compare actual canonical proof bytes and the complete
  additional fields in scope. SHA-256 identifies evidence files only.
- Keep first-fold Lean checking. Check fresh recursive CPU C/R/D outputs with
  Lean, and record mutation rejection by its actual verifier or decoder.
- Run current engine code. CPU is the common reference. Other engines need
  CPU comparisons on the same inputs and output scope, without separate Lean
  generation for each engine.
- Enforce the staged checks for an explicit source/build/input dependency
  set. Missing artifacts, absent tests and selected job failures must fail.
- Independently generate the exact first-fold expectation and complete the
  registered recursive loop. Each producer uses its own successor as its
  next input. Rust expected outputs never produce Lean expected values.

## Execution rules

Use the existing Python coordinators and Lean/Rust producers. No new Rust
feature, environment setting, proof format, protocol parameter or hash family
is part of this task.

Each native test invocation has an outer 300-second cap. Each Lean command
uses `formal/nightstream-fprime/scripts/validate.sh` and an outer 1,500-second
cap. Use one build/execution queue. The current production RSS guard is
16 GiB, as recorded in `NIGHTSTREAM_CRATE_GOAL.md`; old exceptions do not
authorize a new run. Historical results retain their original scope.

## Current work

- Exact nonzero archive/CPU/engine comparison: implemented; focused positive
  and negative tests pass. Engine successor/caller comparison is not claimed
  by a fold-output-only comparison.
- Fresh current CPU export for the Lean checker: implemented; both focused
  Rust checks pass, including every recomputed C trace field.
- Required-check dependency selection: implemented; focused tests pass.
- The existing independent recursive replay stopped at an optional final-LF
  difference in its PiCCS input. The checker now follows the Lean parser's
  exact optional-LF contract; the stopped comparison and its changed-target
  checks pass without changing the saved producer values.
- All five archives were restored into a new directory. Fresh Lean C/R/D
  checking and the native comparison of that restored nonzero proof pass,
  including all 945,983 canonical proof bytes and 55 native PiDEC rejections.
  `GOLDEN_CONFORMANCE_CHECKS.json` records this local scope.
- Full producer execution, fresh verifier mutation records, required CI
  execution, and independent generation remain open. The native driver now
  covers the exact third fold needed for the registered iteration-4 check;
  its full execution remains pending.

Large generated inputs and outputs stay outside Git. Completed commands,
source versions, exact comparison scope and failures will be recorded here
or in adjacent JSON evidence. No completion is inferred from old receipts.
