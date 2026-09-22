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
- The current CPU run passed all 25 staged phases: folds 1–3, iteration-4
  terminal acceptance, rehashed false-opening rejection, and both archive
  comparisons. Phase time totaled 2,230.22 seconds; the longest phase took
  180.06 seconds. Peak RSS was 14,038,003,712 bytes, below the 16 GiB guard.
  This is staged execution, not a complete-call performance measurement.
- Fresh Lean checks pass for all three folds. Each compares every canonical
  proof byte, 177,326 private caller words, 278 public caller words, all seven
  caller result fields, and 234,755,400 physical-witness bytes. Each records
  34 Lean D public rejections, ten decoder rejections, one explicit C
  rejection, the parent-bound and blocked-handoff checks, and 55 native D
  rejections. C proof messages remain verifier inputs in these checks.
- The current recursive proof matches all 945,983 restored reference bytes.
  The complete independent C input, phase and proof words also match the
  current CPU values. An explicit source/input audit permits reuse of the
  retained C computation. Its 21 previously unexecuted finish rejection cases
  now pass. R/D and successor work uses a separate continuation directory;
  the old failed receipt remains unchanged.
- Independent iteration-2 reductions now pass: all 39 D matrix batches,
  final D acceptance, all 945,983 proof bytes, 55 native D rejections and
  30 Lean finish rejections. C remains retained imported computation.
  The direct current CPU parent comparison also passes for all 253,011,276
  canonical field coefficients, including tail-mutation rejection. This is
  field equality; the JSON representations differ. All 16 current child
  witness files match the retained comparison targets in 579,873,475 bytes.
  The parent receipt was written on 2026-09-22 at 01:33:25 UTC.
- The independent 2→3 successor and current CPU comparisons pass. They cover
  all proof bytes, 177,326 private and 278 public caller words, all seven
  caller result fields, 234,755,400 physical-witness bytes, the complete fresh
  witness and claim, and all 16 children. Only a final LF differs in the
  fresh witness and claim JSON. The new carrier passed the canonical row
  check for 6,377,559 active rows, eight row rejection cases and 26 physical,
  assignment and commitment rejection cases. Its 1,188 commitment and 270
  public coefficients match. These comparisons completed at 01:53:58 UTC.
  Iteration-3 source preparation then passed at 01:54:33 UTC on 2026-09-22;
  it reads only Lean's own returned fresh witness, claim, children and digit
  ranges. This completes the handoff, not the next independent fold.
- Required-check workflow wiring is implemented. Actual CI execution still
  needs the three missing published archives, confirmed Metal runner routing,
  and branch rules that require the final check. Local CPU-to-Lean handoff
  checks pass. Current Metal execution remains pending.
- Independent first-fold generation is implemented but has not run. The
  independent 3→4 generation, successor and final terminal checks remain
  pending, so the complete independent 2→3→4 sequence is still incomplete.
  The 9/10 target is not yet complete.

Large generated inputs and outputs stay outside Git. Completed commands,
source versions, exact comparison scope and failures will be recorded here
or in adjacent JSON evidence. No completion is inferred from old receipts.
