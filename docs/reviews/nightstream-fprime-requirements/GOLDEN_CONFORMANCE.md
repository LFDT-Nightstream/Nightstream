# Selected golden-vector conformance

Status: complete for the selected 1→2→3 goal, at 14:04:47 UTC on 2026-09-22.
This record belongs to `nico/golden-conformance`.

The goal is reproducible local conformance checks for the existing
Nightstream Goldilocks profile (`b = 2`, `k_rho = 16`, `B = 65536`). The
engineering target of at least 9/10 conformance confidence is met for the CPU–Lean
1→2→3 sequence: the first fold and one recursive fold, with their complete
input connection. This rating is not a cryptographic security bound or a
universal soundness claim. At the owner's request, CI enforcement and Metal validation are out
of scope. Neither is an acceptance criterion or a pending blocker.

## Acceptance

- Restore the recorded archives into a clean directory and run the selected
  comparisons there. A local restore does not close protected external
  reproduction or backend approval.
- Connect the Lean-checked reference to the current CPU result.
  Compare actual canonical proof bytes and all additional fields in scope.
  SHA-256 identifies evidence files only.
- Keep first-fold Lean checking. Check fresh recursive CPU C/R/D outputs with
  Lean, and record mutation rejection by its actual verifier or decoder.
- Run current CPU code for the selected first-fold and recursive cases.
- Run the staged checks locally with recorded source, build and input
  identities. Missing selected artifacts or failed phases must fail the run.
- Independently generate the exact 1→2 expectation and connect its complete
  successor to the inputs of the completed independent 2→3 fold. Compare
  all 17 source witnesses and their public values, package and context,
  state and message request, carried parent, all 17 successor digest frames,
  caller C/R/D links and the public input consumed by the retained C replay.
  A digest match alone does not establish this input connection.
- Check final-state-3 terminal acceptance and the `ce-evaluation`,
  `ce-matrix-evaluation` and `fresh-private` rejections. Rust expected outputs
  never produce Lean expected values.

The registered `fresh-recursive-loop` obligation requires the broader
2→3→4 replay. It remains open; completion of this selected 1→2→3 goal does
not close that obligation. A third independent fold is not required here.

## Execution rules

Use the existing Python coordinators and Lean/Rust producers. No new Rust
feature, environment setting, proof format, protocol parameter or hash family
is part of this task.

Each native test invocation has an outer 300-second cap. Each Lean command
uses `formal/nightstream-fprime/scripts/validate.sh` and an outer 1,500-second
cap. Use one build/execution queue per worktree. Other worktrees can run
their own builds with separate writable build outputs. The current production
RSS guard is 16 GiB, as recorded in `NIGHTSTREAM_CRATE_GOAL.md`; old exceptions do not
authorize a new run. Historical results retain their original scope.

## Current work

- Exact nonzero archive/CPU/engine comparison: implemented; focused positive
  and negative tests pass. Engine successor/caller comparison is not claimed
  by a fold-output-only comparison.
- Fresh current CPU export for the Lean checker: implemented; both focused
  Rust checks pass, including every recomputed C trace field.
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
- Independent iteration-3 C generation passed at 06:21:43 UTC on 2026-09-22,
  using Lean's own successor inputs. All 28 rounds and 12 matrix batches
  passed. Comparisons cover 560 round field words, 28 transcript transitions,
  all 27,540 evaluation field words, 29,288 proof-input words, 15 phase fields
  and eight outgoing-state words. The 754,531-byte Lean input differs from
  the native input only by its optional final LF; all 510,404 phase bytes
  match. All 21 finish rejection cases pass. The run took 14,146.21 seconds;
  its longest Lean child took 1,220.85 seconds and its longest Python child
  took 23.50 seconds, within their 1,500- and 300-second caps. The comparison
  target is the newly generated native iteration-3 result. Direct complete
  current CPU3 comparisons were not completed and are now outside this goal.
  Receipts and exact counts are in
  `GOLDEN_CONFORMANCE_CHECKS.json`, under `independent_iteration_3_C`.
- Independent iteration-3 R and digit generation passed for both ranges.
  The R comparison with this continuation's native parent passed for all
  253,011,276 field coefficients and rejected a changed tail. At 06:36:55 UTC
  on 2026-09-22, the guard stopped the digit comparison before the checker
  started because other Lean processes were active. Its failed receipt and
  the four passed generation receipts remain intact. The same digit command
  then passed at 06:46:59 UTC in 3.02 seconds under its 300-second cap. All
  16 children, 4,685,394 blocks and 4,048,180,416 field coefficients match
  the continuation's native target; the changed tail was rejected. The
  D continuation then completed commitment batches 0–3 and pad batches 0–2.
  At 06:50:30 UTC, the guard stopped `d-pad-3` before launch because other
  Lean processes were active. The continuation exited with status 1; the
  remaining D work is incomplete. Paths and scope are in
  `GOLDEN_CONFORMANCE_CHECKS.json`, under `independent_iteration_3_reductions`.
- The queued continuation passed `d-pad-3-after-queue`. In total,
  commitment batches 0–15 and pad batches 0–14 passed within their
  1,500-second caps. At 07:19:31 UTC on 2026-09-22, the guard stopped
  `d-pad-15` before launch because another `lake` process was active. The
  queued driver exited with status 1. This was the third build-queue conflict,
  despite waiting before each child command; no mathematical mismatch was
  recorded. All previous failure receipts remain intact. At that point,
  execution stopped under the formal project's then-current three-round
  stop-and-report rule. The JSON record identifies the queued driver, its
  custody hash, completed batches and failed receipt.
- The owner clarified that build serialization applies per worktree. The
  earlier host-wide interpretation was wrong; other worktrees do not need
  to pause. Commit `3ddf6eb9` records this scope and changes the formal
  stop-and-report rule to five rounds. Commit `48da0896` scopes the guard's
  lock and process checks to the canonical worktree; all 13 focused tests
  pass. Caps and command restrictions are unchanged. This task now has a
  private Rust target: 5,532 regular files were checked, with no shared file
  inodes or symlinks into the original target. The checker bytes match, and
  its release build passed in 0.213 seconds under the 300-second cap.
  `private-rust-target.json` and `private-rust-build.json` in the run directory
  record these checks.
- `worktree-source-transition.json` sealed the source transition at
  07:40:10 UTC on 2026-09-22. Only the guard and its focused test differ from
  the continuation's saved source map; its original pin is unchanged.
  The first fold's 19 original files match current CPU step 1 in 89,980,008
  exact bytes. Its previously absent pin was then created normally. The
  audit records a new default-runtime baseline; the old continuation pin
  had no selected-runtime entry. Both source/input maps, runtime and private
  checker are checked at replay initialization. Six stub tests passed.
  The new driver started at 07:42:53 UTC. `d-pad-15-worktree` passed in
  30.946 seconds with the original command and 1,500-second cap. At the
  07:45:27 UTC snapshot, commitment and pad batches 0–16 had passed.
  D generation continued after that snapshot. All earlier counts and failures
  remain dated records. The JSON record links the audit, driver identity and
  new receipts.
- At 07:53:12 UTC on 2026-09-22, the owner selected 1→2→3 with the complete
  input connection. The owned third-fold guard was stopped with `SIGTERM`;
  its interrupted receipt is retained. This was an owner scope change, not
  a proof or data failure. Remaining 3→4 production and comparisons are no
  longer acceptance criteria. `third-fold-owner-stop.json` records the stop.
- All final-state-3 native terminal checks passed at 08:15:56 UTC on
  2026-09-22. The valid result was accepted. Changed CE K and A evaluations
  were rejected against the complete witness openings; changed fresh private
  data failed relation row 0. All seven child commands, including the three
  mutation preparations, passed within 300 seconds each. The longest child
  took 246.751 seconds; the coordinator took 1,129.177 seconds in total.
  `two-fold-terminal.json` and its log record this selected terminal predicate
  on the actual retained witnesses, with no new proof backend.
- The `selected_123.py` driver started independent first-fold generation at
  08:18:51 UTC on 2026-09-22. It owns this first-fold checkpoint; no separate
  completed first-fold receipt was reused. Full generation passed at
  14:02:17 UTC in 20,606.455 seconds. The selected driver finished with exit 0
  at 14:04:47 UTC. `independent-1-2-3-result.json` records the complete
  selected result; `first-second-input-bridge/result.json` records the input
  connection. Source, driver and bridge checks also passed at completion.
- Independent first-fold C generation and comparison passed at 11:12:28 UTC
  on 2026-09-22. All 28 rounds, 560 round field words and 28 transcript
  transitions matched the fresh native target. All 27,540 evaluation words,
  29,288 proof-input words, 15 phase fields and eight outgoing transcript
  words matched. The 271,058-byte Lean input differs from the native input
  only by its optional final LF; the 190,581 phase bytes match exactly.
  All 21 finish-input and output rejection cases passed. The 798 C child
  commands took 8,923.394 seconds in total; the longest took 598.218 seconds.
  Each child passed within its existing cap. Receipts are under
  `independent-first/step-1-to-2/logs`.
- Independent first-fold R/D passed at 13:43:10 UTC on 2026-09-22 against
  this replay's fresh native targets. All 253,011,276 parent coefficients
  and 4,048,180,416 digit coefficients matched, including zero entries and
  tails. The D result covers 19,008 commitment words and 25,920 evaluation
  words; 53 matrix ranges cover all 6,377,559 rows. Complete public C/R/D
  outputs and all 945,983 native NIFS proof bytes matched the Lean result.
  The normal native PiDEC wrapper also matched all 17 D result fields.
  The R and D comparison routines each rejected a changed final coefficient.
  Lean `checkPiDECInput from-replay` rejected all 30 selected input/output
  cases; native `pi_dec::verify` rejected all 55 selected PiDEC mutations.
  These 55 cases are not the full NIFS mutation suite. The separate native
  parent-bound check also passed. All 180 recorded child runs passed:
  175 Lean runs under 1,500 seconds each and five Rust/Python runs under
  300 seconds each. Their aggregate elapsed time was 9,040.659 seconds;
  the longest took 419.643 seconds. These totals exclude C and successor
  work. Receipts and rejection owners are in `first_fold_generation.R_D`
  in the adjacent JSON record.
- The independent first successor passed all 16 child runs in 1,145.590
  seconds in total. The longest, `canonical-rows`, took 495.883 seconds.
  All 253,011,276 assignment coefficients and 6,377,559 active rows passed;
  all eight row rejection cases and 26 physical, assignment and commitment
  rejection cases passed. Direct current CPU1 comparisons then passed for
  the complete R parent, NIFS result and proof, 940,444 caller bytes,
  234,755,400 physical bytes and all 16 child witnesses (444,373,866 bytes).
  The 96,258,853-byte fresh witness and 39,388-byte claim match the CPU
  results after the CPU files' final LF is removed. Their receipts and logs
  are `selected-123-first-current-{parent,nifs,outputs}` in the run directory.
- The complete first-to-second input connection passed at 14:03:46 UTC.
  All 17 sources cover 4,685,394 blocks each. Both the first-successor and
  original-input projections match the second fold's retained input:
  410,910 public bytes and 562,056,218 source bytes per projection. Checks
  also passed for the full four-word state and message, next request,
  carried R parent, all 17 caller-derived 32-byte frames, package, verifier
  context, caller links and retained C public fields 1, 2 and 6. Digests do
  not replace these value comparisons. The final driver checked and reused
  the completed state-3 terminal acceptance and three rejection receipts,
  then passed the current CPU/Lean handoff check. Iteration-2 C remains
  retained computation under its explicit source/input audit. The selected
  goal is complete; protected external acceptance and the broader 2→3→4
  obligation are not closed. CI and Metal remain outside the selected scope.

Large generated inputs and outputs stay outside Git. Completed commands,
source versions, exact comparison scope and failures are recorded here
or in adjacent JSON evidence. No completion is inferred from old receipts.
