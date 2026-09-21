# Golden conformance CI

The `Golden conformance` workflow selects checks with
`golden_conformance_changes.py`. Its final `Golden conformance` job fails if
a selected check fails or does not run. Repository branch rules must require
that check; a workflow file does not set branch rules.

The CPU job downloads and restores the five recorded archives, builds the
current executables, runs native folds 1–3, and runs fresh Lean verifier,
caller, physical-witness and mutation checks for all three folds. Each native
command keeps the 300-second cap and each Lean command keeps the 1,500-second
cap. Native phases also enforce the documented 16 GiB RSS guard.

When Lean or reference inputs change, a separate job invokes the exact
independent first-fold generator and replay for iterations 2→3→4. It compares the original witnesses,
proofs, callers, physical assignments, fresh witnesses, commitments and child
witnesses with the current CPU outputs. The replay uses its own Lean successor
for its next Lean input. This wiring is not evidence that generation has run.

Retained C stages alone took 20,684.78 seconds. The independent job uses the
documented self-hosted platform maximum of 5 days (7,200 minutes), because the
default 360-minute job is insufficient for the selected complete sequence.
The per-command project caps still apply. All authenticated downloads occur
before computation. The final receipt is printed in the job log; the job does
not depend on an artifact upload after `GITHUB_TOKEN` expires at 24 hours.
Complete first-fold and recursive-run timings remain to be measured.
See [job timeouts](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idtimeout-minutes)
and [self-hosted limits](https://docs.github.com/en/actions/reference/limits).

Metal consumes the exact CPU files and fresh Lean input snapshots from the CPU
job. It checks their content before and after the engine comparison. It does
not repeat Lean generation. A missing Metal device fails the producer.

CPU and independent jobs use `[self-hosted, linux]`; Metal uses
`[self-hosted, macOS]`. These are GitHub's default OS labels, and each assigned
runner must match all listed labels. The independent replay needs Linux
because its coordinator uses GNU `/usr/bin/time -v`.
See [default runner labels](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow#using-default-labels-to-route-jobs).

Actual runner registration and repository access remain unverified. The OS
selectors do not establish available resources or a working Metal device.
Those require a registered runner and successful execution of the checks.
Archive publication also remains an operational requirement.
The older envelope, child and later-fold archive tags returned HTTP 404 in
the public release check; downloads fail until those exact assets are available.
The independent replay release is available. Missing archives cannot count as
a successful reproduction.
