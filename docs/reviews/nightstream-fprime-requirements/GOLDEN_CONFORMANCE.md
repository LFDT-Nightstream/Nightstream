# Selected golden-vector conformance

Status: revalidation of the rebased PR is in progress. The completed run at
`74508b1fc63d642b8570da957d2fbae0a618b6de` does not certify the rebased CPU code.

The selected scope is CPU–Lean **1→2→3**, for the Nightstream Goldilocks
profile `b = 2`, `k_rho = 16`, `B = 65536`. It includes the first fold, one
recursive fold, their complete input connection and terminal checks at
state 3. CI enforcement, Metal validation and a third independent fold are
outside this goal.

The owner's “9/10” target is a qualitative engineering judgment, not a
measured score, probability or cryptographic security bound. Report the
acceptance criteria and executed evidence below instead of assigning a
numerical result. Selected examples do not prove universal Rust correctness
or discharge the external Fiat–Shamir assumption.

## Acceptance criteria

- Restore the five recorded archives into a clean directory.
- Run the current CPU producer for both selected folds within the existing
  per-command time and RSS limits.
- Check fresh CPU C/R/D outputs with Lean, including rejection classes
  recorded separately for each verifier or decoder.
- Compare complete proof bytes, parent and child witnesses, caller values,
  physical witnesses and fresh outputs with independent Lean results.
- Connect all 17 first-successor source witnesses and public values to the
  second fold's consumed inputs. Also compare package, context, state,
  message, carried parent, all 17 digest frames and C/R/D transcript links.
- Check terminal acceptance at state 3 and rejection of `ce-evaluation`,
  `ce-matrix-evaluation` and `fresh-private` mutations.
- Keep reproducible repository commands and source/input records. Missing
  inputs or failed checks prevent a successful result. Digests identify
  retained files; actual value comparisons establish equality.

## Reproduction

The commands and required inputs are in
[scripts/GOLDEN_CONFORMANCE.md](../../../scripts/GOLDEN_CONFORMANCE.md).
`golden_conformance_ci.py cpu` runs current CPU generation and fresh Lean
verification. `independent` generates the two Lean folds and checks their
connection. `check_selected_replay.py` permits reuse only after the explicit
source/input audit and complete comparisons with a fresh CPU run.

The selected entrypoints stop at state 3. The separate registered
`fresh-recursive-loop` obligation still covers 2→3→4 and remains open; it is
not an acceptance criterion for this goal. Local archive restoration does
not claim protected external acceptance or publication of new release assets.

## Historical evidence

[GOLDEN_CONFORMANCE_CHECKS.json](GOLDEN_CONFORMANCE_CHECKS.json) retains the
original receipts, timeline, paths and scope. That run completed at
2026-09-22 14:04:47 UTC. It established the selected two-fold connection,
complete CPU comparisons and state-3 terminal checks for the earlier source
snapshot. Its extra CPU fold and stopped independent third fold are historical
work, not requirements for reproduction.

Iteration-2 C reused earlier independent computation under the recorded
source/input audit. First-fold C/R/D and the second-fold R/D and successor
were independently generated. Lean verifier checks that consume Rust proof
messages remain separate evidence from independent generation.
