# Selected golden-vector conformance

Status: passed for the selected 1→2→3 cases on the rebased implementation.
[GOLDEN_CONFORMANCE_REVALIDATION.json](GOLDEN_CONFORMANCE_REVALIDATION.json)
records the current run, source audit and update to base `c92d6e148`. The older
run at `74508b1fc` remains a separate historical record.

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

## Current validation

- 72 focused Python tests pass, including checks with a symlinked temporary
  directory. Release builds and Rust formatting pass.
- All 18 selected CPU stages pass. The longest took 189.48 seconds; peak
  RSS was 12,444,114,944 bytes (11.59 GiB), below the 16 GiB guard.
- The first terminal rejection test exposed an old error-order expectation.
  The combined row evaluator checks the fresh relation before running
  openings. Both test assertions now require that specific relation error.
  The repeated rejection passed; the failed receipt remains in the archive.
  The other 17 passed stages were retained with complete byte comparisons.
  Production Rust, Lean and build inputs did not change during this repair.
  That mutation covers `FreshRelation`. The state-3 `ce-evaluation` and
  `ce-matrix-evaluation` cases below use the legacy verifier; they do not
  establish production running-opening rejection coverage.
- Both fresh Lean checks pass: all proof bytes, 177,326 private caller words,
  278 public caller words, all seven result fields and 234,755,400 physical
  bytes per fold. Each records 34 Lean public-check rejections, ten decoder
  rejections, one C rejection, two internal checks and 55 native D rejections.
- Both retained independent results match the current CPU proof, parent,
  children, caller, physical witness and fresh outputs. All 17 first-successor
  sources connect to the second fold's consumed inputs. The source audit
  permits only the comparison-crate rename; independent generation was reused.
- Fresh state-3 terminal acceptance and the three specified rejection cases
  pass. All 20 comparison stages pass; the longest took 247.13 seconds,
  within the 300-second native/Python cap.
- The subsequent base merge changes none of the validated Rust/Lean code,
  selected artifacts, build inputs or replay coordinators. Its source diff
  check is recorded with the results.

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
