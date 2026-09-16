# Current Stage 1 baseline closure

## Active goal: independent prover replay

The owner changed the active target on September 13: independently execute
the selected prover computations in Lean and Rust from the same setup,
public inputs and original witnesses. Rust results enter only comparison
checks. Preserve SuperNeo v1.2 and the selected Nightstream Goldilocks
profile (`b = 2`, `k_rho = 16`) and package. Finish and
measure **PiRLC first** before integrating PiDEC, PiCCS or HyperNova replay.
PiCCS starts with a measured first round on the actual witness when that
milestone begins. Full independence requires the composed phases.

The owner also requires a SuperNeo + HyperNova run with Nebula inactive.
Use the selected `Poseidon2HashChainV1Package` and plain commitments:
fresh, running and parent claims carry `adv = None`. The selected NIFS
resources contain no `LaneScheme`. Reject unexpected auxiliary commitments;
preserve the complete HyperNova state, transcript, witnesses and tails.
Nebula code being compiled into the crate is not Nebula relation execution.
The regression `selected_plain_step_rejects_auxiliary_commitments` passes:
the unchanged plain input is accepted, and zero/nonzero auxiliary tuples
on fresh, running and supplied-parent claims reject. See
`SELECTED_PLAIN_REPLAY.json` for the source hashes and bounded test command.

The local PiRLC checkpoint now passes: all 253,011,276 carrier coefficients
match the actual Rust parent, and a change to the last tail coefficient
rejects in its range. The prepared kernel and selected `honestResponse`
bridge pass the axiom audit and exact target check. Static/build/axioms and
the complete Rust test build pass. `PIRLC_WITNESS_REPLAY.json` in the
requirements review directory records the source hashes, commands, times
and memory. Release upload and fresh-download verification remain external.
PiDEC uses this Lean-computed parent.

PiDEC has two required results: every private digit, then every child
commitment and evaluation computed from those digits. The private target is
`LeanGraph.Targets.PiDECWitnessReplay`: total equality with the existing
scalar split, including rejection outside the strict bound and successful
output for every bounded input. The selected honest-witness bridge fixes
the message consumer. Preserve parent provenance from the PiRLC replay;
native children enter only the comparison. All 16 children and complete
carrier tails must match. Check signed boundary values and changed-target
rejection. Private digit agreement alone does not close the message result.

The local private PiDEC checkpoint passes: all 4,048,180,416 child
coefficients match freshly generated Rust children, and the final-tail
mutation rejects. The total kernel, selected consumer, exact target, audits,
boundary tests and Rust test build pass. See `PIDEC_WITNESS_REPLAY.json`.
Lean's two private ranges took 8.38 and 142.15 seconds. The child message
calculation uses these same digits. No PiCCS or HyperNova replay is claimed
by this private digit result.

The complete local PiDEC message replay now passes. Lean computes all
19,008 commitment field words from the same private digits. It also computes
all 1,728 Pad words and all 24,192 matrix-evaluation words. The 53 disjoint
matrix ranges cover every selected row in [0,6,377,559); Lean checks the
complete range coverage and adds the results. All 25,920 evaluation words
and all 56 common-point words match Rust. The final changed matrix
coefficient rejects at child 15, matrix 13, lane 53, imaginary component.
The earlier complete child comparison also checks all 4,320 verifier-derived
public words on this same trace. `PIDEC_COMPLETE_REPLAY.json` records the
sources, proof links, commands, coverage and scope. The range reports retain
the earlier measurements and compiler fixes.

The exact targets remain `LeanGraph.Targets.PiDECCommitmentReplay` and
`LeanGraph.Targets.PiDECChildEvaluationReplay`, with successful checked split
as the kernel premise. The public projection link retains its explicit
parent-public-opening premise. No claimed Rust commitment or evaluation
constructs the Lean expected values. The point remains conditional on
Lean-verified Rust PiCCS messages until independent PiCCS replay is complete.

The small `pidec-evaluation-comparison` gate rechecks the saved complete
values and target mutation. It does not establish source-generation
provenance, package identity, or full proof encoding. The original parent,
53 source-bound row runs and completed Pad supply the separate local
calculation evidence. Protected full-generation acceptance and release
upload/fresh-download verification remain external. Status stays
`Compiler-closed`.

The active computation milestone is PiCCS round one from the original
17 witnesses and public inputs. The exact graph target is
`LeanGraph.Targets.PiCCSFirstRoundReplayKernel`. It combines the complete
completion-sum formula, original and aggregated source constructors,
prepared coefficient equality, stored invocation rows, and total norm/selector
cache equality, and the complete prepared norm scan.
`piccs-first-round-kernel` checks its audit, literal target and
dependency graph. The producer input
contains no Rust rounds or newly claimed evaluations.

The first adjacent pair is measured. Reference polynomial construction took
60.04 seconds; shared gamma powers reduced it to 0.0247 seconds. All ten K
coefficients (20 field words) remain byte equal. Original-source images take
about 3.74 seconds per pair on that reference path. The retained source decoder also preserves the
complete prior PiRLC prefix. Six input rejection cases pass.
`PICCS_FIRST_ROUND_REPLAY.json` records the exact scope and source hashes.

Source/output weights now move before matrix evaluation. Reusing one complete
94-row invocation reduced the 47-pair run from 199.1 to 23.5 seconds, including
input loading. All 20 field words match the reference. Starting inside the
invocation and crossing its boundary also matches. A changed target coefficient
is rejected. The endpoint proof includes all original source lanes and both
canonical carried sums; stored-row equality retains the loader-success premise.

Weighted original blocks now share each requested read within an invocation.
The cache theorem covers repeated and missing keys. Prepared norm cubics and
suffix-selector weights also have complete coefficient equalities. Arbitrary
extension-field values use the original norm constructor on a cache miss.
The existing graph target includes the exact combined cached pair expression.

The norm contribution is now closed for the complete recorded input. The
prepared worker sum equals the original norm sum over all 2^27 pairs,
including the proved zero suffix beyond the complete carrier. The four inner
K coefficients match the actual Rust CPU norm function on all 17 original
sources; a changed coefficient rejects. The measured scan fell from 139.93
to 35.17 seconds, with 1,360,840 KiB peak RSS. `PICCS_NORM_REPLAY.json`
records inputs, exact scope and the checks. It is not full-Q or transcript
comparison.

The fresh contribution is also calculated across all 3,188,780 active pairs,
including row 6,377,558 paired with the first padded row. Exact coefficient
proofs skip zero monomials and exponent-zero factors. The selected positive
degree theorem closes the entire remaining Boolean suffix. The Option-preserving
source constructor is proved equal to the original-source reference sum; its
IO task and mutable cache loops remain implementation links. The complete run
took 623.73 seconds and 2,459,352 KiB peak RSS. `PICCS_FRESH_REPLAY.json`
records its exact scope and source hashes. This is not a full Rust Q comparison.

The complete first-round execution now matches Rust: all ten Q coefficients,
alpha, gamma, the challenge, both transcript states and the claims. A changed
coefficient rejects. `PICCS_FIRST_ROUND_REPLAY.json` binds the complete fresh,
norm and carried contributions to the original sources. The selected
source-to-whole-polynomial theorem remains an unchecked review patch;
its eleventh check still requires the requested owner approval.

Later rounds retain the existing `PrefixFold` authority. The generic
`PiCCSPrefixRound.roundPolynomial_evaluate` identifies the pair sum after any
challenge prefix with the original completion-sum specification. It does not
supply the stored endpoint arrays. `PiCCSSignedFirstFold.foldOne_prepare`
proves the signed-input cache equal to the original fold, including odd tails.
`PICCS_PREFIX_REPLAY.json` records the measured original-input prefix checks.
It also records the complete second-round inner norm: all 63,252,819 groups
and 17 sources match Rust's production fold and norm functions. A changed
coefficient rejects. The cached kernel is proved equal to the original
interpolation followed by the norm cubic; equal non-signed endpoints are
retained. The full Lean scan took 376.15 seconds at 1,362,264 KiB peak RSS.
The fresh and carried contributions still need to be composed for that round.
Remaining work is the complete later-round execution, final evaluations,
HyperNova and complete proof encoding. Generic
`program.row?` row caching was measured and removed: it expands a slow reference
path. Future sharing must retain the numeric invocation evaluator.

For PiRLC, Lean checks the supplied PiCCS proof and derives all mixing
challenges. The PiCCS messages and final claims are Rust inputs until the
PiCCS prover replay is complete. The 17 source witnesses are original inputs;
the Rust combined witness is only a target. Compare every coefficient of
the complete carrier, including tails, and reject a changed target
coefficient with its block and lane. Missing source blocks mean exact zero,
not omitted comparison coverage.

Each plain executable kernel must have an audited equality with the
existing specification. No new `native_decide`, `implemented_by`, `extern`
or assumed comparison result can supply that equality. PiRLC uses
`PiRLCFinite.combineAssignments`, connected to the honest response, and
the existing ring action; later phases use their existing prover semantics.

Register each milestone's exact target, premises, dependencies, gates and
open requirements in `obligations.json`. A kernel proof does not close its
execution gate. Bind inputs and targets by path, SHA-256, producer, source
commit and regeneration command. Large inputs must be retrievable through
the selected GitHub release assets before the result is called reproducible.
Record time and peak memory per invocation. Derive chunk sizes from a
measured run; the Lean cap is 1,500 seconds and the native cap is 300 seconds.
Retain one build queue, the owner's ten-attempt rule, this checkout and
`nico/f-prime-constraints-cuda-formal` only. Replay gives evidence on tested
inputs; it does not prove universal Rust correctness.

## Earlier baseline and retained evidence

Owner: the user's September 13 baseline goal. Work only on
`nico/f-prime-constraints-cuda-formal`, starting from `f714497c`.

The final result is the existing selected Goldilocks lifecycle, with
`b = 2`, `k_rho = 16`, its current package, and the approved FS/MSIS
boundaries. The public flow is `Stage1Envelope::initial(z0)`,
`package.extend(envelope, message)`, then
`package.verify(&expected_state, &envelope)`.

## Exact obligations

| Obligation | Required result | Closing evidence |
| --- | --- | --- |
| `stage1-assignment` | Arbitrary selected rows and their actual public input imply the full typed step at the decoded context. | Literal `LeanGraph.Targets.Stage1Assignment`, witnessed by `stage1Assignment` through `ActualPiDECOutput.selectedRowsAndPublic_imply_step`; exact acceptance and axiom audit. |
| Existing terminal and security targets | Bind the decoded step to the verifier-owned context or a named collision; retain the checked linear history bound. | `Stage1TerminalAssignment`, `Stage1TerminalParent`, and `HyperNovaLinearSecurity`, with their existing complete premises and required gates. |
| Public lifecycle | Construct the base assignment from actual state/advice and extend an active envelope through the existing native NIFS, caller packet and witness completion. | Public API execution, full Lean packet/assignment comparison, exact openings and terminal acceptance, with sampler/counter failures explicit. |
| Nonzero running input | Execute a further fold with actual nonzero running witnesses, then construct and verify its successor. | Complete C/R/D values and bytes, transcript/public/counter equality, exact next assignment and required mutation rejection against independent Lean checks. |
| Baseline delivery | Correct stale records and retain exact evidence for every technical link in scope. | Signed checkpoint, ordered static/build/axiom gates, affected identity/conformance checks, independent review and current requirements links. |

`stage1-baseline` tracks the combined result in the existing obligation map.
Its open requirements remain until their actual tests and review exist.
This registration does not change lean-graph's schema or acceptance rules.

The symbolic terminal false-acceptance target and six-record reconciliation
passed at `8084c256` and `5222c1d5`. The current staged nonzero C/R/D result,
complete proof bytes and mutations pass; see `NONZERO_NIFS_GATES.json` and
`NONZERO_NIFS_REVIEW.json` in `docs/reviews/nightstream-fprime-requirements`.
The complete later assignment and terminal checks also pass; see
`NONZERO_SUCCESSOR_GATES.json` and `NONZERO_SUCCESSOR_REVIEW.json`.
The aggregate stays compiler-closed while the two public active-call checks
await their specific execution allowance. Evidence delivery and external
production approval remain separate.

## Checkpoint requirements

The checkpoint must compile every `neo-fold-clean` test target with
`cargo test -p neo-fold-clean --release --no-run`, including the fixture
binary's test harness. Integration tests must stay in their integration
target; removing a failing harness is not the repair.

Commit reports and SHA-256 manifests. Store new evidence archives outside
Git; do not put generated witnesses or graph metadata in Git inside archives.
Existing committed archives remain historical evidence until verified
external copies and replacement references exist. Do not rewrite history.
Each named gate must have a documented fresh-checkout command and available
inputs. Small test inputs belong in the test fixtures directory; larger
external inputs need a retrieval location and checked manifest before the
gate is called reproducible. The owner selected GitHub release assets in
this repository on September 13. Record the release, asset URL and hash.

Before another expensive recursive run, record its measured stage costs,
memory measurements and the basis of its feasibility estimate. The existing
300-second native cap applies. An invocation that reaches it is a failed
slice. Report the stage and measurement; do not repeat an unchanged run or
silently convert failure into an ignored pass. The owner chose to keep the
existing project limits and report measured memory on September 13; the
review's proposed 64 GB ceiling is not adopted.

Only the coordinator runs Lean, Cargo and validation commands. Subagents
may read, draft and review. Reused Ironwood code must retain its verified
upstream license, authors and exact source commit. The reviewed snapshot is
Apache-2.0 OR MIT, Copyright (c) 2026 Zcash Protocol Developers, commit
`22dfee003b639eff660f68ea69a98a00409a9cb1`; preserve the notices described in
`external/ironwood/PROVENANCE.md`. A review citation is not code reuse.
Publish the requirements map only from committed inputs.

The public flow remains `Stage1Envelope::initial`, `package.extend` and
`package.verify`. Add no redundant initialization wrapper or public state
machine interface. Remove obsolete square-root assurance consumers only
after their uses and map references move to validated linear consumers.
General lemmas still required by those consumers are not obsolete.

## Named map dispositions

These are closure decisions and required evidence, not premature status changes.

| Record | Disposition | Required result or condition |
| --- | --- | --- |
| `N.security.error_budget` | Close the symbolic selected-terminal bound; numerical deployment choices are out of scope. | Connect terminal acceptance with no valid application history to the existing first-failure and linear visited bounds. Keep depth, queries, `g`, FS loss, marked hash collisions and actual adaptive MSIS advantage explicit. Do not substitute an extraction-success bound for false acceptance. |
| `L.language.expressions` | Close as a definition. | `Circuit.Basic.Expr` and `Env` supply the required evaluation semantics. Its `definition` status is accurate; no additional theorem is required by this record. |
| `L.language.contract` | Close as a definition. | `FormalCircuit` requires specification, footprint, soundness and completeness fields. Concrete production instances and their compiler gates remain the separate implementation evidence. |
| `L.encoding.actual` | Close the stale technical link. | `ActualPiDECOutput.selectedRowsAndPublic_imply_step`, the literal `Stage1Assignment` target and the terminal context/collision target cover arbitrary assignments. |
| `P.binding.context` | Close the selected technical link; production approval remains external. | Canonical descriptor plus `ActualContextSecurity.terminal_implies_matchingStepOrCollision`, exact assignment target and package-owned Rust context checks. |
| `P.delivery.terminal` | Close the selected terminal implementation; approved backend delivery remains external. | Selected `stage1::verify`, the typed Lean terminal target and reproducible acceptance/rejection checks on actual openings. Remove the stale citation to the generic verifier. |

## Premises and limits

The assignment target uses the actual public projection, a four-word digest,
and all selected rows. It assumes no honest encoder, NIFS acceptance,
sampler success or child-output match. Its context is decoded; selected
verifier authority remains the separate terminal/hash contract.

Reuse the approved cryptographic premises, mathematical invertibility,
declared clock and query conditions, conditional sampler success, and
canonical counter bounds. No new security model or numerical budget is
selected. The checked staged trace now reaches iteration 3 from actual
nonzero running witnesses. Capped stages supply their exact evidence and
terminal acceptance; they are not an uninterrupted public `extend` run.

One active obligation and one Lean/Rust build queue apply. The standing
owner override is ten rounds per obligation for this session; previous
attempts count. Native invocations retain the 300-second cap and Lean
commands the 1,500-second cap. Do not repeat an unchanged failed run.

This work excludes constraint reduction, new profiles, protocol changes,
new proof backends, Stage 2 and general cryptographic research. Required
correctness repairs must retain the appropriate preservation and package
checks. External production approval remains separate from local evidence.
