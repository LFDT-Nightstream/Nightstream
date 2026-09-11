# F′ conformance fixes

Code commit: `2c63f41de09cfc47e6e4be4afd17e3e8a7726049`. Review date: 2026-09-09.

This change fixes the native PiRLC transcript endpoint and adds the missing local terminal-opening proof connections. It preserves the existing constraint leaves and assemblers. It does not claim full Stage 1 closure.

The selected profile remains Goldilocks, b = 2, k_rho = 16, B = 2^16, 28 PiCCS rounds and a joint domain at most 2^28, under the accepted owner decision. SuperNeo v1_1 remains the relation authority.

## Changes

| Boundary | Result and source |
|---|---|
| Native PiRLC endpoint | `crates/neo-fold-clean/src/paper/reductions/pi_rlc.rs` and `paper_exact_protocol.rs` derive projection metadata on a local transcript clone. Both leave the caller at the Lean sampler endpoint. The regression failed before the fix and passes after it. The real-fold fixture now uses canonical running inputs and the actual PiCCS outgoing transcript. |
| Arbitrary CCS opening | `formal/nightstream-fprime/NightstreamFPrime/Layout/ProductionRelation/AcceptedOpening.lean` derives selected logical rows and the actual public input from an arbitrary accepted full carrier. It does not assume canonical padding. |
| Selected context and complete next state | `formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualContextSecurity.lean`, `terminal_implies_matchingStepOrCollision`, derives the exact decoded Step and complete advertised terminal preimage, or the named state-hash collision. No row, public-equality, representation, encoder, context-equality or counter-bound premise is added. |
| Canonical terminal statement | `formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Stage1/Terminal.lean` checks the counter below the field modulus and fixed state widths, matching the existing native public boundary. Rejection theorems cover modulus-shifted counters and wrong state lengths. |
| Actual PiDEC parent opening | `formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualTerminalSecurity.lean`, `terminal_implies_parentOrBaseOrCollision`, derives the exact output match and consumes the terminal's sixteen actual child witnesses. The result is the decoded PiDEC parent opening, the base branch, or a named hash collision. Interior extraction remains separate. |
| Evidence and requirements map | `scripts/lean_graph/obligations.json` registers literal terminal-assignment and terminal-parent targets, required independent reviews, axiom checks and declaration exports. The original broader Stage 1 target remains open. The website map separates current local checks, reported historical evidence and production status. |

## Validation

All 91 candidate conformance diagnostics passed. The complete command logs, input manifest, source patch, new source files and review requests are retained outside the repository in `../nightstream-stage1-evidence/conformance-fixes-9e49e7fb` (path relative to the repository root). These are this task's local diagnostics. They are not approved-checker records or the earlier implementation agent's missing archive.

The checked source started at `9e49e7fbe03d61b0e16eec1d6f839d33d57af9bb`. Its pre-pin patch SHA-256 is `2abc1c405def03e774e8614d0ce15608de66443c67bdce0748d3ed7b74c8f921`; `reviewed-source.json` also names the three new Lean files and their exact hashes. `diagnostics/` records every selected command, completion check, outcome and log. Identity metadata identifies bytes; it is not proof authority.

The 91 checks cover exact physical A/B/C and fourteen logical matrices, independent base and recursive raw assignments, fresh and sixteen-child commitments, separate K and A0–A13 evaluations at both points, complete nonzero Lean/paper_exact/optimized results, exact caller and parent handoffs, and mutations. Base mutations cover 562 proof, 282 statement and 843 output cases. The recursive case also checks 56 point mutations. Every required child family is present.

Post-pin checks are recorded separately in `executions.json`. They confirm that the stored payload and verifier-owned pins use the checked candidate. Full logical-assignment mutation coverage remains open as described below.

| Post-pin check | Result | Elapsed seconds |
|---|---|---:|
| `package_loader` | 8 tests passed | 57.770 |
| `production_binding` | 3 tests passed | 111.225 |
| `pilot_lean_nonzero_parity` | 5 tests passed; explicit conformance cases also ran in the 91-check set | 45.026 |
| `nightstream_fprime_setup_parity` | Stored setup passed | 10.611 |
| `f_prime_package_production` | 3 tests passed, including valid row/column/coefficient mutations | 126.401 |
| Physical expanded matrices | Exact comparison passed | 56.671 |
| Logical expanded matrices | All 14 matrices and mutations passed | 182.255 |
| `per_application_matrix_program` | Complete traversal passed | 57.389 |
| `per_application_assignment` | Physical assignment passed; logical mutation coverage failed | 297.680 |
| Focused logical mutation retry | Reached the 300-second cap | 300.030 |
| Final logical mutation run | Completed; blocks 12, 13 and 14 are absent from canonical rows; mutation gate failed | 281.735 |
| `nifs_pi_ccs_lean_nonzero_parity` | 5 tests passed | 128.414 |
| `nifs_pi_rlc_lean_nonzero_parity` | 4 tests passed | 86.776 |
| `nifs_pi_dec_lean_nonzero_parity` | 3 tests passed | 71.730 |
| `validate.sh all` | Boundary, library and axiom gates passed | 428.631 |
| lean-graph Python suite | 81 tests passed | 6.822 |
| Requirements export | Build, 7 export/snapshot tests and JavaScript syntax passed | 0.129 / 0.183 / 0.099 |

Every Rust invocation used release mode with `RUSTC_WRAPPER` empty and a 300-second cap. Lean commands used `validate.sh` with a 1,500-second cap. Builds and tests ran one process at a time through the shared command guard; registered checkpoints apply their own guard. No proof backend ran.

Earlier failures are retained in the archive: stale stored width assertions, a matrix-mutation helper that selected a zero sentinel, an incorrect test target name, website test setup errors (wrong directory and missing build output), initial Lean proof errors, missing fixture output directory, and the bounded fixture-generation timeouts below. Corrections and their later results are separate records. None is erased or counted as a pass.

Completed before that run: the full Lean library, boundary and axiom gate passed in 118.098 seconds. Four PiRLC tests, seven projection tests and the carried-accumulator crosscheck passed after formatting. Fresh package, binding and expansion emission took 143.532, 229.326 and 14.287 seconds, and matched the selected input bytes exactly.

Both honest base and actual-child recursive PiCCS inputs are accepted by executable Lean. The shared optimized driver with a bounded evaluator and prepared openings reproduces both serialized inputs and proofs byte-for-byte. Full-profile production evaluation is a separate obligation.

The complete recursive child-family generator reached the 300-second cap twice. Both failures remain recorded. Measured work was divided into K, A0–A5 and A6–A13 jobs, which passed in 54.510, 171.181 and 166.217 seconds. Their completed outputs match those written before the timeouts. No bound, family or profile was reduced.

## Registered proof checkpoint

`checkpoint stage1-terminal-assignment` passed on saved snapshot
`7fa56d8616309016cf7e27a9f2426765dd72914b602feb68b3f50d17df30e5dd`.
It checked all five registered targets, including both new terminal targets.
The dependency-export gate took 1,262.189 seconds in total; its cold Lean build
took 881 seconds. The exact-target gate took 191.361 seconds in total and reused
the matching isolated build. Its build and acceptance commands took 9 and 25
seconds. These are checkpoint timings, not a comparison of equivalent builds.
Ordinary proof edits should continue to use focused incremental builds.

The retained `status` reports both gates as passed, current and diagnostic.
Its `stale` and `rejected` lists are empty for this saved Lean snapshot.
Both terminal obligations remain `closed: false`: approved-checker evidence,
independent target-meaning review and decomposition review are missing.
The original broad Stage 1 assignment target remains open. Checking both
literal terminal witnesses does not grant any phase status.

Pending review requests:

- Terminal assignment: `0bc783f51177f130731cd93cf35a1e7042ddf2ba7b3ca2aede5e9f0ef94ff23b`.
- Terminal parent: `248326508a2a1d701b297486778341c02db2aa46a36beaaf7772f30609f11d9e`.

## Scope still open

The new local targets have Lean proof checks; their independent meaning and decomposition reviews remain pending. No named phase is newly declared Compiler-closed, Conformance-closed or Production-closed by this change.

| Phase | Compiler-closed | Conformance-closed | Production-closed |
|---|---|---|---|
| PiCCS | No new closure claim; source and axiom checks recorded | Open: required reviews and complete mutation evidence | Open: full-profile evaluator and package-only lifecycle |
| PiRLC | No new closure claim; endpoint fix does not change rows | Open: must follow the exact conformance-closed PiCCS output | Open |
| PiDEC | No new closure claim; terminal parent theorem added | Open: must follow the exact conformance-closed PiRLC output | Open |

The new terminal predicate explicitly adds canonical counter and state-width checks. This is a target-meaning change that must be reviewed; a local proof does not approve it. The original broader `stage1-assignment` target remains unregistered/open. The two narrower terminal targets do not replace it.

Full history and quantitative NIFS/security composition, the full-profile production evaluator, and the package-only production lifecycle remain open. A proof backend still requires separate owner approval after the required conformance work.

The fixed package, setup, expansion, pilot, base fixture and ownership data now match the reproduced Lean output. Verifier-owned Rust identity constants and the two Lean production-context constants select those bytes. This alignment follows the 91 pre-pin checks and repeat emission. Historical prefix candidates remain separate; they are not an alternate production authority.

The retained `per_application_assignment` test passed physical row satisfaction and exact logical value/satisfaction checks, then failed its full mutation-coverage requirement: no effective coordinate mutation was found for assignment blocks 12, 13 and 14. The check is not weakened. This is an additional open logical mutation gate; the 91 diagnostic pass count does not erase it. The final focused run finished in 281.735 seconds and confirmed that all three blocks are absent from the canonical rows. They are `piCcsPayload`, `runningRoundC0` and `runningRoundC1` in `PerApplicationAssignmentPlan.BlockKind`. No row-satisfaction failure was reported. This finding does not by itself establish a false `StepHoldsFor` result. The mutation gate remains failed and the unused allocation remains visible. The preceding focused run hit the time cap. The final test removes a duplicate full row traversal after the complete vectors have already been compared; all value, independent-row and mutation requirements remain.

Full-profile production execution and approved phase closure must not be inferred from these results.

The cumulative package footprint is unchanged by these proof and endpoint fixes: 29,225,729 physical rows and 29,344,425 physical columns; 6,377,559 logical rows, 254,260,583 logical columns and a 254,260,620-coordinate carrier. The carrier has 37 padding coordinates and fits the selected 2^28 joint domain with 14,174,836 coordinates of headroom. The current theorem ledger is in `formal/nightstream-fprime/CONSTRAINT_TREE.md`. These dimensions alone are not conformance evidence.

The logical root already has eight children, including `NextPreimage`. The earlier seven-child finding is stale for this source. The phase and root assemblers are preserved. No new leaf wrappers, Rust features, environment settings or protocol hash family are introduced. The existing redundant sampler coordinates remain an open footprint item; this change does not hide them with a smaller reported width.

The requirements website source and export are updated locally and refer to the code commit above. The connected Sites account cannot edit the existing published project, so this report makes no live publication claim.

The primary worktree, frozen corpus, owner goal, architecture specification, AGENTS files and both external reviews remain unchanged. No proof backend was run. The user subsequently authorized commits and the three-branch merge sequence; the code commit is identified above. The final merge identifiers and repository status are recorded in the task handoff after the merges.

## Necessary next work

### Three-block classification, 2026-09-11

Source: `4fd19b00cf1e052ecbe9bd6923a3463ef186b36d`. The ranges below are
half-open logical-coordinate ranges. `PerApplicationAssignmentPlan` still
emits these blocks, but the selected consumers read the source forms below.

| Block | Allocation and range | Actual consumer and correspondence | Retained rejection evidence |
| --- | --- | --- | --- |
| `piCcsPayload` (12) | `PiCCSActionPayloadBlock.payloadStart` and `block.coordinateCount`: `[193995298,195242354)` | `DirectPiDECPrefixPlan.piCcsPayload` selects `PiCCSPayloadWiring.form`; `form_eval` and `form_eval_source` identify its values. | Archive records `diagnostics/piccs-{proof,statement,output}-mutations/0.json` pass for the authoritative fields. |
| `runningRoundC0` (13) | `RunningTransitionRetainedGeometry.roundC0Start` and `roundC0Block.coordinateCount`: `[195242354,195243502)` | `RunningTransitionDirectPlan.Location.form` selects `PiCCSTranscriptOutputForms.pointForm` at component 0; `pointForm_eq_outputState`, `pointForm_eval`, and `pointSource_c0` identify it. | `diagnostics/recursive-piccs-point-mutations/0.json` passes for the authoritative point components. |
| `runningRoundC1` (14) | `RunningTransitionRetainedGeometry.roundC1Start` and `roundC1Block.coordinateCount`: `[195243502,195244650)` | The same consumer selects `pointForm` at component 1, with `pointSource_c1`. | The same recursive point-mutation record covers this component. |

The archive's `pinned-logical-assignment-final.log` reports that all three
blocks are absent from the canonical rows. This agrees with the selected
consumer definitions. They are redundant retained copies; the inspected
consumers do not obtain protocol authority from them. The source review
found no missing binding at these consumers.

This closes the classification task only. The allocation-wide mutation gate
still fails. The retained mutation records are scoped diagnostics, not new
conformance approval. Removing the copies requires a separate checked
allocation change and new package pins. No constraint or test was changed,
and the known failed full scan was not rerun.

### Remaining implementation work

Resolve the unused retained blocks and the strict mutation target without adding
copy rows or weakening the required semantics. Any resulting relation-identity
change must repeat exact matrices, independent assignments, complete nonzero
parity and mutations before another pin update. Obtain the pending independent
target, decomposition, formula and branch reviews. Then follow the owner order
for PiCCS closure, PiRLC, PiDEC and the package-only production lifecycle.
Full history/security composition and the separate backend decision remain open.

The [evidence archive](conformance-fixes-evidence.zip) contains original command records, logs, serialized phase inputs/results, source patches, input identities and pending review requests. It is a review archive, not an accepted-checker store. Large opening carriers, library/build products and derived graph data remain in the external evidence store.
