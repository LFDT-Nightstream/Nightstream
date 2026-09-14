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

The commitment kernels now use proved native-word ChaCha and the existing
proved native-word field operations. Both measured block products match the
earlier specification-backed outputs. First-use key time includes package
dimension initialization; it must not be charged to every block.
`PIDEC_COMMITMENT_PILOT.json` records the measurements. The complete first
range, `0..74272`, finished in 155.69 seconds at 1,511,560 KiB peak memory.
Use that measured extent for the remaining ranges. The input partitioner
copies the original Lean parent lines and records their hashes; missing
blocks remain exact zero.

`LeanGraph.Targets.PiDECCommitmentReplay` names the final commitment target.
Its only premise is the successful checked split of the supplied parent.
The proved key, product and finite sum yield every child commitment in
`honestMessages`. The new graph gates require all 64 contiguous ranges,
all 16 by 22 by 54 commitment coefficients, and changed-target rejection.
All 64 ranges now pass, covering every one of the 4,685,394 carrier blocks.
Lean also computes the final sum; Rust only decodes and compares the complete
values. All 19,008 coefficients match, and a changed coefficient at child 15,
row 21, lane 53 rejects. Missing ranges, gaps, wrong row counts and noncanonical
field values reject before Lean emits a result. The ranges took 5,982.10 s in
total; the largest took 160.41 s at a maximum 1,512,080 KiB peak RSS. The final
Lean read/sum/write took 376 ms. `PIDEC_COMMITMENT_REPLAY.json` and
`PIDEC_COMMITMENT_RANGES.json` record the full evidence. Complete child
evaluations are next; PiCCS and HyperNova replay remain later milestones.

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
