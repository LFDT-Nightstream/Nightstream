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

The record `pidec-evaluation-replay` remains open for complete execution.
Its exact target, `LeanGraph.Targets.PiDECChildEvaluationReplay`, now passes
with the audited `PiDECEvaluationFromBlocks.familyFromBlocks_honestMessages`
consumer. The only premise is a successful checked split. The kernel computes
Pad and all 14 matrix families at `ParentValues.point`, with all 54 extension
coefficients per family. The graph records this target and its metadata gate.
No claimed evaluation, opening or cryptographic premise enters the kernel.

The shared block executor reads the canonical compact matrix program. Its
proof identifies the result with the existing `PaperAlgebra.evaluationFamily`
and `honestMessages`. Sparse rows retain repeated and cancelling entries;
omitted blocks and the selected padded suffix are proved zero. The former
unchecked family draft has been replaced by these checked modules.

For Pad, `PiDECPadWeightedProduct.products_value` proves the accelerated
weighted product equals the 54-row sum. `PiDECPadBlockRange` connects complete
block coverage to the full Pad accumulator. At actual parent block 0, every
accelerated child value matched the earlier row kernel. The weighted kernel
took 2.60 ms after initialization; the reference call took 112.35 ms including
initialization. This is not a steady-state speed comparison.

The complete Pad replay now passes. All 64 contiguous ranges cover every
one of the 4,685,394 carrier blocks, including the tails. Lean derives the
point from accepted C execution, splits the Lean parent blocks, computes each
weighted product, and adds all partial sums. Rust only decodes and compares.
All 1,728 Pad field words and all 56 point words match native children. A
change at child 15, lane 53, imaginary component rejects at that location.
The range commands took 1,759.75 s in total, with a largest run of 45.03 s and
peak RSS of 3,864,584 KiB. Final Lean read/sum/write took 35 ms. See
`PIDEC_PAD_REPLAY.json` and `PIDEC_PAD_RANGES.json` for the source-bound record.

All 14 matrix families remain open. The first sparse Poseidon row probe was
stopped after 248.19 s with observed process RSS of 29,122,284 KiB. The proved
numeric reader now evaluates the same first row's 14 ports in 4.20 ms. Its
equivalence preserves arbitrary reads, retained outputs, selector-scaled
constants and failed lookups. `kernel_eq_evalSparse` connects this scalar
interpretation to the existing child-coefficient kernel without added premises.

The next probe exposed full-package source scans in C ordinary rows. The
proved packet accessor now reads that block's first and last rows in 83.04 ms
and 183.83 ms. These are endpoint feasibility measurements with a column-index
read, not actual-witness replay. Ten endpoints in blocks 0 through 4 were
measured. The shared canonical source cache now preserves every lookup and
builds 1,412,568 rows in 1.37 s. Pilot endpoints pass through this cache.
The first actual Poseidon invocation computes rows [0,94) for all children,
matrices and lanes. The sparse reader output matches the dense Lean reader
byte for byte; total command time fell from 107.45 s to 74.27 s, including
57.84 s to load the complete parent. Peak RSS was 8,104,956 KiB.
`PiDECMatrixSelectedBatch.selectedInvocation_eq_range` identifies this
computation with the canonical weighted range. The integer parent reader
now matches the same bytes and lowers peak RSS to 3,451,752 KiB; its one-time
load is slower. The Phi81 loader now uses direct finite-index selection,
with total equality to the original loader. All remaining block endpoints
pass; the final Phi81 row lookup takes 1.80 ms.

The Poseidon schedule now shares each retained read between row and state
construction and reuses its final state for pins. The existing row theorem
is preserved. The actual 16-invocation result matches its prior bytes, while
the slowest child's arithmetic falls from 13.65 s to 6.83 s. The selected
range theorem covers every child and matrix. The sparse range theorem and
exact cache/grouped-load theorems cover the other opcodes. The executor now
accepts block-local row bounds, with complete 94-row Poseidon and 34-row
Phi81 groups required. Its actual first Phi81 group passes in 0.37 s of
arithmetic after loading the parent. These are partial ranges; complete
coverage and native matrix comparison remain open. See
`PIDEC_MATRIX_REPLAY.json`.

Sparse interface forms are now cached once per invocation, with equality
to the existing interface. The repeated 16-invocation result has unchanged
bytes. A further 1,600 invocations, rows `[1504,151904)`, pass in 646.16 s
at 4,596,052 KiB peak RSS. The Lean merger checks complete contiguous row
coverage and the C-derived point, sums every matrix coefficient, and joins
the complete Pad result in the existing native comparison format. Its
range-addition declarations are audited. Complete source-derived matrix
coverage and the final actual merged comparison remain open.

The next actual range, `[151904,302304)`, passes in 671.73 s. The complete
parent has measured magnitude 115. Lean now computes that maximum itself,
and a proved guard returns explicit zero rings for children above the
maximum's bit positions. All 16 children and every load guard remain.
The same 16-invocation output has unchanged bytes. Its maximum scan takes
0.58 s; child task times include scheduling, and parent-load times vary.
Arbitrary adjacent range addition and the full split-block identity now
have focused checked declarations in `PiDECMatrixMergeClosure`.

Keep the full-plan proof imports outside the replay executable. An import
through `PiDECMatrixZeroRead` initialized a reference plan before C execution;
the stopped log is retained. The executable now imports only the magnitude
and guard module, while library/audit imports retain the equality proofs.

`PIDEC_MATRIX_RANGES.json` is the compact index of actual disjoint results and
remaining row gaps. It excludes historical duplicate and synthetic outputs.
The next large Poseidon range passes in 488.61 s. The pilot ordinary block
and first C ordinary range also pass. Worker-derived slices now preserve the
same output bytes for both kernel kinds. Their maximum child spans fall from
5.20 to 2.76 s (Poseidon) and 9.99 to 7.04 s (C ordinary). Parent loading
remains variable; these task spans do not measure complete command speed.

The first Poseidon block is now complete: all 12,350 invocations and
1,160,900 rows have actual range results. Its final two equal ranges used
3,767 invocations each, sized from measured rates under the 1,500 s cap.
The range index records exact contiguous coverage; other matrix blocks and
the complete native comparison remain open.

The second Poseidon block is also complete, covering all 12,350 invocations
and global rows [1,160,900,2,321,800). Its final two ranges passed in 833.78
and 850.77 s. All three ranges are retained with complete values and source
records; remaining matrix blocks and the complete native comparison are open.

The third large Poseidon block is complete: all 7,604 invocations and
714,776 rows have saved results. Its two ranges passed in 917.98 and
773.69 s. The first three blocks now cover global rows [0,3,036,576).

The C pin block and all 811,669 ordinary C verifier rows have complete
matrix range results. The final Eval_A, CCS/norm and final-identity ranges
passed in 620.48 s, 158.01 s and 706.51 s. The contiguous saved prefix is now
[0,3,864,783). These are PiDEC child evaluations of verifier rows; PiCCS
prover replay and the complete native matrix comparison stay open.

The R sampler blocks are complete, including all 220,881 ordinary rows for
17 sources. The first source passed in 141.02 s; the remaining 16 sources
passed together in 491.60 s with peak RSS 5,480,796 KiB. The saved prefix,
including the earlier first product group, now reaches row 4,100,120.
Product and later rows remain before the complete matrix comparison.

All 686,664 commitment-combination rows in the PiRLC product block have
complete PiDEC matrix results. The final seven-source and eight-source
ranges passed in 1,190.92 s and 1,274.49 s, with peak RSS 10,330,588 KiB.
The saved prefix now reaches row 4,786,750. Public-input and evaluation
products, First54 grids and later blocks remain before the full comparison.

Public-input products and all Eval_K products now have complete matrix
results. The two-cell measurement also covers the first two Eval_A sources,
including all 14 blocks and both source rules. The combined range passed
in 634.41 s. Saved coverage reaches row 5,108,050; the remaining 15 Eval_A
sources, First54 grids and later rows remain before the full comparison.

The next pure Eval_A range, 411,264 rows, reached the 1,500 s cap and
added no coverage. `PIDEC_EVALUATION_PERFORMANCE.json` records that failure
and the scalar update optimization, whose existing specification theorem
passes unchanged. All bytes match on the actual 7,344-row comparison;
unprofiled command time changed from 120.79 to 120.23 s. This is a small
gain. The profile points to sparse-row evaluation as the main cost.
The Lean runtime fork is reviewed but not benchmarked or installed.
Re-measure pure Eval_A before selecting another large range.

The complete family target and metadata gate remain in the graph. Kernel
closure and the successful Pad result do not close the matrix execution
requirement. Full comparison must cover all 25,920 evaluation field words,
the common point and a changed target. Native targets are retained in
`PIDEC_EVALUATION_TARGETS.json`; no native producer rerun is required. Earlier
proof and first-range checkpoints remain in `PIDEC_EVALUATION_FAMILY.json`,
`PIDEC_EVALUATION_PILOT.json` and `PIDEC_EVALUATION_ROWS.json`.

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
