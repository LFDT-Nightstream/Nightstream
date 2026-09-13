# Stage 1 baseline closure

Active owner goal: finish and freeze the current implementation, starting
from `f714497c` on `nico/f-prime-constraints-cuda-formal`. The full scope is
registered in `scripts/lean_graph/STAGE1_BASELINE.md` and the existing
lean-graph obligation `stage1-baseline`. This goal remains active.

## Audit checkpoint

Independent audits cover all 73 transcript/compiler/export/runtime records
in groups T, L and P, the selected public API, and the actual next recursive
input. The retained package and witness files match the cited manifests.
The audit distinguishes stale descriptions, mathematical assumptions,
unexecuted paths and external approval; none is silently treated as another.

| Claim | Current evidence | Remaining work |
| --- | --- | --- |
| Arbitrary selected assignment | `ActualPiDECOutput.selectedRowsAndPublic_imply_step` gives the full decoded-context step. The new literal `LeanGraph.Targets.Stage1Assignment` and its closure pass the build and exact acceptance check. | Final graph snapshot/metadata and map reconciliation. |
| Transcript, decoder and emitted package records | The T/L open model links and several P loader, pin, conversion and terminal descriptions are stale. Exact declarations and preserved artifact hashes are listed in the audit. | Correct these entries at the completed baseline evidence update. |
| Public lifecycle | `package.extend(envelope, message)` now connects private base construction and the active native producer. The public base call passes a full Lean-carrier and reference-commitment comparison, then terminal verification. | Execute the successful active call and cache recovery with actual nonzero running witnesses. The base test does not cover that path. |
| Nonzero running execution | The saved iteration-2 envelope has six nonzero running witnesses; its retained inputs match the earlier manifests. | The current stage loader always creates base inputs. Add the actual later-input route, execute C/R/D, and compare the complete later caller assignment and terminal output. |
| Selected public authority | Stage 1 uses the package-owned relation. Legacy public routes require the scope check recorded in the runtime audit. | Resolve applicable Stage 1 authority paths without starting Stage 2. |
| Production approval and execution limits | Local proof and conformance records exist. The uninterrupted full producer previously exceeded 300 seconds. | Keep these conditions explicit; staged results do not claim an uninterrupted execution or external approval. |

The assignment target has only the existing application fit/key, actual
public projection, four-word digest and selected-row premises. It adds no
encoder, sampler-success, NIFS-acceptance or child-output assumption.
Verifier-owned context remains the separate terminal/hash contract.

Its first focused build passes in 2.72 seconds and exact acceptance in
2.02 seconds. The existing graph suite passes all 82 tests after replacing
its obsolete assertion that the target must be absent. Independent review
passes after restoring three records accidentally omitted by the first
registry edit and requiring the new metadata marker. The records and review
scopes are checked against the preceding committed map; none was removed.

The final ordered gates pass: static 8.93 seconds, library 0.92 seconds
(3,920 jobs), and axioms 0.87 seconds (4,008 jobs). The requirements-site
lookup returns `project_not_found` for its retained project ID; publication
remains open. No replacement site or branch was created.

The full audit and initial findings are retained in
`STAGE1_BASELINE_AUDIT.zip`. The next active implementation obligation is
the public lifecycle. The approved profile and cryptographic boundaries,
one build queue, ten-round owner override and command caps remain in force.

## Public base checkpoint

The public base test passes in 199.45 seconds. It calls
`Stage1Envelope::initial`, `package.extend` and `package.verify`, compares
every logical carrier coordinate with the independent Lean base fixture,
checks the 45 zero tail coordinates and the scalar-reference commitment,
and checks counter rejection. Dummy base NIFS public digits remain separate
from the retained zero running witnesses.

The affected recursive handoff regression also passes: 100.66 seconds,
including compilation, with the existing commitment and point mutation
checks. It reuses the earlier iteration-1-to-2 proof and does not execute
the new active public producer. The selected public API documentation
example compiles. Independent source review passes with exact source
hashes and leaves active extension and cache recovery open.

Ordered static, build and axiom checks pass in 8.88, 0.97 and 1.02 seconds.
No package definition, emitted bytes or pins changed. Logs, source snapshots
and the source review are retained in `STAGE1_PUBLIC_BASE_EVIDENCE.zip`.
The next check loads the actual iteration-2 envelope and its six nonzero
running witnesses into the existing staged C/R/D producer.
