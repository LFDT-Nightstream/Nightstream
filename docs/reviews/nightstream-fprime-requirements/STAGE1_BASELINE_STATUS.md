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
| Nonzero running execution | The actual iteration-2 source route loads six nonzero running witnesses and ten virtual zero witnesses. All seventeen source openings and a private-source substitution rejection pass. | The C stage reached the 300-second cap. Execute a feasible C/R/D sequence, then compare the complete later caller assignment and terminal output. |
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

## Feedback checkpoint

The crate-wide test build now compiles every target, including the fixture
binary's test harness. The broken Cargo binary-path reference belonged to an
integration test imported into that harness. The integration checks now live
in `stage1_nifs_execution.rs`, under the unchanged `nifs_stage1_nifs` target;
no harness was disabled. The final `--no-run` check passes in 18.16 seconds
(reported maximum RSS 617,372 KiB). The moved integration target passes three
tests in 20.45 seconds (reported maximum RSS 3,432,844 KiB). Its previously
ignored full-producer test retains the recorded cap failure and exclusion.

`HyperNovaFalseAcceptance.probability_linear_bound` now bounds the actual
selected full-opening false-acceptance event: terminal acceptance with no
advice list of the advertised length and forward application result. On the
original mixed law it proves the symbolic bound

`Pr[FalseAccept] ≤ Σ(j < depth) [h_j + a_j - g(Q_j,a_j) + deltaFS(Q_j) + weakLoss + testError + 17 * m_j]`.

Here `a_j` is the actual good-active visit mass, `h_j` is the marked hash
collision mass, and `m_j` is actual adaptive MSIS success on that same guarded
law. The depth, query functions, guarded FS models, invertibility, declared
clock bounds and moments remain explicit. No source conditioning, positive
returned-history mass or numerical deployment choice is introduced. This is
not a claim about bare NIFS Boolean acceptance or a new proof backend.

The focused proof passes in 2.32 seconds, its literal
`LeanGraph.Targets.HyperNovaTerminalFalseAcceptance` target builds in 4.52
seconds, and exact target acceptance passes in 2.12 seconds. Independent
source review passes. The graph suite passes all 83 tests, including a check
that the test-build gate cannot omit the fixture harness or integration target.
Ordered final static, build and axiom gates pass in 8.88, 11.54 and 4.62
seconds. The frozen graph metadata, boundaries and literal-target gates also
pass. The combined graph checkpoint reached its 1,500-second cap during the
separate frozen security gate; exit 124 is a failed checkpoint, not a pass.
Cleanup ended at 1,540.14 seconds. Reported maximum process RSS was
10,894,524 KiB; this is not the simultaneous memory of the full process tree.
The separate security gate passes on the same frozen snapshot in 987.48
seconds, including orchestration, with maximum process RSS 11,049,956 KiB.
`FALSE_ACCEPTANCE_GATES.json` identifies all four passing frozen gate records.
Controlled-checker approval remains separate from these local compiler
results and the independent source and graph reviews.

Four obsolete source/history square-root wrappers and their audit entries
are removed. The linear source/history chain remains. Lower lemmas still
used by the prepared-work chain, and the old invalid-source consumer, are
retained; removing those would require a further consumer change. The six
map dispositions are explicit in `scripts/lean_graph/STAGE1_BASELINE.md`.

The real iteration-2 sources pass all seventeen opening checks and a private
source-substitution rejection in 134.64 seconds. The combined C/R attempt
and a separate C attempt each reached the 300-second cap without returning
a parent or C proof. Source authentication consumed about 118 seconds in
both. Their peak memory was not measured; process samples are not a peak.
The owner chose existing limits with measured memory reporting. Further
execution requires a measured feasible stage; these failures are not passes.
The explicit-prior Lean emitter preserves the earlier fixture bytes, including
the new explicit-state route. Later C/R/D, caller and terminal execution remain
open. No constraint, profile, transcript or package identity changed.

The explicit-state emitter rejects zero and overflowing counters, wrong state
widths and noncanonical field words without writing a fixture. The affected
Rust checker builds and rechecks the existing iteration-1-to-2 caller,
including all physical and logical rows, in 35.67 seconds with maximum process
RSS 3,632,528 KiB. This regression preserves earlier evidence; it does not
execute the pending iteration-2-to-3 fold.

New archives are outside Git. `EVIDENCE_RELEASE.json`,
`EVIDENCE_SHA256SUMS`, `TERMINAL_REPLAY_INPUTS.json` and `EVIDENCE_STORAGE.md`
describe the prepared release assets and fresh-checkout replay. The previously
missing sixteen child witnesses are now archived with matching input hashes.
GitHub release upload and fresh-download verification remain pending sign-in.
Existing archives and links remain until their replacements are verified.
The retained Sites project is still unavailable, so no live publication is
claimed. The local publication command already rejects uncommitted inputs.
