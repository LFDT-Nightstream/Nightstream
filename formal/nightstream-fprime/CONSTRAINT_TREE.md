# Nightstream F′ Stage 1 constraint tree

This file is the concise audit map for the current Lean-authored
`Poseidon2HashChainV1` package and its reusable prefix. It defines no relation
and gives no digest authority. The audit path is:

```text
paper formula
  → Lean predicate
  → leaf FormalCircuit
  → phase assembler
  → proved physical layout
  → canonical Lean package
  → exact Rust matrices and independent assignment check
```

Status on this cut:

The stored package and its identities below precede the confirmed application
wiring defect. The source repair is in progress. No repaired package has been
approved or pinned. External trial emission does not validate a replacement.
Existing conformance evidence must not be transferred to the changed relation.

- Pilot: the standalone prior-state and output-hash pilot is
  **Conformance-closed** on package identity
  `[5272192602150446227, 11110764831345399822, 12712750146236044807, 13354028730245635118]`.
  Current independent review confirmed the complete required compiler,
  fixture, result-parity, exact-matrix, production-assignment, independent-row,
  and mutation evidence recorded below. **Production-closed** remains open.
  This retained pilot result applies only to that stored identity. It is not
  evidence for a repaired identity and does not close PiCCS, full Stage 1, or
  the arbitrary emitted-assignment-to-`StepHoldsFor` obligation.
- PiCCS: status open. The stored complete-result fixture remains synthetic.
  The current unpinned candidate now has a proof constructed from the checked
  nonzero base opening, complete ring outputs, and a passing complete
  Lean / `paper_exact` / optimized result comparison. Its independent
  physical and logical prefix assignment gate, owner mutations, and proof
  mutations pass. Current pilot results, parent binding mutations, and full
  independent opening checks also pass. The remaining phase proof connections
  and exact-cut review are required before **Conformance-closed**.
- PiRLC: retained candidate only. Its identity-dependent fixture was not
  regenerated after the current PiCCS state-binding update. Work remains
  frozen until PiCCS is conformance-closed.
- PiDEC: retained candidate only. Its identity-dependent fixture is not
  evidence for the current PiCCS handoff and was not regenerated.
- Running-instance branch: compiler and phase-local conformance gates are
  green; status open as part of the cumulative package cut.
- Accumulator: the zero-row SuperNeo verifier composition and canonical
  package edge, final package identity, and recursive fixed point are
  kernel-checked; status open as part of the cumulative package cut and its
  exact external review.
- Application: open. The stored logical relation accepts an application
  output detached from the prefix state-hash output. The regression and exact
  source repair are recorded below. The shared-form, matrix-substitution,
  canonical encoding, transport, and arbitrary-assignment application proofs
  build. The current candidate passed exact physical and logical matrices
  and independent base-assignment evaluation. The detached-application
  regression now rejects at logical row 6,377,546. The universal theorem and
  the remaining conformance gates are still required.
- Stage 1: open. The eight-child opaque assembly order, derived offsets,
  aggregate footprints, and logical soundness under child assumptions are
  kernel-checked. The generic package-row-to-`StepHoldsFor` theorem requires
  `Represents`; the concrete closure covers the constructed
  `(bound raw).assignment`. The implication from an arbitrary satisfying
  emitted assignment to its decoded `StepHoldsFor` remains open; the
  application regression now gives a concrete failure of the stored relation.
  The stored cut has a concrete application, recursive fixed point,
  `2^28` proof, scoped deterministic closure, security-or-collision theorem,
  sealed package, terminal metadata, and setup parity. Those results do not
  close the repaired relation. Complete repair validation, the package-only
  Rust lifecycle, downstream reruns, and exact external review remain open.

The first 2026-09-04 independent Linux review rejected its cut because a
compressed Rust PiCCS path was reachable, its source fingerprint was not
reproducible, current audit comments were stale, and the retired loader suite
was red. A second review of manifest `f8697af3...` confirmed those repairs but
found a separate public route through the unapproved Nebula F′ prototype. This
working cut makes that Stage 2 module crate-private and keeps its dependent
integration targets inactive. It remains unapproved until an independent
reviewer checks its new exact manifest. No status below treats either rejected
review as approval.

## Current proof and execution priority

The owner has prioritized smaller proof obligations and measured execution
costs. The target remains the implication from every assignment accepted by
the selected matrices and actual public-input boundary to the decoded
`StepHoldsFor`. A theorem about only `(bound raw).assignment` cannot replace it.
The existing leaf → phase assembler → eight-child Stage 1 assembler stays in
place. Each parent uses opaque child contracts and proves shared-value wiring.

The current dependency map is below. Module names in this table are relative
to `NightstreamFPrime/Export/Stage1/`; the generic decoder is in
`NightstreamFPrime/Layout/Stage1/StateDecoder.lean`.

| Obligation | Current proof and exact remaining connection | Focused build target |
|---|---|---|
| Accepted assignment and public boundary | `PerApplicationMatrixProgramSemantics.matrixProgramExact` connects the selected rows. `ActualPiCCSInputs.selectedRowsAndPublic_imply_phaseAndHashes` derives the one cell from the actual CCS public marker and composes the exact PiCCS phase predicate, typed running-input agreement, fresh public-hash equation, and claimed next-state hash. It has no separate one-cell or representation premise. The verifier-owned context and final typed step connection remain open. | `NightstreamFPrime.Export.Stage1.ActualPiCCSInputs` |
| Actual value decoding | `PiCCSAssignmentSoundness.decodedEnv_location` and `ActualPreimageFraming.rowsZero_implies_actualPreimageCanonical` derive form values and canonical preimage framing. They require no canonical trit encoding. | `NightstreamFPrime.Export.Stage1.ActualPreimageFraming` |
| Shared inputs and application step | `PilotDecodedEnvironment.priorWord_agrees`, `outputWord_agrees`, and `priorPublic_agrees` identify the pilot/PiCCS readers. `ActualApplicationStep.selectedRowsZero_implies_decodedStep` proves `zNext = application.step zi witness` for values decoded from the same assignment. | `NightstreamFPrime.Export.Stage1.ActualApplicationStep` |
| Pilot and PiCCS children | `PilotDecodedPhase.selectedRowsZero_implies_specHolds` and `PiCCSDecodedPhase.selectedRowsZero_implies_phaseHolds` derive the opaque phase contracts from the selected rows. | `NightstreamFPrime.Export.Stage1.PiCCSDecodedPhase` and `NightstreamFPrime.Export.Stage1.PilotDecodedPhase`, run in sequence |
| Exact hash preimages | `ActualHashSlots.rowsZero_implies_nextPreimageSerialization` and `selectedRowsZero_implies_hashSlots` derive both hash equations from the pilot contract, actual preimage framing, and next-preimage rows. The next constructor uses the decoded prior counter plus one and needs no non-wrap premise. Connecting this decoded constructor to the complete typed lifecycle input/output remains open. | `NightstreamFPrime.Export.Stage1.ActualHashSlots` |
| Typed accumulator and running values | `ActualPiCCSInputs.evalRunning_eq_priorRunning` identifies the complete PiCCS running claim with the actual decoded prior preimage. `evalFreshPublic_eq_priorPublic` and `selectedRowsZero_implies_freshPublicHash` bind its fresh public input to that same hash. The complete typed NIFS proof/output and later-phase reader agreement remain open. A PiCCS-only environment cannot supply PiRLC/PiDEC proof fields. PiRLC remains frozen until PiCCS is **Conformance-closed**. | `NightstreamFPrime.Export.Stage1.ActualPiCCSInputs`; the remaining full-step owner is `PerApplicationFixedPointSoundness` |
| Decoded step from arbitrary rows | `ActualRunningTransition` derives the transition and actual preimage wiring. `ActualStep.selectedRowsAndPublic_imply_baseStep` proves the zero-counter branch. `selectedRowsAndPublic_imply_piCcsCheck` proves the concrete NIFS PiCCS check on the same decoded inputs. `selectedRowsAndPublic_step_iff_baseOrPiDec` isolates the remaining recursive PiDEC acceptance and computed-output equations. Canonical context binding remains open. | `NightstreamFPrime.Export.Stage1.ActualStep` |
| PiDEC from arbitrary rows | `ActualPiDEC.selectedRowsAndPublic_imply_phaseHolds` derives the exact output predicate through the canonical physical layout. `evalPoint_eq_piCcs` preserves the PiCCS point; `decodedEnv_location` identifies each PiDEC-owned form. Equality with the verifier-computed PiRLC parent remains open and the retained parent-wiring regression below fails. | `NightstreamFPrime.Export.Stage1.ActualPiDEC` |
| Complete parent result | Compose the derived hash equations, application equation, typed phase results, branch condition, and public/context binding into the unchanged `StepHoldsFor` definition. No `Represents`, `Encodes`, application-correctness, or NIFS-correctness premise may enter at the final acceptance boundary. | Existing owner `NightstreamFPrime.Export.Stage1.PerApplicationFixedPointSoundness`; the final theorem is open |

Before implementing an open arrow, record its exact Lean statement, a short
mathematical argument, its named dependencies, and the build target that checks
it. If an argument contains an unresolved step, refine that step before adding
proof code. A helper is necessary only if it removes a missing assumption or
proves a distinct required connection. Do not add another record that merely
passes the same missing representation facts onward.

Use [lean-graph](../../scripts/lean_graph/README.md) for the dependency and evidence workflow. Its current
registered targets are `LeanGraph.Targets.PilotAssignment` and
`LeanGraph.Targets.PiCCSAssignment`; the full Stage 1 target remains open.
Use `checkpoint piccs-assignment` to run or resume its registered checks, and
`explain piccs-assignment` to inspect the remaining work. The guide gives the
full Python invocation and graph-query commands.
The latest diagnostic checkpoint rebuilt those targets from captured source
in 558.478 s and exported the declaration metadata in 48.829 s. Its subsequent
assignment check reused the retained build and still ran the build command
(5.256 s), then checked the acceptance driver (10.197 s). The current
`explain piccs-assignment` report identifies both results as passed, current,
and diagnostic. Checker approval and the target-meaning review remain missing;
accepted closure is open. These results do not grant an assurance status.

The original graphs confirm the existing child-contract and assembler
dependencies. They do not contain `ActualHashSlots` or `ActualPiCCSInputs`.
The new `selectedRowsAndPublic_imply_phaseAndHashes` theorem now composes
those component proofs at the phase boundary. Its first build passed in
14.326 s and the full axiom gate in 11.685 s, with only the allowed axioms.
It retains the actual public equality and four-word digest shape as inputs;
it derives the one cell before applying the opaque phase result. The full
Stage 1 target remains open and must not be replaced by this phase result.

The original graph remains evidence for its older snapshot. The new
`requires LeanGraph.Targets.PiCCSAssignment` query shows the exact current
target: arbitrary assignment, the constant-column premise, and accepted
selected rows imply `PhaseHolds`, without a `Represents` premise. It does not
replace the stronger public/context binding or full `StepHoldsFor` target.
No matrix, witness instruction, relation, or identity changed during this
workflow update. Keep target meanings and allowed premises explicit before
registering the stronger final target.
Use focused incremental builds for proof edits and graph export when it
answers a dependency question. Keep conformance checks and required reviews
separate from graph metadata and local diagnostic status.
Registered declaration keys can now preserve an assignment-target result after
a complete export from the selected source. The export gate itself still
checks its full declared source group. This does not waive conformance checks
or reviews tied to their exact snapshots.

The hash argument now composes the two existing builder theorems with the
row-derived representations, equal context keys, equal initial states, and
encoded counter increment. The final four public pin rows identify the
decoded output digest with the claimed digest. The public marker supplies
the one cell; no canonical packet or representation premise is supplied at
that boundary. The exact typed input/output constructors for the final
arbitrary-assignment theorem remain open. In particular, the decoded
`ActualHashSlots.nextPreimage` has not yet been identified with the complete
typed lifecycle `nextHashPreimage`.

The focused hash build passed in 12.267 s. The public-boundary extension
first failed because a local name used the reserved word `public`; the
corrected build passed in 5.173 s. The full axiom gate then passed in
73.462 s, 3,663 jobs, with only the three allowed logical axioms. It audits
all seven new public theorems in `ActualHashSlots` and
`RecursivePublicOutputPlan`. These additions do not change an existing
relation, row, witness instruction, fixture, or identity. They close these
proof components only; full Stage 1 and current PiCCS **Conformance-closed**
status remain open.

The three typed PiCCS input connections now build in
`ActualPiCCSInputs.lean`. They reuse `StateDecoder.evalRunning_eq_running`,
`ActualPreimageFraming.priorWord_eq`, and the shared public-input reader
theorem. They do not repeat the decoder's component proofs or add copy rows.
The first focused build passed in 12.352 s; the full axiom gate passed in
12.078 s, 3,664 jobs. All three public theorems use only the allowed axioms.
No existing relation, matrix, witness instruction, fixture, or identity
changed.

The selected-context connection remains distinct. The old
`PerApplicationFixedPointSoundness.rowsZero_implies_stepHoldsFor` uses the
context decoded from a constructed packet. Its verifier-bound variant fixes
the context through `PerApplicationVerifierBoundAssignment.bind`. Neither
result derives the verifier-selected context from the acceptance boundary
for an arbitrary assignment. The final theorem must supply that connection;
the new typed input equalities do not assume it.

The temporal external review dated 2026-09-05 20:44 UTC was read completely.
Its arbitrary-assignment finding applies, and its stored-package geometry
mismatch remains open until a validated replacement is reviewed and pinned.
The external candidate has current emission, exact matrix, independent
assignment/opening, complete PiCCS result, and mutation records; those records
do not update the stored package. The review's emitter and loader allocation
finding requires measured cost evidence for any further execution change.
The existing proved-width optimization and rejected cache trials are recorded
below. Broader memory phases and a `2^24` target are outside the active Stage 1
contract. The review does not replace the required named, exact-source PiCCS
formula-coverage review or grant permission to resume PiRLC.

Use a focused owner-module build during each proof edit, then audit its public
theorems. Run the required full axiom and boundary gates at the checkpoint.
Retain matrix, assignment, fixture, and mutation evidence while the relevant
definitions, inputs, and emitted bytes are unchanged. A semantic or relation
identity change reopens every affected dependent check before any new pin.

Execution work starts with the measured emitter and canonical binding paths.
The current emitter baseline is 143.327 s, including its build, and matches all
128,464,976 retained candidate bytes. A 10-second native sample found repeated
application circuit construction while evaluating retained-prefix widths.
`PiRLCRetainedGeometry.prefixLogicalWidth_eq` already proves that the prefix
owns exactly 192,090,438 coordinates for every application. The first speed
change uses that proved value during compilation, through
`prefixLogicalWidth_eq_directPrefixLogicalWidth`. Its focused axiom audit
passed in 134.431 s, including the dependency rebuild. The emitted package
then matched every retained candidate byte. The observed native interval from
process launch to the final output write was 138.215 s before the change and
124.913 s after it. The corresponding command totals, including builds, were
143.327 s and 155.116 s; these totals are not comparable warm-build timings.

The final full axiom gate passed in 82.519 s, 3,662 jobs. A fresh final emission
after removing the cache experiments passed in 146.332 s including the build;
its native interval to the final output write was about 124.747 s. All
128,464,976 bytes matched the retained candidate, with SHA-256
`07a18d8a24b064ae008660f08e75ac3ee0d91ba74482903b72e8ba0e7db97627`.
The profile and all rows, columns, source order, serialized inputs, and
relation identities remain unchanged. A speed result does not close a
semantic or conformance obligation.

Source-scan caching was rejected: the trials preserved the bytes but showed
no clear extra speed gain over the width change; the added cache code was
removed.

The final canonical binding matched all 1,603 retained bytes. Its command took
230.540 s including the rebuild; the native interval to the final output write
was about 215.309 s, compared with 224.371 s before the width change. These are
single-run observations. Baseline launch times came from the native sampling
report; the final launch observation has one-second resolution. No new package
or verification-key identity was pinned.

The final repository boundary gate passed in 15.224 s. The 20:28:58 UTC
preservation check found no protected-file change, staged path, or source-size
violation. It still exits 1 for the same two previously flagged frozen
generated-file paths, with no additional path flagged. All build output and
emissions were outside the repository. No commit, stage, reset, stash, file
removal, restore, or discard command was used. Both required reviews were read
completely at the 20:20 checkpoint before the next command or edit; the proof
map retains the applicable universal-assignment and production obligations.
Current Rust matrix, assignment, nonzero-result, and mutation evidence remains
attached to the unchanged candidate bytes. Those expensive gates were not
rerun for this compiler equality change, and no phase status was promoted.

## Fixed profile and semantic authority

Lean is the semantic authority for exact SuperNeo v1_1. The fixed Nightstream
Goldilocks profile is:

| Parameter | Value |
|---|---:|
| Base `b` | 2 |
| `k_rho` | 16 |
| Bound `B` | 65,536 |
| Fresh / running PiCCS inputs | 1 / 16 |
| PiRLC inputs | 17 |
| PiDEC children | 16 |
| Ring degree | 54 |
| Main Ajtai rank `κ` | 22 |
| CCS matrices | 14 |
| PiCCS rounds | 28 |
| PiCCS round coefficients | 10 |
| Public-input words | 270 |

`Eval_K` is the separate Pad family. `Eval_A` is the separate 14-matrix
family. No v1.0 Pad-as-matrix-zero encoding is present on the canonical Lean
path or a normal-build public F′ path. The retired compressed Rust emitter is
crate-private reference code. Its unapproved Stage 2 caller is also
crate-private and is not a production API. Transcript, state, package-identity,
and verifier-context binding use Poseidon2 only.

The public digest encoding is exact `encHash`:

```text
[marker = 1, 256 little-endian digest bits, 13 zero cells]
```

Rust now uses this same 270-cell encoding. `decodeHash_encHash` recovers every
canonical four-word digest, and `encHash_injective_fixed` proves that two such
public inputs are equal only when their digests are equal. The canonical state
preimage is 49,393 Goldilocks words. Each running group contains 1,188
commitment words, 270 public-input words, and 1,620 evaluation words.

`Layout.ProductionRelation.Plan` is the key-facing matrix authority boundary.
It derives all 14 typed SuperNeo matrices from 13 meaningful sparse row forms
in the canonical Boolean-row order; matrix slot 13 and every padding row are
zero by construction. `Plan.matrixVectorAt_matrix` proves that each matrix
image equals its sparse row-form evaluation. `ProductionRelation.polynomial_zeroImages`
proves that the fixed 74-term polynomial accepts the all-zero padding rows.
`PerApplicationStructuralPlan.structuralPlan` constructs the complete
low-norm plan. `PerApplicationMatrixProgramSemantics.matrixProgramExact`
proves that the canonical matrix program implements its exact 14 row forms.
These matrices are the relation in the current sealed package.

## Logical phase hierarchy

```text
Lifecycle/Pilot.lean                         ✓ two hash children
Lifecycle/PiCCS/v1_1/Formal.lean             ✓ twelve-child assembler
Lifecycle/PiRLC/v1_1/Formal.lean             ✓ seven-child assembler
Lifecycle/PiDEC/v1_1/Formal.lean             ✓ six-child assembler
Lifecycle/Stage1/RunningTransition.lean      ✓ running-instance branch
Lifecycle/Stage1/Accumulator.lean            ✓ exact NIFS verifier result
Layout/Stage1/AccumulatorSemantics.lean      ✓ zero-copy phase composition
Export/Stage1/AccumulatorPackage.lean        ✓ zero-row package edge
Lifecycle/Stage1/Application.lean            ✓ per-application proof contract
Layout/Stage1/ApplicationInputs.lean         ✓ zero-copy four-word ABI
Layout/Stage1/ApplicationSemantics.lean      ✓ typed state representation
Export/Stage1/ApplicationPackage.lean        ✓ direct plan and custody theorem
Export/Stage1/PerApplicationPackage.lean     ✓ generic final package and F edge
Export/Stage1/PerApplicationSoundness.lean   ✓ package rows imply StepHoldsFor
Lifecycle/Stage1/VerificationKey.lean        ✓ acyclic key binding preimage
Lifecycle/Stage1/Terminal.lean               ✓ outer terminal semantics
Export/Stage1/TerminalPackage.lean           ✓ zero-row terminal metadata
Lifecycle/Stage1/Interface.lean              ✓ symbolic child interfaces
Lifecycle/Stage1/NextPreimage.lean           ✓ counter increment and initial-state carry
Lifecycle/Stage1/Formal.lean                 ✓ sole eight-child FormalCircuit
Layout/Stage1/AssemblerInputs.lean           ✓ compact cross-phase wiring
Layout/Stage1/AssemblerSoundness.lean        ✓ deterministic relation composition
Layout/Stage1/PreservationClosure.lean       ✓ final physical preservation
```

PiCCS leaf ownership:

| Leaf | Rows | Column delta |
|---|---:|---:|
| Statement binding | 160 | 0 |
| Digest-only statement absorption | 224,368 | 224,368 |
| Challenge derivation | 51,504 | 51,504 |
| Round transcript | 149,184 | 149,184 |
| Initial claim | 116,631 | 116,631 |
| SumCheck chain | 424,657 | 424,601 |
| `Eval_K` terminal | 8,542 | 8,542 |
| `Eval_A` terminal | 109,630 | 109,630 |
| CCS terminal | 20,794 | 20,794 |
| Norm terminal | 752 | 752 |
| Final identity | 130,503 | 130,501 |
| Output binding | 4,076,512 | 4,076,512 |

PiRLC leaf ownership:

| Leaf | Rows | Column delta |
|---|---:|---:|
| Input binding | 0 | 0 |
| Sampler chain | 1,008,848 | 1,007,199 |
| Commitment combination | 3,049,596 | 3,049,596 |
| Public-input combination | 693,090 | 693,090 |
| `Eval_K` combination | 277,236 | 277,236 |
| `Eval_A` combination | 3,881,304 | 3,881,304 |
| Output binding | 0 | 0 |

PiDEC leaf ownership:

| Leaf | Rows | Column delta |
|---|---:|---:|
| Input binding | 0 | 0 |
| Public-input split and range checks | 22,680 | 18,090 |
| Commitment recomposition | 1,188 | 0 |
| `Eval_K` recomposition | 108 | 0 |
| `Eval_A` recomposition | 1,512 | 0 |
| Output binding | 0 | 0 |

The PiDEC verifier first checks centered parent magnitude `< 2^16`. Its
accepted path then uses the exact 16 signed radix-two digits. The computable
decision agrees with the semantic predicate. The accepted path cannot use
`fallbackDigit` for an out-of-range parent.

## Proved cumulative and final layout

The stored-artifact ledgers, before the application repair, are:

- `PilotProduction.physicalRowCountValue_eq` and
  `PilotProduction.physicalColumnCount_eq`;
- `PilotPiCCS.cumulativeFootprints_eq`;
- `PilotPiCCSPiRLC.cumulativeFootprints_eq`;
- `PilotPiCCSPiRLCPiDEC.cumulativeFootprints_eq`;
- `PilotPiCCSPiRLCPiDECRunningTransition.cumulativeFootprints_eq` and
  `jointDomain_le_twoPow28`;
- `Export.Stage1.Package.circuitPackage_layout_values`;
- `Export.Stage1.Package.circuitPackage_jointDomain_le_twoPow28`.

| Endpoint | Physical rows | Physical source columns / joint domain |
|---|---:|---:|
| Pilot | 14,623,730 | 14,722,512 |
| Through PiCCS | 19,936,967 | 20,064,823 |
| Through PiRLC | 28,847,041 | 28,973,248 |
| Through PiDEC | 28,872,529 | 29,040,586 |
| Through running-instance branch | 29,218,024 | 29,336,724 |
| Final application package | 29,225,729 | 29,344,425 |

The stored recursive relation has 6,377,559 structural rows and logical width
264,627,433. Its Φ81 carrier width, and therefore its exact joint domain, is
264,627,486. This is below `2^28 = 268,435,456` with 3,807,970 points of
headroom. The outer terminal metadata adds no row or column.

The current source shares the application input/output, PiCCS ordinary
preimages, and running-transition preimages with the actual pilot hash inputs.
This removes 328 application coordinates and another 6,075,790 duplicated
preimage coordinates. PiDEC proof outputs now share the running-transition
child coordinates, which removes another 2,019,168 coordinates. The 38-block
schedule has these proved values:

| Current source representation | Exact value | Lean evidence |
|---|---:|---|
| Logical active rows | 6,377,559 | `Poseidon2HashChainV1Package.structuralRowCount` |
| Logical columns | 256,532,147 | `Poseidon2HashChainV1Package.logicalWidth` |
| Complete carrier / live joint domain | 256,532,184 | `Poseidon2HashChainV1Setup.carrierWidth_eq` |
| Ring columns | 4,750,596 | `Poseidon2HashChainV1Setup.messageColumns_eq` |
| Carrier alignment | 37 | difference of the preceding width theorems |
| Boolean domain | 268,435,456 | `jointDomain_le_twoPow28` |
| Logical-column delta from stored package | −8,095,286 | difference of the exact width theorems |
| Carrier-column delta from stored package | −8,095,302 | difference of the exact carrier theorems |

Physical rows and columns remain 29,225,729 and 29,344,425. The full existing
`Poseidon2HashChainV1Closure` build passed in 49.453 seconds on this source,
including the fixed-point and domain theorems. Its final step theorem still
covers canonical raw assignments. These counts do not establish arbitrary
assignment soundness or validate newly emitted bytes.

## Stored canonical package cut

| Final sealed package value | Exact value |
|---|---:|
| Physical rows | 29,225,729 |
| Logical relation rows | 6,377,559 |
| Logical relation columns | 264,627,433 |
| Private columns / constant source index | 29,344,146 |
| Public columns | 278 |
| Total physical columns | 29,344,425 |
| Caller-owned private inputs | 177,326 |
| Witness instructions | 1,211,182 |
| Assertion rows | 201,386 |
| Ordinary compiled rows | 1,412,568 |
| Poseidon2 invocations | 7,757 |
| Compact templates | 326 |
| Compact invocations | 170,918 |
| Padded matrix rows | 268,435,456 |

The stored verifier-owned identities, before the repair, are:

```text
structural = [16811687277879400436, 860456718252016362,
              16809159788790180735, 11447255358434504088]
package    = [5272192602150446227, 11110764831345399822,
              12712750146236044807, 13354028730245635118]
context    = [1980942344823989826, 5434686752167889125,
              1771901317533452586, 10480267795687330756]
vk         = [7060461157808352439, 12469274673870775826,
              13276126990617570414, 14803309506206887238]
```

No external reviewer has approved a source fingerprint for this cut. The
reviewer can reproduce the exact cut with:

```bash
python3 scripts/fprime_stage1_review_manifest.py check \
  FPRIME_STAGE1_REVIEW_MANIFEST.json
sha256sum FPRIME_STAGE1_REVIEW_MANIFEST.json
```

The canonical manifest covers Git HEAD, current tracked and untracked source,
tests, configuration, owner inputs, paper inputs, executable file modes, and
the six required artifacts below. It hashes raw worktree bytes in two matching
passes and rejects missing inputs, scoped tracked deletions, symlinked inputs,
Git-LFS pointers, or concurrent changes. The manifest is review metadata. It
is not a protocol digest or verifier authority.

SHA-256 identifies bytes only:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Standalone pilot parity | 199,534 | `99da89867114d3408979398b55f9869aad02fceebe5b80d4b6064d29c57445fe` |
| Base-step caller fixture | 367,974 | `24a854951c27519833956cf05856ded14f52ffcabf03a4f0a25984c7d4a33202` |
| Final sealed package | 126,436,452 | `1b0c9977724998b4b261001d503899439eedc9427d04950f00b4da5bf442427b` |
| Separate expanded package | 117,391,937 | `952a21ddca40e6223c7f8a0696ee5716eea741e6d6352a6b4e14cad1c634ef21` |
| Package-binding parity | 1,611 | `20e14acdcbe4e5df79a2eb1dce2a506c2ef458451b476cd1eaddfa17f7b59275` |
| PiCCS parity | 1,642,619 | `6406418a687884a68437f902d23bba7ab5e535c26961f0ad0b88a32e94f5e0e7` |
| PiCCS ownership | 1,649 | `9577bab3983538562d756a7d1458ba61549ae044400300466b097e9444b3d964` |
| Ajtai setup parity | 903 | `2279463a3b76aa273626d2028b62d4cb4d1ad30da945db723347050ec08cba51` |

The standalone pilot parity artifact was regenerated for the current verifier
context. The PiRLC, PiDEC, sampler, application, and terminal parity artifacts
were not regenerated and are not current-cut evidence.

## Exact conformance evidence

Assignment, commitment, and artifact checks below use the stored identity
unless stated otherwise. The new application and checker builds are scoped
source evidence; they do not validate a new emitted relation.

The application wiring regression is a required failing test on the stored
relation. `base_step_rows_reject_a_detached_application_output` in
`crates/nightstream-fprime/tests/base_step_assignment.rs` kept the complete
original prefix, hash, and public coordinates, then substituted a different
valid application witness, local values, and output. The mixed assignment
remained bounded. All 6,377,559 logical rows and zero padding accepted it, so
the expected rejection failed after 174.688 seconds total / 158.58 seconds
native. This is a constraint defect, not only a missing canonical
representation proof.

The repair maps application input/output words directly to the actual
`PiRLCPoseidonGeometry` prior/output preimage forms at slots `35+i`.
`ApplicationDirectPlan.Location.input_form_eq_pilot` and
`output_form_eq_pilot` prove unconditional form equality.
`ApplicationMatrixProgramSubstitution` proves the emitted source ranges use
those forms. PiCCS ordinary prior/output preimages and the running-transition
prior-state slice/output preimage now reuse the same pilot coordinates.
Their unconditional form-equality and matrix-program proofs also build. No
copy rows are added. No separate executed attack is claimed for these other
preimage copies; the full arbitrary-assignment connection remains open.

Lean evidence for the current parent-owned transcript and payload repair:

The ordered payload-table change passed the top-level
`Poseidon2HashChainV1Closure` target (33.726 s). The canonical transport,
sampler-invocation, and digest-lane batch constructors now select the existing
proved direct expressions. Their focused proofs, expansion/completeness
parents, and full dependent build pass. The latest complete axiom gate passed
(192.664 s; 3,655 jobs; allowed logical axioms only), and boundary enforcement
passed (12.120 s). All 731 checked source/configuration files match the
external validation copy. These are compiler and construction results.
The complete arbitrary-assignment-to-`StepHoldsFor` theorem, current Rust
conformance gates, and production lifecycle remain open. Full Stage 1 is not
Compiler-closed, Conformance-closed, or Production-closed. No new identity or
artifact was pinned.

- `PiCCSPoseidonPlan/Retained.lean` and `RetainedValues.lean` own the existing
  retained coordinates, physical output forms, and source-view transport.
  Parent readers use these contracts without importing the complete phase plan.
  No alias layer, allocation, or copy row was added by the move.
- Running round challenges and PiCCS ordinary transcript readers use actual
  retained Poseidon2 output forms. Their Lean matrix-program equality and
  source-preservation proofs build. The computed readout also has an exported
  expression rewrite with equality for every raw source assignment.
- `PiCCSPoseidonPlan.Payload` is a typed family of parent-supplied forms.
  `DirectPiDECPrefixPlan.piCcsPayload` selects `PiCCSPayloadWiring.form` from the
  parent's actual ordinary source map. The leaf owns its permutation and
  binding rows. The parent-owned phase plan builds (13.022 s).
- `PiCCSPayloadWiring.form_eval` proves the exact declared action expression in
  the decoded environment for any assignment with one=1. `form_eval_source`
  derives canonical payload values from ordinary source encoding. The
  canonical prefix constructor uses this theorem; it no longer obtains payload
  authority from the separate block at position 13. Canonical encoding builds
  (24.127 s). The generic preservation contract's payload agreement is
  discharged by that parent proof.
- `lowering?_isSome`, `lowering_supported`, and
  `form_eq_compileCombination` prove total bounded affine selection, exact
  source support, and the ordered constant/term form. The syntactic shape
  proof uses a proof-local matrix witness only; it selects no emitted relation.
- `MatrixProgram/Affine.lean` checks canonical field coefficients and owns the
  indexed affine source-word table. `PoseidonInput.Term.taggedAffine` is wire
  opcode 5: `[5, table, substitution, tags, requiredTag, laneCount]`. A table
  word is `[constant, [[sourceColumn, coefficient], ...]]`. Invalid coefficients,
  missing words, and unresolved source mappings fail on the selected path.
- `PiCCSPayloadMatrix` proves that physical column remapping and the existing
  source substitution produce the exact parent form, including entry order
  (6.273 s). `PiCCSPoseidonMatrixProgram` selects that table and substitution.
  Its complete `matrixProgram_row?` theorem passes (4.706 s). The Stage 1 matrix
  theorem above consumes this result. No Rust matrix equality is inferred.
- The verifier-context read now proves its source range and uses the raw
  package view. The sampler's final PiCCS state uses the shared output contract.
  Both dependent proofs build. PiRLC `InputBinding` is unchanged.
- Rust's production interpreter and independent reference now contain separate
  opcode-5 decoders and use their own source-substitution implementations.
  `cargo fmt --all` passed (2.301 s). Production matrix-program tests passed
  8/8 (30.750 s including compilation); independent affine-interpreter tests
  passed 2/2 (31.249 s including compilation). These are generic IR tests.
  The current full matrix, assignment, valid nonzero result, handoff, and
  mutation gates are still required.
- `PiCCSActionPayloadBlock.materializedPayloadExpressions_eq` proves the
  ordered payload list. `PiCCSPayloadMatrix.table_eq_ofSemantic` proves that
  fail-closed traversal gives the exact original table. These proofs use
  symbolic list induction and do not evaluate the complete schedule. The
  changed table module compiled in 2.3 s. Parent matrix and transport proofs
  consume the equalities instead of reducing the constructor.
- The candidate emitter completed in 138.587 s, compared with the earlier
  incomplete run stopped at 832.014 s. Its external sink is 128,464,976 bytes,
  SHA-256 `07a18d8a24b064ae008660f08e75ac3ee0d91ba74482903b72e8ba0e7db97627`.
  This identifies candidate bytes only. No matrix, assignment, parity, or pin
  claim follows from emission.
- Canonical sampler invocations use `fastEntryState` and `fastWindowState`;
  canonical digest-lane batches use `fastLaneSource`. Their existing equality
  theorems preserve the original expressions. The package expansion and
  completeness proofs still hold. PiRLC `InputBinding` is unchanged.
- Canonical binding now selects the existing structural-identity and
  application-digest streams. `canonicalValues_eq_production` proves equality
  with the complete canonical binding. The direct structural identity,
  denotation, and compiler-replacement theorems are explicitly axiom-audited.
  The complete gate passed (7.477 s). Canonical-source emission passed
  (214.862 s) and produced the 1,603-byte schema-1 binding fixture outside the
  repository. No caller supplies an identity or component digest in this mode.
- The unpinned candidate has structural identity
  `[6395483780002767467, 10973147492092157342, 5061748961914067290, 11266143791648088911]`,
  package identity
  `[7846963510245796854, 7907862517137483439, 11604698816813472102, 3534977988706792170]`,
  and key digest
  `[10140968464856899361, 2114737288795659729, 13558527408610742526, 619546183755297756]`.
  These are Lean-derived candidate values. They are not production pins or
  conformance evidence. Matching setup metadata and the canonical physical
  expansion were emitted to external sinks.
- On this candidate, the exact physical A/B/C comparison passed in 42.402 s.
  Their nonzero counts are `[93701820, 39358148, 28868018]`. The separate final
  14-matrix comparison passed in 56.651 s over all 6,377,559 active rows,
  including entry order, indices, coefficients, constant placement, public
  mapping, and padding. The per-matrix nonzero counts are
  `[33616548, 4650801, 139742950, 117473734, 315481803, 1535215502,
  33006078, 1726758, 32220219, 30971178, 30233769, 31133970, 30392685, 0]`.
- A fresh Lean base fixture for the current verifier context was emitted in
  30.296 s. The candidate base-assignment gate passed in 166.783 s. Independent
  evaluation accepted all 29,225,729 physical rows, compared all 256,532,147
  production logical coordinates, checked strict signed-unit bounds and the
  public projection, and accepted all 6,377,559 logical rows plus padding.
  The carrier adds exactly 37 zero coordinates. This gate does not compute a
  selected-key commitment, construct an honest PiCCS proof, or close PiCCS
  conformance. The fixture's inner zero proof is a base-only placeholder.
- The candidate matrix mutation gate passed in 134.003 s. Block reordering
  changes expanded row 0; a nonzero entry's in-range column and coefficient
  mutations change expanded row 3,037,336. Each mutated program decodes with
  the independent interpreter and then fails the same selected structural
  identity. The gate first accepts the unchanged candidate. An earlier run
  stopped on a stale coefficient-1 assumption (101.643 s, exit -6); it is not
  credited. The corrected selector skips zero coefficients and requires an
  actual expanded-row change before testing identity rejection.
- The detached-application gate passed in 164.077 s on this candidate and
  its fresh base fixture. The altered application satisfies its own 7,700
  rows when paired with its own state. Keeping the original prefix, hash,
  and public coordinates while replacing the application witness/local
  suffix fails at canonical logical row 6,377,546. This closes the reproduced
  detachment case; it is not the universal assignment-to-`StepHoldsFor` proof.
- The actual selected-key commitment of the current complete base carrier
  passed in 139.335 s and emitted 1,188 canonical field words outside the
  repository. This is a current input commitment; full Lean commitment and
  output-opening conformance remain separate from this Rust execution.
- `PiCCSInputCheck` schema 2 carries the complete running statement: the
  common point, sixteen commitments and public inputs, sixteen `Eval_K`
  families, and all `16 × 14` `Eval_A` families. Both checkers decode these
  values. Rust compares its decoded statement and canonical `I_K`/`I_A`
  blocks with Lean before checking the proof. The production statement and
  terminal-identity agreement theorems now quantify over this input. The
  focused Lean build passed in 4.938 s after correcting the dimension proof
  and its syntax; full axioms passed in 11.899 s, with only allowed axioms.
  No relation, circuit, transcript schedule, witness IR, or identity changed.

  The retained valid nonzero fresh proof was encoded in schema 2 without
  changing any proof, claim, or transcript value. Every complete phase result
  is identical to the previous result. Lean passed in 16.037 s including the
  executable build; warm checks took 10.630 s and 10.410 s. The final Rust
  comparison passed in 43.798 s including compilation, 34.632 s in the binary.
  These are individual execution measurements, not performance guarantees.

  `tests/pi_ccs_input_codec.py` passed all 24 encoding cases in 183.193 s:
  canonical bytes with or without one final newline pass; wrong dimensions,
  old schema, extra fields, noncanonical words and number spellings fail.
  Separate changed `Eval_K`, changed `Eval_A`, and distinct nonzero running
  payload cases fail in Lean and both Rust engines at the first SumCheck
  round. The payload case compares all 49,304 decoded running words. These
  are intentionally invalid inputs. Early Rust rejection provides no complete
  Rust trace. A changed output also rejects, with all fifteen complete phase
  results equal across Lean and both Rust engines (35.083 s Rust invocation).

  The first valid fixture has sixteen zero running openings. The prefix
  evaluator now puts the complete schema-2 running statement into the actual
  prior preimage before witness execution. The opening evaluator
  handles zero or signed copies of the checked fresh carrier and evaluates
  nonzero running claims at both the prior and outgoing points. The schema-2
  zero-running `Eval_K` opening replay passed in
  58.113 s including compilation and checked all 54 coefficients. The raw
  PiCCS prefix assignment and owner-mutation replay passed in 224.407 s
  including compilation, 194.10 s in the test. It checked the physical and
  logical PiCCS rows on the schema-2 fixture. The signed-running fixture below
  adds a nonzero paper phase. Its parent hash/preimage and raw-assignment
  connection remain open, as does the independent current-cut review.

  The reusable command is `check_pi_ccs_input <candidate> <id0> <id1> <id2>
  <id3> <input> <Lean-result> <accept|reject>`, built with `cargo build -p
  neo-fold-clean --release --locked --bin check_pi_ccs_input`. Run it with
  `RUSTC_WRAPPER=""` under the existing 300-second cap. The Lean input/result
  side is `validate.sh pi-ccs-input-check <input> <Lean-result>`. Neither
  command replaces bounded-opening or independent-assignment checks.
- The repository `validate.sh static` gate passed in 12.016 s. The preceding
  temporary-copy attempt failed because that copy contains an old
  `tests/PayloadCacheDiagnostic.lean` outside its import roots. The repository
  has no such file. No file was removed. All 732 checked repository Lean,
  audit, script, and package-definition files match the validated copy.
- `generate_pi_ccs_fixture prepare` now executes the exported base witness
  and logical transport, checks all 6,377,559 scalar matrix rows against the
  package polynomial, checks the bounded carrier/public projection, and
  computes its actual selected-key commitment. Preparation passed in
  149.445 s. The external cache contains the full 256,532,184-coordinate
  carrier and 13 live matrix-image prefixes; matrix 13 is implicitly zero.
- `generate_pi_ccs_fixture rounds` uses those prefixes and the executable
  Lean pre-SumCheck state for the identical commitment and public input.
  The existing Rust protocol driver absorbs each message before sampling
  its challenge. All 28 honest scalar rounds and the terminal identity from
  the folded prefixes passed in 6.166 s. The fixed norm polynomial is
  `z^3 - z`; no terminal value is solved after the transcript.
- `generate_pi_ccs_fixture running-rounds` now accepts an explicit combined
  running-evaluation prefix and starts from Lean's computed initial claim.
  It checks that the sixteen running commitments and public inputs are
  signed copies of the checked fresh opening. At `b = 2`, the signed-source
  norm contribution follows from the odd polynomial `z^3 - z`. The round
  oracle composes the multilinear prefix with the prior-point equality
  factor and the existing fresh CCS/norm term. These entrypoints are fixture
  infrastructure; parent hash/assignment evidence remains open.
  The retained zero-running proof replay passed in 6.751 s (6.158 s in the
  binary); all 41,533 output bytes match the old proof exactly. The earlier
  invocation took 6.166 s. These are single-run timing observations.
- `generate_pi_ccs_fixture running-prefix` builds that prefix from the actual
  canonical matrices and bounded carrier. Its coefficient-weighted Bar/ring
  kernels pass the direct full-ring check at every native lane. The complete
  256,532,184-coordinate prefix passed in 52.744 s; its evaluation at the
  prior CE point equals Lean's nonzero initial claim. The signed-running
  rounds passed in 9.891 s, and all fresh ring families in 188.687 s.
  The sixteen running openings are alternating signed copies of the same
  checked fresh carrier. Their input claims use its previously checked output
  point; their output families use the new verifier-derived point.

  The resulting schema-2 input is 1,485,960 bytes, SHA-256
  `5e2f7568dd6314f8a5c457cbf565c1c9aedc004e8879d31a0b134ab485831c3b`.
  All 280 round coefficients are nonzero. Lean accepts (10.747 s), and
  `paper_exact` and `optimized` match every complete phase result (35.348 s).
  The Lean result is 3,886,682 bytes, SHA-256
  `8c9b7418f86023981b77933dda442d333347bc296bd00b39923bc5abbdcf1b36`.
  These hashes identify evidence bytes; they are not protocol authority.

  The independent evaluator checks the fresh carrier and each signed running
  input/output family from the raw canonical rows. `Eval_K` passed in
  67.457 s including compilation. All fourteen `Eval_A` families passed in
  80.909–125.400 s each, including all zero matrix-13 coefficients. The
  fresh CCS check passed in 64.732 s over all 6,377,559 active rows and the
  262,057,897-row zero suffix. The selected-key commitment check passed in
  117.722 s for all 1,188 coefficients. Opening evaluation remains separate
  from full F′ parent hash and assignment acceptance. PiCCS
  **Conformance-closed** is still open; no identity was re-pinned.

  The same signed input and proof pass all four mutation groups in both Rust
  engines: 56 common prior-point limb changes (108.675 s), 282 statement
  changes (185.610 s), 843 output/shape changes (116.935 s), and 562 proof
  changes (283.236 s). Each invocation first compares every positive phase
  result with Lean. The separate `point-mutations` group requires the nonzero
  running fixture; the zero-running fixture retains its parent-hash point
  rejection gate. All invocations stayed within the 300-second cap.

  The parent connection has now been checked with these same serialized
  claims. The native pilot target was migrated to schema 2 and rebuilds all
  running fields in `Lifecycle.XOut.serializeRunning` order. The zero-running
  case passed in 117.913 s, including its build and existing pilot/mutation
  checks. The signed case failed in 67.801 s at `prior public-input binding`.
  Its recomputed parent digest is
  `[6295585170289862968, 2824516971530000521, 186639149412602767, 13966806424625331640]`;
  its unchanged fresh public input encodes
  `[1490222372662734711, 16390513960815826474, 4802270639494233510, 11297521189134956078]`.

  The two tests now share only the caller-input assembly in
  `crates/nightstream-fprime/tests/support/pi_ccs_parent.rs`. The canonical
  row evaluator remains separate from witness generation. On the final
  shared-input code, `external_positive_pi_ccs_prefix` passed the zero-running
  case in 224.315 s, including compilation (193.31 s in the test). It checks
  all 19,936,967 physical rows, all 3,864,823 logical prefix rows, and the
  owner mutations. The previous invocation took 224.407 s; these are single
  observations, not a speed claim.

  The same row target failed on the signed case in 37.260 s at canonical
  physical row 7,312,120 in the prior-state hash. It did not reach the logical
  row check. This is a failed positive-parent gate, not a conformance pass.
  The earlier paper-phase result and opening checks remain valid within their
  stated scope. A positive signed parent needs a matching fresh public input,
  bounded fresh opening, commitment, proof, and the required dependent checks.
  PiCCS **Conformance-closed** and the current-cut independent review remain
  open. PiRLC `InputBinding` is unchanged.
- `generate_pi_ccs_fixture fold-base` constructs the packed integer fold of
  the checked nonzero fresh carrier with sixteen zero running openings. It
  consumes all seventeen samples from the existing sampler, starting at the
  accepted PiCCS outgoing state. The first sampled rotation acts on each
  54-coefficient block in `Z[X]/(X^54 + X^27 + 1)`. Every entry of that
  rotation matrix is checked against the integer kernel before the full run.
  The sampler, PiRLC relation, and frozen `InputBinding` are unchanged.

  The focused fixture target passed all six tests in 15.502 s including its
  build (0.03 s in the tests). The new checks cover every ring basis pair and
  all 131,071 integers strictly between `-2^16` and `2^16`. The sixteen signed
  digits recompose exactly; out-of-range values fail without a fallback.
  These are fixture-arithmetic checks, not a phase conformance claim.

  The full folded carrier was generated in 49.468 s including compilation,
  34.733 s in the binary. It contains 256,532,184 signed integer coefficients
  in a 513,064,368-byte external file. The maximum magnitude is 53; the exact
  sampled matrix bound is 70, below the fixed `B = 65,536` bound. Only digit
  positions 0 through 5 are nonzero in this carrier; all sixteen positions
  remain part of the profile. The 270 public coordinates recompose from their
  sixteen signed digits. The metadata uses signed construction integers and
  is not a canonical protocol input or a source of semantic authority.

  The child commitments and separate `Eval_K`/`Eval_A` claims are checked
  below. The matching recursive witness passes the complete row checks below;
  the next honest PiCCS proof remains open. These fixture checks do not claim
  PiRLC conformance on the new transcript state. No relation, row, footprint,
  or identity changed.
- `generate_pi_ccs_fixture child-commitment` derives each native signed digit
  and uses the complete selected commitment key. It preserves all native
  alignment coordinates. The six nonzero child runs passed in 37.783–166.314 s
  each, including any compilation. `zero-child-commitments` checked every
  parent coefficient and its high digits before emitting the ten zero
  children; it passed in 83.272 s, including compilation (55.438 s in the
  binary). All sixteen children remain explicit in the resulting data.

  `check-child-commitments` replayed all seventeen samples and the outgoing
  state, then checked every weighted commitment coefficient and all 270
  public coordinates against the actual folded fresh claim. It passed in
  34.645 s. The native generator and recombination checks are separate from
  the independent opening check.

  `crates/nightstream-fprime/tests/pi_ccs_child_commitments.rs` reads the raw
  folded carrier and Lean setup. It uses the independent indexed key
  expander and signed integer convolution, with no fixture-generator ring
  routine. The full check passed all 22 rows for all sixteen children:
  19,008 commitment coefficients and all child public coordinates, in
  239.118 s including compilation (225.686 s in the test).

  An earlier complete run timed out; a first work-distribution change then
  hit a stack overflow. Neither is passing evidence. The final evaluator
  shares key expansion across children and uses heap accumulators with
  Rayon's work distribution. It retains the full comparison under the
  300-second cap. The external runner also uses lean-graph's host lock,
  process preflight, and explicit wall and monotonic deadlines.

  Canonical mutations add two to a coefficient of child 0 and subtract one
  from the same coefficient of child 1. All 1,188 weighted sums remain
  unchanged. The independent checker rejected the altered opening at row 0
  in 47.128 s (expected exit 101). Thus weighted recombination is not counted
  as proof of each child opening. All generated data and execution records
  remain outside the repository. PiCCS **Conformance-closed**, the next
  nonzero-running proof, and the independent phase review remain open.
- `generate_pi_ccs_fixture child-family` shares one matrix traversal across
  all sixteen actual digit children. It computes separate Pad and matrix
  families from the selected package and retains all 256,532,184 native
  coordinates, including the 36 nonzero alignment coordinates. The preceding
  one-child/all-families attempt timed out at 300.119 s and produced no
  complete claim; it is not passing evidence.

  `crates/nightstream-fprime/tests/pi_ccs_child_evaluations.rs` independently
  decodes the canonical matrix program and evaluates the full folded carrier.
  It checks all sixteen children and all 54 extension coefficients in each
  family: 12,960 extension coefficients, or 25,920 field words in total.
  Matrix `A13` is checked as zero throughout; Pad remains a separate family.

  | Family | Generation (s) | Independent values and mutations (s) |
  |---|---:|---:|
  | K | 50.645 | 43.723 |
  | A0 | 48.275 | 70.172 |
  | A1 | 46.996 | 62.166 |
  | A2 | 55.843 | 103.187 |
  | A3 | 88.927 | 116.335 |
  | A4 | 176.788 | 142.185 |
  | A5 | 106.910 | 193.833 |
  | A6 | 78.925 | 109.197 |
  | A7 | 81.859 | 95.654 |
  | A8 | 65.204 | 119.614 |
  | A9 | 100.808 | 97.177 |
  | A10 | 66.855 | 104.329 |
  | A11 | 79.544 | 101.275 |
  | A12 | 75.558 | 125.970 |
  | A13 | 74.610 | 78.485 |

  Each family check rejects 1,728 changed field words, 1,728 noncanonical
  words, 15 cancelling adjacent-child pairs, a missing child, and a truncated
  ring for each child. These mutations use the same checker that compares
  claimed values with the independently computed values. The cancelling
  pairs preserve their weighted sum and still fail the individual check.

  The independent signed ring product now accumulates exact integers before
  field reduction. Its fixed-degree bound is checked at compile time.
  `pi_ccs_opening_arithmetic` passed all three tests against the original
  modular evaluator: every signed basis product, dense field boundaries in
  both extension components, and an out-of-range digit rejection (9.299 s
  including compilation). An initial syntax failure is recorded separately.
  On the identical A4 inputs, total time changed from 201.194 s to 142.185 s;
  the interval after input loading changed from 131.694 s to 73.322 s. Those
  intervals include row processing and mutations. Loading and compilation
  also varied, so these are single-run observations. The A0–A3 records retain
  the original modular evaluator; the other table entries use the checked
  integer implementation. Source copies and all input/result records remain
  outside the repository.

  `pi_ccs_child_recomposition` passed in 0.601 s. It checks all 810 combined
  extension coefficients against the exact preceding PiCCS output, including
  its point and outgoing transcript state, and keeps the same child
  commitments and public inputs across all families. The complete five-field
  running-claim packet is assembled outside the repository for the next
  caller fixture. It adds no row, column, relation, or identity. The recursive
  assignment check below uses this packet. The distinct-child PiCCS proof
  below uses these claims. Current independent phase review remains open.
- `Export.Stage1.RecursiveStepFixture` constructs the next caller packet from
  the actual base output, accepted PiCCS proof, and sixteen checked child
  claims. It uses the existing phase execution and checked public split.
  The PiCCS result and base fixture remain byte-identical after sharing that
  execution. Focused builds and emission pass. Six constructor cases reject
  changed fresh public input, prior running state, proof, child point, child
  public input, and noncanonical JSON; none emits an output file.

  The candidate `recursive` check independently binds the packet to those
  exact inputs. It passes all 29,225,729 physical rows, all 256,532,147 logical
  coordinates, 37 alignment zeros, all 6,377,559 active logical rows, and the
  zero suffix in 195.104 s including compilation. The first invocation failed
  because the new Rust checker grouped proof output families incorrectly;
  its corrected source follows Lean's per-source K-then-A order. The failed
  run is retained. Base-only guards remain in the separate base entrypoint.

  `recursive-mutations` first checks the positive physical assignment, then
  changes raw child commitment, `Eval_K`, and `Eval_A` columns after witness
  generation. The independent evaluator rejects all three cases in 87.512 s
  including compilation, at rows 28,872,943, 28,883,163, and 28,883,919. A
  separate changed-commitment caller packet fails the witness assertion at
  row 28,869,721; that earlier rejection is not independent-row evidence.

  The exported witness IR then produces the bounded fresh carrier, all
  canonical scalar matrix images, and its selected-key commitment in
  282.279 s. All data and logs remain external. Its independent commitment
  and opening checks pass below. The distinct-child proof uses separate
  cubic norm terms. No phase status, relation, footprint, or identity is
  promoted or changed.
- The fixture oracle now retains each actual child's cubic norm residual in
  `ProtocolPolynomial.normAtMessage` source order. Initial child values stay
  in signed bytes until their first fold; all sources use the same challenge.
  The linear Pad/matrix prefix combines their gamma-weighted values. The
  existing signed-copy shortcut remains limited to its earlier fixture case.

  `nifs_pi_ccs_fixture_oracle` passes 9/9 (16.853 s including compilation).
  The distinct-source case checks all ten interpolation values and an
  extension-field value through 28 folds against direct finite MLE sums.
  It includes a child beyond the fresh prefix, strict digit rejection, and
  equality of the combined linear kernel with the indexed separate kernels.
  The initial build failed because the new method was not yet connected;
  the connected input path and focused checks passed without suppressing it.

  The actual child linear prefix matches Lean's initial claim (58.348 s).
  All 28 honest scalar rounds and the final scalar identity pass (24.542 s).
  The fresh Pad and 14 matrix families pass (217.772 s); all sixteen child
  output families use the resulting point. Five batches cover every family
  within the 300-second cap. K and A0 at the earlier identical inputs are
  byte-identical to the retained outputs: 70.198 s for the batch versus
  98.920 s for the earlier separate runs, including differing compilation
  work. These are single observations, not a kernel speedup claim.

  Lean accepts the assembled schema-2 input and proof (13.123 s). Rust
  `paper_exact` and `optimized` match its complete 17-source result (35.043 s).
  The same proof passes the parent-bound PiCCS assignment prefix, all
  19,936,967 physical and 3,864,823 logical prefix rows, and owner mutations
  (190.511 s). Its fresh source is the checked recursive caller above; its
  running sources are the actual digits, not substituted signed copies.
  Independent evaluation now passes all 17 Pad families and all 238 matrix
  families, with 54 extension coefficients in each family. Each invocation
  uses the same serialized input and accepted Lean result. Fresh-family
  checks take 37.549–88.185 s; sixteen-child checks take 56.042–86.782 s.
  All 918 output coefficients for zero matrix A13 are zero. The child
  evaluator binds the original commitments and public inputs at the input
  point, then checks each output family at Lean's new point. Each family
  rejects 1,728 changed words, 1,728 noncanonical words, 15 cancelling pairs,
  one missing child, and 16 truncated rings. The fresh CCS check passes all
  6,377,559 active rows and the zero suffix (79.016 s); the fresh commitment
  check passes all 22 key rows and 1,188 coefficients (143.595 s).

  Both Rust engines reject 562 proof mutations (265.371 s), 282 statement
  mutations (148.476 s), 843 output mutations (126.201 s), and 56 common
  prior-point mutations (103.485 s). The pilot check uses this exact
  recursive parent and passes in 91.555 s: 930 preimage mutations for each
  of its standalone and positive-prefix inputs, plus 274 public and nine
  malformed-input mutations per input. These are development diagnostics
  under the shared guard and 300-second cap. Complete logs, commands, source
  hashes, input manifests, and per-family times remain outside the repository.

  The complete input is 636,887 bytes, SHA-256
  `18a9e25604ecb575966830c64976ea4797420238b18e5cca9ec18e50c88f9e1b`.
  This hash records the checked bytes; it is not protocol authority.
  The current lean-graph map still selects the earlier signed-copy opening
  target. Its reviewed registrations must cover these separate fresh and
  actual-child checks, their full raw inputs, and the recursive parent.
  The registered `piccs-public-assignment` checkpoint now passes on snapshot
  `eceded83ab806ac3db3fdcf1dd88dc0af31c8fdd6f0064e727a2c768721fef63`.
  Declaration metadata and exact-target evidence are current diagnostics.
  The target build audits the pilot, PiCCS, and public-boundary witnesses with
  only `propext`, `Classical.choice`, and `Quot.sound`. The graph confirms the
  proof path from row-derived preimage representation through both hash
  observations to the public PiCCS target. The statement still leaves the
  selected context and full typed Stage 1 step open.

  Lean-graph records 903.302 s for the metadata gate and 119.108 s for the
  exact-target gate. The latter reuses the checked build cache: its build
  command takes 4.980 s and its acceptance file takes 9.532 s. These command
  times are not the whole gate time. A review request is bound to the same
  source, policy, checker, exact target, premises, argument, and parent use.
  No review decision or approved checker result was created. Required branch
  coverage, exact-cut reviews, and accepted evidence remain open. PiCCS is not
  **Conformance-closed**; PiRLC `InputBinding` remains frozen.
- `generate_pi_ccs_fixture family ... all ...` computes the full Pad and
  14 matrix output families from the same carrier and actual package rows.
  Weighted matrix rows are aggregated before the linear bar transform and
  ring product. Each constant coefficient must equal the separately folded
  scalar output. All families passed in 193.881 s, including all 54 zero
  coefficients of matrix 13. The A5 family matches its separate 47.731 s
  execution. Package loading is shared across the complete family run.
- The registered `nifs_pi_ccs_fixture_oracle` target passed 4/4 in 13.407 s
  including its build. The zero-running and signed-running cases compare
  cached completion sums with direct finite MLE sums through all 28 folds,
  including singleton/odd/even prefixes and extension-field challenges.
  The signed case evaluates each source's norm separately and includes a
  nonzero running term and prior point. The ring test checks all 54
  coefficients against independently weighted basis products and monomial
  reduction. The fourth test checks the two coefficient-weighted running
  kernels against full ring products at every native lane. These are
  arithmetic checks, not positive phase fixtures.
- The resulting positive PiCCS input is 146,346 bytes, SHA-256
  `49585bfbd8a5df953d55705e509d0a68fe5bf924c8b15e3ed007f8a209d65850`.
  It has one checked nonzero fresh opening and sixteen literal zero running
  openings. The fresh commitment has 1,188 nonzero words, 279 of the 280
  round coefficients are nonzero, and all 54 fresh Pad coefficients are
  nonzero. Real zero coefficients are retained in the matrix families.
  Executable Lean accepted it in 10.812 s. `check_pi_ccs_input ... accept`
  passed in 34.038 s: Lean, `paper_exact`, and optimized agree on every
  required phase result field, including all transcript states and complete
  output families. This is current positive value evidence. It does not
  substitute for the independent opening and exact-cut review gates.
- `external_positive_pi_ccs_prefix` passed in 211.585 s, including its build
  (183.85 s native). The same positive input and proof drove the exported
  witness IR. The independent evaluator checked all 19,936,967 physical
  prefix rows and 3,864,823 logical prefix rows, and compared each used
  production logical coordinate with its independent decoder. It rejected
  one generated-column mutation, 12 row-owner mutations, 12 nonempty
  column-family mutations, and three public-segment mutations. Two declared
  column families contain no columns. Both pilot hash inputs use the checked
  base output preimage in this prefix test; this does not claim a complete
  next Stage 1 step.
- `check_pi_ccs_input ... proof-mutations` passed in 249.770 s. Both
  `paper_exact` and optimized first passed the complete positive comparison,
  then rejected all 562 proof mutations: two malformed proof shapes and
  every limb of every coefficient in all 28 rounds. No partial or timed-out
  run is counted as this evidence.
- `check_pi_ccs_input ... statement-mutations` passed in 143.826 s. Both
  verifiers rejected 282 changes across the shared digest, inconsistent
  running points and digests, padding, all 16 indexed running commitment,
  public-input, Pad and 14 matrix families, and the fresh statement.
  The first attempt failed in 34.843 s because it incorrectly required a
  common-point change to reject inside PiCCS when all running openings are
  zero. Such openings are valid at every point under the paper formulas.
  The accepted digest schedule assigns that binding to the parent pilot;
  its current preimage-mutation gate must reject the changed point with the
  old public digest. No relation change was made to force rejection.
- `check_pi_ccs_input ... output-mutations` passed all 843 cases in
  98.518 s. Both verifiers rejected each indexed commitment, public input,
  Pad and matrix output family, all retained-point coordinates, all digest
  lanes, and ten malformed output shapes. The first sequential run timed
  out at 300.038 s and is a failed gate. Independent source cases now run
  in parallel within one process, using the runtime's available workers;
  the retry preserved every case and the 300 s cap.
- The current-context Lean pilot fixture was emitted through `validate.sh
  pilot-parity` in 18.079 s. `external_current_pilot_and_positive_pi_ccs_prior`
  passed in 96.620 s including its build (88.28 s native). It checks the
  complete standalone Lean/Rust pilot result and the exact base-output
  preimage used by the positive PiCCS input. Each input rejects 930 preimage
  mutations, 274 public-word mutations, and nine malformed encodings.
  Every one of the 56 shared-point limbs is covered in each preimage. This
  closes the parent-hash rejection left separate by the zero-running
  PiCCS statement test.
- `pi_ccs_opening_values` adds a separate evaluator for all 54 opening
  coefficients. It uses the independent canonical Lean row decoder and
  Goldilocks arithmetic, a direct quadratic-extension implementation, the
  closed Phi81 dual basis, and polynomial convolution/reduction. It does
  not use the generator's bar matrix, ring product, or row expander. The
  focused basis and tensor checks passed 2/2 in 12.460 s including the build.
  Full current `Eval_K` and all 14 `Eval_A` families pass on the
  256,532,184-coordinate raw carrier. Every run checks all 54 coefficients
  and retains the carrier-alignment positions. The separate row checks and
  opening checks share the independent canonical row decoder; they are not
  two independent implementations of that decoder.
- The same raw cached carrier passed all 6,377,559 active CCS rows and the
  262,057,897 implicit zero rows in 75.552 s, including its build. The
  independent evaluator checks strict `b=2` coordinates, the 270-word public
  projection, and the 37 zero alignment coordinates. It calls no production
  witness generator, row expander, or constraint evaluator.
- The independent selected-key commitment check passed in 124.939 s,
  including its build (111.36 s native). It compares the complete 73-word
  Lean setup authority with the package binding, checks the Lean RFC block
  and indexed coefficient vectors, and recomputes all 22 commitment rows
  with separate ChaCha20, 256-bit reduction, and signed-integer convolution.
  All 1,188 coefficients match the positive input. C0 alone passed first in
  51.976 s including its build; its 3.43 s measured calculation justified
  the full 22-row run with one package load. No commitment backend or proof
  backend is involved.

The independent opening-family run times in seconds are:

| Family | Time | Family | Time | Family | Time |
|---|---:|---|---:|---|---:|
| `Eval_K` | 41.978 | A0 | 57.226 | A1 | 56.409 |
| A2 | 56.653 | A3 | 56.860 | A4 | 68.136 |
| A5 | 78.558 | A6 | 56.374 | A7 | 56.593 |
| A8 | 56.257 | A9 | 56.429 | A10 | 56.255 |
| A11 | 56.555 | A12 | 56.454 | A13 (zero) | 56.350 |

These use `cargo test -p nightstream-fprime --release --locked --test
pi_ccs_opening_values external_positive_opening_family -- --ignored --nocapture`,
with `RUSTC_WRAPPER=""`, explicit external candidate paths on stdin, and one
300 s cap per invocation. No relation identity changed during these checks.

- The old round, ordinary transcript, and payload allocations still exist in
  the 38-block assignment layout. Remove them only with the updated cumulative
  ledger and transport theorems. The first three endpoint pin families also
  need a proved redundancy check. Current rows and dimensions are unchanged.

- `PiCCSAssignmentSoundness.rowsZero_implies_arithmeticSpecs` builds and is
  audited. Accepted ordinary rows imply all eight arithmetic child contracts
  in the environment decoded from the same arbitrary assignment. Preimage
  framing and NextPreimage reuse this decoder. No source packet or canonical
  coordinate-encoding premise is supplied.
- `PiCCSDecodedTranscript.rowsZero_implies_indexedSemantics` and
  `rowsZero_implies_traces` derive the complete indexed transcript and its
  four phase slices from actual parent payload forms and accepted Poseidon
  rows. `PiCCSDecodedEndpoints.sourceForm_eval` reads all four final states
  from the same decoder. The outgoing eight words use their existing owned
  block; the decoder previously returned zero for these unclassified source
  coordinates. Classified arithmetic forms are unchanged.
  `rowsZero_implies_endpointStates` uses the existing 32 endpoint rows, and
  `rowsZero_implies_transcriptSpecs` derives all four transcript contracts.
  The corrected endpoint build and axiom audit passed in 8.072 s. Two initial
  file checks exposed proof elaboration errors; the corrected endpoint file
  passed in 3.965 s. No row or column was added for this connection.
- `PiCCSDecodedPhase.rowsZero_implies_specHolds` joins the eight arithmetic
  and four transcript contracts through the existing opaque phase assembler.
  `selectedRowsZero_implies_phaseHolds` projects the required row families
  from the complete Stage 1 structural plan and uses `plan_fixedPoint` to
  select the exact key-facing relation. It concludes the exact PiCCS
  `PhaseHolds` predicate in decoded values for arbitrary accepted assignments
  with the required one-coordinate. It assumes no `RawValues`, `Encodes`, or
  `Represents`. The corrected selected-row build and axiom audit passed in
  7.646 s with only `propext`, `Classical.choice`, and `Quot.sound`. Earlier
  assembler checks exposed missing namespace qualifications; these failed
  runs remain in the validation record. The full Stage 1 assignment theorem,
  exact-cut review, and PiCCS **Conformance-closed** status remain open.
- Fresh canonical emission after the decoder change passed in 167.917 s.
  An exact byte comparison passed against the same 38-block candidate used
  by the matrix, independent assignment, valid opening, parity, and mutation
  gates above: 128,464,976 bytes, SHA-256
  `07a18d8a24b064ae008660f08e75ac3ee0d91ba74482903b72e8ba0e7db97627`.
  Rows, columns, canonical bytes, and relation identity are unchanged.
  The prior executed value and matrix evidence is retained. This byte
  comparison adds no independent semantic or matrix-conformance claim.
- The full `validate.sh axioms` gate passed after these proof changes:
  3,658 jobs, 42.856 s. The repository boundary gate passed in 15.084 s.
  Validation used the external copy and output sinks under the stated caps;
  no Rust implementation or fixture changed in this proof slice. The
  17:48:57 UTC preservation record found no protected-file, staged, or scoped
  source-line violation. It retained the same two previously reported frozen
  generated-file differences, so that check exits 1. No controlled command
  built or maintained the frozen package, and no commit, stage, reset, stash,
  removal, restore, or discard occurred.
- `PilotDecodedHashes.rowsZero_implies_hashFacts` derives both Poseidon2
  hashes from arbitrary accepted pilot rows and the actual preimages used by
  the PiCCS decoder. The shared sponge proof now accepts exact input-form
  equalities; its existing encoded caller supplies those equalities through
  its prior proofs. The focused build and axiom audit passed in 46.839 s.
- `PilotDecodedEnvironment` reads each pilot arithmetic location from its
  compiled form and retains the PiCCS view elsewhere. It proves that neither
  preimage interval is overwritten and that all 1,330 ordinary rows hold in
  this same environment. The first file check failed on a reducibility limit
  and two match rewrites; the corrected proof uses the existing length theorem
  and passed in 9.830 s. No proof-engine limit was raised.
- `PilotDecodedPhase.rowsZero_implies_specHolds` combines the actual hash
  equations, ordinary rows, and eight digest-binding rows through the existing
  pilot assembly proof. `selectedRowsZero_implies_specHolds` projects these
  facts from the selected complete Stage 1 rows. Both apply to arbitrary
  assignments with the enforced one-coordinate and have no `RawValues`,
  `Encodes`, or `Represents` premise. Their focused build and axiom audit
  passed in 79.322 s. The full `validate.sh axioms` gate then passed in
  76.607 s, 3,661 jobs. Only `propext`, `Classical.choice`, and `Quot.sound`
  occur in the new proof audits. The complete `StepHoldsFor` connection and
  exact-cut conformance review remain open.
  Fresh canonical emission after the pilot changes passed in 144.418 s and
  matched the same 128,464,976 candidate bytes exactly. The repository boundary
  gate passed in 16.091 s. The 18:24:18 UTC preservation check found no
  protected-file, staged, or source-line violation; only the same two earlier
  frozen generated-file differences remain. No new relation identity or
  repository artifact was pinned or written in this proof slice.
- The pilot/PiCCS decoded views now have proved agreement on every prior
  preimage word, next-preimage word, and fresh public-input word:
  `PilotDecodedEnvironment.priorWord_agrees`, `outputWord_agrees`, and
  `priorPublic_agrees`. Their build and axiom audit passed in 16.128 s.
- `StateDecoder.priorRepresents` and `outputRepresents` connect canonical
  decoded words to the existing pilot interfaces. The older canonical-packet
  helpers now use these same proofs. In
  `ActualPreimageFraming.rowsZero_implies_preimageRepresentations`, accepted
  rows derive both representations for the actual hashed inputs; no caller
  supplies a representation or coordinate-encoding premise. The corrected
  build and axiom audit passed in 10.125 s. The first attempt exposed an
  ambiguous namespace and a proof-inference reduction limit; explicit
  arguments resolved them without raising proof-engine limits.
- `ActualPreimageFraming.rowsZero_implies_contextKeys` proves that the two
  actual preimages carry the same context key.
  `ActualNextPreimage.rowsZero_implies_decodedHeaders` proves preservation of
  the decoded initial state and equality between the actual next counter
  word and the field encoding of the decoded prior counter plus one. It does
  not assert natural-number non-wrap of the output decode. The combined
  build and axiom audit passed in 10.231 s.
- `ActualApplicationStep.selectedRowsZero_implies_decodedStep` derives the
  application equation on the decoded current states from the selected
  complete Stage 1 rows and the enforced one-coordinate. Its witness comes
  from the existing owned application forms. The corrected build and axiom
  audit passed in 7.875 s; the first attempt needed explicit bounds and
  geometry arguments. The final hash equations, typed NIFS input/output
  connection, and full `StepHoldsFor` theorem remain open.
  The full axiom gate after these changes passed in 20.326 s, 3,662 jobs;
  the repository boundary gate passed in 15.335 s. The 19:22:22 UTC
  preservation check found no protected-file, staged, or source-line
  violation. It still reports the same two earlier frozen generated-file
  differences. No new relation identity was pinned.
- `ActualRunningTransition` evaluates the existing source map on an arbitrary
  assignment. The ordinary-row compiler theorem then derives the physical
  transition contract, without `Encodes`, `RawValues`, or `Represents`.
  Its state/output form proofs connect the transition to the actual pilot
  preimages. The selected public-boundary theorem proves HyperNova's base
  state equality and complete default running output. It derives the one
  cell from the public marker; the decoded zero counter is the branch case.
  The core, readout, and base builds pass in 4.150, 4.066, and 4.814 s.
  The first build failed on a reserved Lean identifier, which was renamed.
  All eight exported theorems pass the focused axiom audit (4.564 s), using
  only the allowed logical axioms. The boundary gate passes in 13.299 s.
  This assembler proof work uses the accepted phase-local continuation
  decision. PiRLC `InputBinding` and all row, layout, profile, and identity
  definitions are unchanged. The full `StepHoldsFor` theorem, selected-key
  connection, independent review, and conformance statuses remain open.
- `ActualStep.selectedRowsAndPublic_imply_baseStep` proves the complete
  decoded `StepHoldsFor` base branch for arbitrary accepted assignments.
  Its typed state, witness, running output, and digest use the actual
  assignment. The prior and next preimage equalities connect both typed
  hash calls to the existing row-derived observations. `decodedFresh` uses
  the PiCCS leaf's inputs; `withDecodedPiCCS` reads its proof fields and
  preserves the still-unconnected PiDEC template fields.
  `selectedRowsAndPublic_step_iff_baseOrNifs` derives the HyperNova envelope.
  `selectedRowsAndPublic_imply_piCcsCheck` then discharges the concrete NIFS
  PiCCS Boolean. `selectedRowsAndPublic_step_iff_baseOrPiDec` leaves exactly
  the PiDEC check over the verifier-computed PiRLC parent and equality of
  the computed running output for a nonzero counter. Neither equation is
  assumed or proved by that equivalence. No representation, generated
  assignment, application-correctness, or NIFS-correctness premise is added.
  The current focused build passes in 4.157 s and the axiom audit in 4.376 s,
  with only the allowed logical axioms. The recursive equations, full
  decoded-step theorem, canonical context connection, and independent
  review remain open. No phase status or row/layout/profile/identity
  definition changed.
- `ActualPreimageFraming.rowsZero_implies_actualPreimageCanonical`,
  `ApplicationAssignmentSoundness.rowsZero_implies_step`, and
  `ActualNextPreimage.rowsZero_implies_actualNextPreimage` build. They derive
  the actual hashed preimage format, application step, counter field
  increment, and four initial-state equalities from arbitrary accepted rows
  and the fixed one-coordinate. They assume no `Encodes` or `Represents`.
  Their current axiom audit passes with only the permitted logical axioms.
  These are necessary parts of the full step proof, not a full closure claim.
- `ActualPiDEC` derives its phase predicate for every accepted selected
  assignment and actual public projection. Its decoder reads PiDEC arithmetic
  forms and preserves the PiCCS parent point. It assumes no raw packet,
  `Encodes`, `Represents`, or semantic phase result. The focused build passes
  in 4.166 s and all seven exported theorem audits pass in 4.012 s. This
  establishes the local phase connection; it does not identify the separate
  retained PiDEC parent allocation with the computed PiRLC output.
- The detached-application regression now independently validates the changed
  application's rows before the splice and requires a nonzero residual in
  the application row range after the splice. The current candidate rejected
  the splice at row 6,377,546 in the 164.077 s run recorded above.
- `Layout.ProductionRelation.CcsOpening` built in 3.714 seconds; its explicit
  axiom audit passed in 3.662 seconds. `Plan.rowsZero_implies_paperCcs` and
  `Plan.rowsZero_implies_freshHolds` connect literal plan rows, bounded
  coordinates, and public projection to an opening with the actual key
  commitment. This generic CCS result does not prove the F′ lifecycle
  semantics of a faulty plan.
- `Lifecycle.PiCCS.v1_1.ZeroRunningPolynomial` built in 4.322 seconds and
  `Export.Stage1.PiCCSInputCheck` in 5.266 seconds. The polynomial's current
  audit passes. The current positive executable result is recorded above.
- `PrefixFold`, `SparseEvaluation`, `ZeroRunningOracle`, and
  `NumericCompletionSum` build and are audited. They connect sparse execution,
  exact carrier completion, and numeric traversal to the canonical tables.
  `ZeroRunningRoundSum.roundSum_eq_sumCompletions` builds and is audited for
  every available round and every trial value. Its target is the exact
  production-key polynomial, with matrix and norm sums over separate proved
  support extents. The full ring outputs, honest Rust generator, and positive
  result checks now pass as recorded above; PiCCS **Conformance-closed**
  remains open.

- `Lifecycle.PilotZeroRunning.defaultRunning_holds` is built and audited. It
  proves valid zero openings for the default running inputs used by the
  standalone pilot fixture.
- `validate.sh pilot-parity <vk0> <vk1> <vk2> <vk3> <path>` takes the four
  canonical verifier-context words and a sink. Generation completed in
  21.582 seconds, including the build, and matched the 199,534-byte fixture
  exactly. The Rust context test independently binds those words to the
  sealed package.
- `validate.sh base-step-fixture <vk0> <vk1> <vk2> <vk3> <path>` emitted
  the 367,974-byte base caller fixture in 28.619 seconds. Its zero inner PiCCS
  proof is a base-only placeholder, not a valid PiCCS conformance fixture.
- The latest `validate.sh static` passed in 11.506 seconds. Focused
  `validate.sh build` targets passed for
  `NightstreamFPrime.Lifecycle.Stage1.Formal` in 9.757 seconds and
  `NightstreamFPrime.Layout.Stage1.PreservationClosure` in 53.349 seconds.
  The latest `validate.sh axioms` passed in 81.227 seconds with only the
  permitted axioms. No `validate.sh all` run is claimed for this cut.
- `validate.sh emit-poseidon2-hash-chain-v1 <path>` completed in
  184.532 seconds. `cmp` confirmed that the freshly emitted 126,436,452-byte
  package matched the stored artifact before the application repair. Those
  matrix comparisons remain evidence for the stored bytes only; the changed
  logical relation requires fresh exact matrix checks.
- `AccumulatorSemantics.phases_imply_holds` proves that the complete
  PiCCS → PiRLC → PiDEC verifier graph equals `Accumulator.Holds`.
- `AccumulatorPackage.circuitPackage_implies_accumulatorHolds` derives that
  result from the unchanged canonical package rows.
- `Lifecycle.Stage1.circuit` is the sole eight-child parent;
  `Lifecycle.Stage1.soundness` and `Lifecycle.Stage1.circuit_coverage` prove
  its parent result and exact child coverage.
- `AssemblerApplicationCompleteness.completeStage1` includes the opaque
  `NextPreimage` child. `AssemblerInputs.nextPreimage_parent_wiring` binds it
  to the pilot preimage columns, and
  `PreservationClosure.physical_implies_compactNextPreimage` retains its
  specification in the sole logical root. The existing five physical rows
  and zero added columns are unchanged (`Lowering.nextPreimageRows_length`,
  `Lowering.nextPreimage_noFresh`). The final physical ledger remains
  29,225,729 rows and 29,344,425 columns.
- `PreservationClosure.physical_implies_compactSpec` derives all eight child
  specifications from the physical rows.
  `PreservationClosure.physical_implies_stepHoldsFor` and
  `PerApplicationSoundness.packageRows_imply_stepHoldsFor` still require their
  respective `Represents` premises.
  `Poseidon2HashChainV1Closure.rowsZero_implies_stepHoldsFor` covers only
  `(bound raw).assignment`. Reconstruction of the decoded step from an
  arbitrary satisfying emitted assignment remains open.
- `PerApplicationMatrixProgramSemantics.matrixProgramExact` and the
  `Poseidon2HashChainV1MatrixRows` theorems connect the final package program
  to all 14 Lean-authored matrices, including zero slot 13 and padding.
- CI is configured to run `formal/nightstream-fprime/scripts/validate.sh all`.
  No remote CI run is claimed on this cut.

The speed path preserves the generic specifications. It uses proved direct
symbolic states, shared PiCCS operation packets, linear Horner construction,
streamed row classification, direct typed serialization, and cached immutable
punctuation buffers. The package formats remain the schema authority. The
emitter uses `LEAN_NUM_THREADS=10` and writes one canonical order.

Rust evidence on this cut:

- `pilot_lean_nonzero_parity` passed all five tests: current verifier context,
  valid zero running openings, complete Lean/Rust pilot results, preimage
  mutations, public-value mutations, and malformed encodings.
- `sealed_package_checks_the_standalone_pilot_assignment` passed in
  177.435 seconds total; the native test took 147.94 seconds. It compares every
  production logical assignment coordinate used by the pilot rows with an
  independent lift and independently evaluates all 14,623,730 physical pilot
  rows and 2,323,138 logical pilot rows. It rejects 274 public-word mutations
  and 10 generated-value mutations, plus two physical-row and two
  physical-column mutations across the two hash owners. Four logical-row and
  four logical-column mutations cover both Poseidon blocks, `pilotOrdinary`,
  and `pilotDigestBinding`. The independent evaluator rejects reads from the
  unavailable proof gap and non-pilot suffix; these checks do not claim
  complete PiCCS or lifecycle coverage.
- `RUSTC_WRAPPER="" cargo test -p nightstream-fprime --release --locked
  --test base_step_assignment -- --ignored --nocapture` passed in 175.017
  seconds total and 158.95 seconds
  native. The real exported witness program produced one nonzero base-step
  assignment. Independent checks accepted all 29,225,729 physical rows,
  compared all 264,627,433 production logical coordinates with an independent
  lift, checked each coordinate is `0`, `1`, or `-1`, and accepted all
  6,377,559 active logical rows plus zero padding. The public projection is
  `encHash` of the base output digest, with 53 zero carrier-alignment
  coordinates. This row test does not itself check a commitment or an honest
  PiCCS proof and does not close PiCCS or full Stage 1 conformance.
- `base_step_assignment_has_a_selected_key_commitment` passed in
  160.124 seconds on the stored identity and key. It computed the actual
  indexed Ajtai commitment of the complete base carrier outside the circuit.
  Indexed-key stream and signed-unit commitment primitive tests passed.
  Executable Lean commitment parity and an honest PiCCS proof remain pending;
  this old-key commitment is not evidence for the repaired relation.
- exact final 14-matrix equality passed all 6,377,559 logical rows. The
  per-matrix nonzero census is
  `[33616548, 4650801, 80774495, 97721533, 315532725, 1535263390,
  33006078, 1726758, 32220219, 30971178, 30233769, 31133970,
  30392685, 0]`;
- `sealed_package_generates_and_checks_the_current_pi_ccs_prefix` passed on
  the existing synthetic fixture in 211.371 seconds total and 183.80 seconds
  native. The test uses the actual production logical assignment transport,
  independently checks 19,936,967 physical prefix rows and 3,864,823 logical
  prefix rows, and rejects mutations for 12 row owners, 12 nonempty column
  owners, and three public segments. The fixture still lacks valid bounded
  openings. The separate current positive prefix run is recorded above;
- the existing PiCCS Lean / `paper_exact` / optimized complete-result and
  rejection tests use a synthetic accepting fixture. Their agreement does
  not establish a valid nonzero positive with bounded openings and canonical
  matrix evaluations. That PiCCS conformance gate remains open;
- the final-identity NIFS engine cross-check, transcript replay, missing
  `Eval_A` rejection, κ=22 setup parity, strict logical-relation header,
  matrix-identity mutations, and production-binding mutations passed their
  focused current-cut targets;
- the complete package-loader target uses only the current sealed package and
  passes all eight structural, encoding, transcript, and rejection tests.

The matrix comparator expands the Lean plan independently and compares every
entry of all 14 final low-norm matrices: row order, column index, canonical
coefficient, constant placement, and public mapping. The intermediate
physical R1CS A/B/C rows are a separate lowering check and are not this final
matrix evidence. The logical assignment evaluator does not call the matrix
builder, row expander, witness generator, or package constraint evaluator.

`Poseidon2HashChainV1Package` is the only normal-build public Stage 1 relation
boundary. It loads the allowlisted package and exposes only its Lean-authored
header, rows, binding, and witness program. The older compressed Stage 1
emitter, native constructors, decider, and Spartan lifecycle are unavailable
in normal builds. The package boundary has no proof-backend entrypoint because
no backend is approved. Its package loading is not evidence that Rust
implements the Lean relation; the semantic, matrix, assignment, and parity
gates above provide that evidence.

Superseded native v1.0 and radix-four integration targets are unregistered.
The compressed Stage 1 modules remain crate-private reference code and have no
normal-build public caller. Their only retained non-test consumer is the
crate-private, unapproved Stage 2 F′ prototype. Integration targets that need
that prototype's former public API are inactive, but their source remains for
future Stage 2 work. Current tests that serve the Stage 1 v1.1 contract use
separate `Eval_K` / `Eval_A` and the canonical nonempty running accumulator.

## Open authority and assembly edges

The existing proof path has an explicit encoding/representation boundary:

```text
final canonical package
  → exact 14-matrix LogicalRelation
  → ProductionKey.key
  → exact Lean-authored application Program and plan identity
  → StepHoldsFor (requires encoding/representation evidence)
  → recursive fixed point
  → rerun every PiCCS gate on the final identity
```

The accepted per-application-package decision requires the verifier to pin or
allowlist one final identity. `Poseidon2HashChainV1` is selected, its final
stored identity is still pinned. The confirmed application defect requires
the source repair and a fully checked replacement before any new identity is
pinned. PiCCS remains status open until the valid nonzero parity, complete production
assignment and mutation gates above pass, followed by independent review of
that exact source and artifact cut.

The remaining owner-ordered work is:

An independent raw-assignment mutation exposes a failed parent-wiring gate.
Zeroing retained `productGroup`, `productInput`, and `productOutput` blocks
leaves all 6,377,559 canonical rows satisfied, while 2,934 PiDEC parent values
differ from the product-output values with the same physical source labels.
Public inputs, phase proof, package, and identity stay fixed. The commitment
claim was not checked, and no full `StepHoldsFor` counterexample is claimed.
`pi_ccs_opening_values::external_rows_reject_detached_pi_rlc_product_blocks`
fails on this case (82.294 s). Shared source labels and canonical witness
generation do not enforce equality on arbitrary accepted assignments.

1. repair the retained PiCCS/PiRLC/PiDEC form wiring exposed by this regression,
   then finish the remaining arbitrary-assignment connections and current axiom
   checks; preserve the current candidate matrix, base-assignment, and
   detached-application evidence while its source, inputs, and identity
   remain unchanged;
2. preserve the completed independent opening and mutation gates for the
   current nonzero PiCCS fixture; finish the required phase proof connections,
   then obtain
   independent review of that
   exact source and artifact cut; rerun every affected gate, including valid
   nonzero results and cumulative handoffs, before any identity re-pin;
3. after PiCCS is conformance-closed, regenerate its exact PiRLC handoff and
   resume PiRLC work in owner order;
4. conformance-close PiRLC, PiDEC, application, and terminal evidence on one
   unchanged final package cut;
5. after separate owner approval of a production backend, connect that backend
   only to `Poseidon2HashChainV1Package` and execute the final production
   `prove → verify` obligation.

No backend is authorized on this cut. Backend acceptance cannot replace any
semantic or conformance gate in this file.
