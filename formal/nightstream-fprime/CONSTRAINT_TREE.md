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

- Pilot: validated candidate; status open on the current identity.
- PiCCS: the final-identity compiler and executable conformance gates are
  green. It remains formally **status open** until an independent reviewer
  approves this exact source and artifact cut.
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
- Application: the verifier-owned per-application `Program`, zero-copy
  four-word state layout, direct canonical lowering, plan identity,
  verifier-context component, verification-key binding preimage, and
  row-to-typed-transition theorem are kernel-checked. The generic final
  package constructor appends that plan, installs terminal metadata, binds raw
  static-key authority, proves exact preservation of every validated prefix
  row family, and proves package rows imply the selected `F`. The selected
  application is `Poseidon2HashChainV1`.
  Status open until the final exact-cut review and complete gate rerun.
- Stage 1: open. The seven-child opaque assembly order, derived offsets,
  aggregate footprints, arbitrary-witness soundness, and the generic
  package-row-to-`StepHoldsFor` theorem are kernel-checked. The concrete
  application, recursive fixed point, final `2^28` proof, deterministic
  closure, security-or-collision theorem, sealed package, and terminal
  metadata, sole logical parent `FormalCircuit`, complete physical
  preservation, and setup parity exist and pass their focused gates. The
  package-only Rust lifecycle, downstream current-cut reruns, and exact
  external review remain open.

The first 2026-09-04 independent Linux review rejected its cut because a
compressed Rust PiCCS path was reachable, its source fingerprint was not
reproducible, current audit comments were stale, and the retired loader suite
was red. A second review of manifest `f8697af3...` confirmed those repairs but
found a separate public route through the unapproved Nebula F′ prototype. This
working cut makes that Stage 2 module crate-private and keeps its dependent
integration targets inactive. It remains unapproved until an independent
reviewer checks its new exact manifest. No status below treats either rejected
review as approval.

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
Lifecycle/Stage1/Formal.lean                 ✓ sole seven-child FormalCircuit
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
| Commitment recomposition | 972 | 0 |
| `Eval_K` recomposition | 108 | 0 |
| `Eval_A` recomposition | 1,512 | 0 |
| Output binding | 0 | 0 |

The PiDEC verifier first checks centered parent magnitude `< 2^16`. Its
accepted path then uses the exact 16 signed radix-two digits. The computable
decision agrees with the semantic predicate. The accepted path cannot use
`fallbackDigit` for an out-of-range parent.

## Proved cumulative and final layout

The authoritative ledgers are:

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

The final recursive relation has 6,377,559 structural rows and logical width
264,627,433. Its Φ81 carrier width, and therefore its exact joint domain, is
264,627,486. This is below `2^28 = 268,435,456` with 3,807,970 points of
headroom. The outer terminal metadata adds no row or column.

## Canonical package cut

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

The verifier-owned identities are:

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
| Final sealed package | 126,436,452 | `1b0c9977724998b4b261001d503899439eedc9427d04950f00b4da5bf442427b` |
| Separate expanded package | 117,391,937 | `952a21ddca40e6223c7f8a0696ee5716eea741e6d6352a6b4e14cad1c634ef21` |
| Package-binding parity | 1,611 | `20e14acdcbe4e5df79a2eb1dce2a506c2ef458451b476cd1eaddfa17f7b59275` |
| PiCCS parity | 1,642,619 | `6406418a687884a68437f902d23bba7ab5e535c26961f0ad0b88a32e94f5e0e7` |
| PiCCS ownership | 1,649 | `9577bab3983538562d756a7d1458ba61549ae044400300466b097e9444b3d964` |
| Ajtai setup parity | 903 | `2279463a3b76aa273626d2028b62d4cb4d1ad30da945db723347050ec08cba51` |

The Pilot, PiRLC, PiDEC, sampler, application, and terminal parity artifacts
were not regenerated for this PiCCS-only cut. They are not current-cut
evidence.

## Exact conformance evidence

Lean evidence on this cut:

- `validate.sh all` passed the boundary checks, full `NightstreamFPrime` and
  `NightstreamFPrimeTests` builds, and every axiom target on this cut.
- `AccumulatorSemantics.phases_imply_holds` proves that the complete
  PiCCS → PiRLC → PiDEC verifier graph equals `Accumulator.Holds`.
- `AccumulatorPackage.circuitPackage_implies_accumulatorHolds` derives that
  result from the unchanged canonical package rows.
- `Lifecycle.Stage1.circuit` is the sole seven-child parent;
  `Lifecycle.Stage1.soundness` and `Lifecycle.Stage1.circuit_coverage` prove
  its parent result and exact child coverage.
- `PreservationClosure.physical_implies_compactSpec` and
  `PreservationClosure.physical_implies_stepHoldsFor` connect all physical
  rows to the exact Stage 1 semantics.
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

- exact final 14-matrix equality passed all 6,377,559 logical rows. The
  per-matrix nonzero census is
  `[33616548, 4650801, 80774495, 97721533, 315532725, 1535263390,
  33006078, 1726758, 32220219, 30971178, 30233769, 31133970,
  30392685, 0]`;
- the independent PiCCS evaluator passed all 19,936,967 cumulative physical
  prefix rows and all 3,864,823 final Lean logical prefix rows;
- physical mutations rejected for 12/12 row owners, 12/12 nonempty column
  owners, and 3/3 public segments. Statement binding and the SumCheck chain
  are the two proved zero-width column owners and have no assignment column
  to mutate;
- complete nonzero PiCCS Lean / `paper_exact` / optimized parity and
  rejection coverage passed 5/5. The fifth test distinguishes a valid-shaped
  terminal-identity `Ok(false)` result from malformed-output `Err` results;
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

The current Lean cut contains this authority edge:

```text
final canonical package
  → exact 14-matrix LogicalRelation
  → ProductionKey.key
  → exact Lean-authored application Program and plan identity
  → StepHoldsFor
  → recursive fixed point
  → rerun every PiCCS gate on the final identity
```

The accepted per-application-package decision requires the verifier to pin or
allowlist one final identity. `Poseidon2HashChainV1` is selected, its final
identity is pinned, the complete prefix-preservation chain is proved, and the
PiCCS gates above were rerun. PiCCS remains status open until the independent
exact-cut review approves this evidence.

The remaining owner-ordered work is:

1. obtain independent review of this exact PiCCS source and artifact cut;
2. after PiCCS is conformance-closed, regenerate its exact PiRLC handoff and
   resume PiRLC work in owner order;
3. conformance-close PiRLC, PiDEC, application, and terminal evidence on one
   unchanged final package cut;
4. after separate owner approval of a production backend, connect that backend
   only to `Poseidon2HashChainV1Package` and execute the final production
   `prove → verify` obligation.

No backend is authorized on this cut. Backend acceptance cannot replace any
semantic or conformance gate in this file.
