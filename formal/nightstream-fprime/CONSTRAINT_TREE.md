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
- PiCCS: all phase-local gates are green; formally **status open** under the
  owner decision until the final fixed-point edge and final-identity rerun.
- PiRLC: validated candidate; status open until an external reviewer approves
  this exact source and artifact cut.
- PiDEC: phase-local work is owner-authorized and all local gates below are
  green; status open until exact external review.
- Running-instance branch: compiler and phase-local conformance gates are
  green; status open as part of the cumulative package cut.
- Accumulator: the zero-row SuperNeo verifier composition and canonical
  package edge are kernel-checked; status open until the final package
  identity and fixed-point gates are rerun.
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
  metadata exist. The sole logical parent `FormalCircuit`, its complete
  physical preservation, setup parity, package-only Rust lifecycle, and exact
  external review remain open.

The official external review approves the package cut before the zero-row
accumulator source additions. The Fable review is older. Both remain evidence,
not approval of this current source cut. No status below treats them as such.

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
family. No v1.0 Pad-as-matrix-zero encoding is present. Transcript, state,
package-identity, and verifier-context binding use Poseidon2 only.

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
The phase compiler still must construct the complete low-norm plan before
these matrices can replace the current package relation metadata.

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
Lifecycle/Stage1/Formal.lean                 △ order, footprint, soundness
Layout/Stage1/AssemblerInputs.lean           ✓ compact cross-phase wiring
Layout/Stage1/AssemblerSoundness.lean        △ deterministic relation composition
Layout/Stage1/Preservation.lean              ○ final physical preservation
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

The final recursive relation has 6,377,555 structural rows and logical width
264,627,433. Its Φ81 carrier width, and therefore its exact joint domain, is
264,627,486. This is below `2^28 = 268,435,456` with 3,807,970 points of
headroom. The outer terminal metadata adds no row or column.

## Canonical package cut

| Final sealed package value | Exact value |
|---|---:|
| Physical rows | 29,225,729 |
| Logical relation rows | 6,377,555 |
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
structural = [12361274004522005101, 16152982874305751731,
              1678681406509731088, 132363367961763188]
package    = [13741043116079060421, 5674518528429785720,
              11166766960909880549, 9224649085053790129]
context    = [10416656309336468580, 11453309885101557621,
              6231848007073348033, 891737494100938774]
vk         = [17821258025285015024, 16394839360284581327,
              1628867512508252830, 9450997652215229796]
```

No external reviewer has approved a source fingerprint for this cut.

SHA-256 identifies bytes only:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Current prefix plan | 93,823,057 | `1e3b40436b9e3fe8eee846bcb6d5c0161717157bf6250e0a843253fc0f61116d` |
| Final sealed package | 121,734,708 | `8c18ad3ad107d872514c06ca335bea8bc53594cc8ac987a0047c38c15c37de84` |
| Pilot parity | 697,379 | `5c0925eb8864d81ab80235dad796457f91cb2063253688f57159ed5029dac251` |
| PiCCS parity | 1,642,578 | `c5640faaab5612a463f92b6e15fb171f553c051be6040cbbc1129944dc786a5d` |
| PiRLC parity | 1,136,741 | `13656f3d450dc377cd7b6768d7663356c5452a5e3bd2970187627450c1b08e94` |
| PiDEC parity | 1,653,850 | `fc7c325c2388e5e1e97d98f90c69f97831b59977939103437c6973a17b16fa8f` |
| PiRLC sampler parity | 9,202 | `e1ee42037d7750725c9442d7693b93eb60dd56c5507577370d4f06e65aad88a3` |
| Application and terminal-layout parity | 842,081 | `b3e636e926560fc0c6bc71852d6364ac301dfaffcafede92ed6a7a4ac2ba6b9b` |

## Exact conformance evidence

Lean evidence on this cut:

- `validate.sh static`: all boundary checks passed.
- full `NightstreamFPrime` root passed on the current cut in 9 seconds.
- `tests/AxiomsStage1Accumulator.lean`: all 18 new theorem roots passed in
  3 seconds with only `propext`, `Classical.choice`, and `Quot.sound`.
- `AccumulatorSemantics.phases_imply_holds` proves that the complete
  PiCCS → PiRLC → PiDEC verifier graph equals `Accumulator.Holds`.
- `AccumulatorPackage.circuitPackage_implies_accumulatorHolds` derives that
  result from the unchanged canonical package rows.
- `validate.sh axioms`: 3,342 jobs passed in 9 seconds. Audited theorems use
  only `propext`, `Classical.choice`, and `Quot.sound`; the production strong
  set and complete-fork extraction roots are included.
- the phase-local Stage 1 parent validates in 3 seconds after explicit
  Poseidon and PiCCS footprint metadata replaced the default executable list
  reductions; the rejected first check exceeded 86 seconds and 25.5 GB RSS;
- the focused application-and-terminal parity file passed in 3 seconds and
  emitted its schema-2 fixture in 33 seconds;
- forced compact package emission improved from a 7.16-second baseline median
  to 5.78 seconds with `LEAN_NUM_THREADS=10`, a 19.3% reduction;
- forced expanded emission improved from a 9.11-second baseline median to
  7.39 seconds, an 18.9% reduction. Every baseline and final output was
  byte-identical to the checked artifact. No reliable compile-time reduction
  was established.
- all five dependent phase parity fixtures and the independent sampler
  fixture were regenerated from the same locked Lean source.
- CI is configured to run `formal/nightstream-fprime/scripts/validate.sh all`.
  No remote CI run is claimed on this cut.

The speed path preserves the generic specifications. It uses proved direct
symbolic states, shared PiCCS operation packets, linear Horner construction,
streamed row classification, direct typed serialization, and cached immutable
punctuation buffers. The package formats remain the schema authority. The
emitter uses `LEAN_NUM_THREADS=10` and writes one canonical order.

Rust evidence on this cut:

- exact final-package matrix conformance: 1/1 on the final sealed identity;
- exact final matrix nonzeros `A/B/C`:
  `[93,701,820, 39,358,148, 28,868,018]`;
- independent assignment evaluation: all 29,225,729 physical rows and the
  padded zero domain passed;
- current-cut owner-family mutation counts are open because the prefix
  conformance support still pins the superseded identity and κ=18 column map;
- strict package loader: 14/14 on the current prefix artifact;
- compact-plan loader: 10/10 in 36.69 seconds;
- pilot parity and mutations: 3/3 in 20.72 seconds;
- complete PiCCS Lean / PaperExact / optimized parity: 4/4 in 3.76 seconds;
- complete indexed PiRLC parity and handoff: 3/3 in 2.34 seconds;
- complete PiDEC Lean / PaperExact / optimized parity: 3/3 in 33.10 seconds;
- PiRLC sampler parity and fail-closed decoding: 2/2;
- final nonzero assignment, independent evaluator, exact matrices,
  application mutations, and terminal-layout comparison: 1/1 in 47.71
  seconds;
- the final-identity NIFS engine cross-check passed its focused nonzero test;
- Poseidon2 Lean vectors: 2/2;
- the full `nightstream-fprime` aggregate remains red because
  `package_matrix_conformance` still pins the superseded prefix identity in
  `check_package_conformance/support.rs`. This is not a semantic failure, but
  it is an open gate.

The matrix comparator expands the Lean plan independently and compares every
final A/B/C row, column index, canonical coefficient, constant placement, and
public mapping. The assignment evaluator does not call the matrix builder,
row expander, witness generator, or package constraint evaluator.

The package consumer is a test bridge. It is not the final production
lifecycle, and its direct package proof is not evidence that Rust implements
the Lean relation. The semantic, matrix, assignment, and parity gates above
provide that evidence.

Superseded native v1.0 and radix-four test targets were removed from Cargo and
the tree. They are recoverable from Git. Their required v1.1 properties are
covered by the exact phase and package gates above. Current tests that still
serve a v1.1 contract were migrated to separate `Eval_K` / `Eval_A` and the
canonical nonempty running accumulator.

## Open authority and assembly edges

PiCCS must remain status open until this exact edge exists:

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
identity is pinned, and the complete prefix-preservation chain is proved.

The remaining owner-ordered work is:

1. retain the verifier-derived PiCCS round point in the phase contract, use it
   to finish the sole Stage 1 `FormalCircuit`, and prove its physical
   preservation;
2. complete Lean-Rust parity for the exact κ=22 Ajtai setup artifact;
3. refresh the remaining prefix conformance identity pin and rerun the full
   Rust aggregate;
4. make the validated sealed package the only reachable Rust production relation
   and remove the alternate radix relation;
5. rerun all exact-cut reviews and every PiCCS, PiRLC, PiDEC, application,
   terminal-layout, matrix,
   assignment, parity, and mutation gate on the final package identity;
6. after separate owner approval of a production backend, execute the final
   production `prove → verify` obligation.

No backend is authorized on this cut. Backend acceptance cannot replace any
semantic or conformance gate in this file.
