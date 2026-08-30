# Nightstream F′ Stage 1 constraint tree

This file is the concise audit map for the current Lean-authored package
prefix through the running-instance branch. It defines no relation and gives
no digest authority. The audit path is:

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
  row family, and proves package rows imply the selected `F`.
  Status open because no concrete production application exists.
- Stage 1: open. The seven-child opaque assembly order, derived offsets,
  aggregate footprints, arbitrary-witness soundness, and the generic
  package-row-to-`StepHoldsFor` theorem are kernel-checked. That package theorem
  still takes the typed external-ABI representation and the existing PiRLC and
  PiDEC scope certificates as premises. Complete parent-circuit physical
  preservation, a concrete application, fixed point, final domain proof,
  security composition, and package-only production path do not exist yet.

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
preimage is 45,937 Goldilocks words. Each running group contains 972
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
Export/Stage1/PerApplicationPackage.lean     △ generic final package and F edge
Export/Stage1/PerApplicationSoundness.lean   △ package rows imply StepHoldsFor
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
| Digest-only statement absorption | 192,400 | 192,400 |
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
| Commitment combination | 2,495,124 | 2,495,124 |
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

## Proved cumulative layout through the running-instance branch

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

| Endpoint | Rows | Source columns / joint domain |
|---|---:|---:|
| Pilot | 13,600,754 | 13,692,624 |
| PiCCS input ABI | 13,600,754 | 13,721,700 |
| PiCCS statement binding | 13,600,914 | 13,721,700 |
| PiCCS statement absorption | 13,793,314 | 13,914,100 |
| PiCCS challenge derivation | 13,844,818 | 13,965,604 |
| PiCCS round transcript | 13,994,002 | 14,114,788 |
| PiCCS initial claim | 14,110,633 | 14,231,419 |
| PiCCS SumCheck chain | 14,535,290 | 14,656,020 |
| PiCCS `Eval_K` terminal | 14,543,832 | 14,664,562 |
| PiCCS `Eval_A` terminal | 14,653,462 | 14,774,192 |
| PiCCS CCS terminal | 14,674,256 | 14,794,986 |
| PiCCS norm terminal | 14,675,008 | 14,795,738 |
| PiCCS final identity | 14,805,511 | 14,926,239 |
| PiCCS output binding | 18,882,023 | 19,002,751 |
| PiRLC sampler chain | 19,890,871 | 20,009,950 |
| PiRLC commitment combination | 22,385,995 | 22,505,074 |
| PiRLC public-input combination | 23,079,085 | 23,198,164 |
| PiRLC `Eval_K` combination | 23,356,321 | 23,475,400 |
| PiRLC `Eval_A` combination | 27,237,625 | 27,356,704 |
| PiDEC input ABI | 27,237,625 | 27,402,496 |
| PiDEC public split | 27,260,305 | 27,420,586 |
| PiDEC commitment | 27,261,277 | 27,420,586 |
| PiDEC `Eval_K` | 27,261,385 | 27,420,586 |
| PiDEC `Eval_A` | 27,262,897 | 27,420,586 |
| Running-instance branch / current endpoint | 27,584,200 | 27,695,988 |
| Accumulator semantic/package edge | 27,584,200 | 27,695,988 |

The current joint domain is 27,695,988. It is below
`2^28 = 268,435,456` with exact headroom 240,739,468. The accumulator edge
adds zero rows and columns. The per-application framework is not embedded in
this candidate and therefore changes no current count. The outer terminal
metadata also adds no row or column. This headroom still must contain the
selected application, its witness inputs, and its lowering. Compact
serialization does not reduce backend rows.

Separately, Lean proves that the current direct low-norm Poseidon2 plan needs
108,160,050 retained S-box coordinates, leaving 160,275,406 coordinates below
`2^28`. That count excludes final outputs and non-Poseidon source values. It
is not the complete final-fit theorem.

## Canonical package cut

| Package value | Exact value |
|---|---:|
| Unpadded rows | 27,584,200 |
| Private columns / constant source index | 27,695,710 |
| Public columns | 278 |
| Total unpadded columns | 27,695,989 |
| Caller-owned private inputs | 166,738 |
| Witness instructions | 1,190,446 |
| Assertion rows | 190,009 |
| Ordinary compiled rows | 1,380,455 |
| Poseidon2 invocations | 7,703 |
| Compact templates | 326 |
| Compact invocations | 167,246 |
| Padded matrix rows | 268,435,456 |

The verifier-owned Poseidon2 relation identifier is:

```text
[5326948389888638380, 15945253772729055182,
 12038831075978321435, 4066786242110063495]
```

The last externally reviewed source-cut fingerprint is:

```text
32291f7c9edbe968a171421da665b9b8de7fe0050044a684f58c4a25c3d9b13d
```

It identifies the superseded 27,584,180-row cut. It is the SHA-256 of the
sorted `shasum -a 256` records for all files under
`formal/nightstream-fprime`, `crates/nightstream-fprime`,
`crates/neo-fold-clean/{src,tests}`, and `crates/neo-reductions/src`, plus the
three Rust `Cargo.toml` files. It excludes `.lake`, generated artifacts, and
this audit file. Paths are part of each inner record.

SHA-256 identifies bytes only:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Compact schema-8 plan | 92,147,859 | `a293eb5e6af2d956c92490a3038285f8bf460fcb06fdde0e5663882b13c82307` |
| Expanded schema-7 package | 113,511,309 | `d3c7be44e8eb74e87a4d13286fd293491f1c53e749401e729497028b56d06f45` |
| Pilot parity | 657,652 | `47b151c8d04dae1ce2898a749396fad27c54b991ca00034a7fde951055e3554a` |
| PiCCS parity | 1,577,847 | `08e356f8b2ef9e8a2a0399be6d5b6404ad3b19a2ea215e12b81c7869c51092f4` |
| PiRLC parity | 1,195,249 | `7a5c877dec8171d7dde79cecf6525da8d99128ea98a1c4ffaa72025d3cf152ce` |
| PiDEC parity | 1,617,291 | `025d3bda2ed276f4e04614062e051d251e058bf4d0a46244f884a7aec9406592` |
| PiRLC sampler parity | 9,202 | `e1ee42037d7750725c9442d7693b93eb60dd56c5507577370d4f06e65aad88a3` |

## Exact conformance evidence

Lean evidence on this cut:

- `validate.sh static`: all boundary checks passed.
- full `NightstreamFPrime` root: 3,338 jobs passed in 6 seconds on the
  accumulator slice.
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
- full `NightstreamFPrime`: 3,350 jobs passed in 29 seconds after the
  per-application package additions; `NightstreamFPrimeTests`: 3,362 jobs
  passed in 10 seconds, including all focused Stage 1 axiom roots;
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

- exact package matrix conformance: 1/1 in 49.60 seconds;
- exact final matrix nonzeros `A/B/C`:
  `[88,443,680, 37,139,823, 27,233,617]`;
- independent assignment evaluation: all 27,584,200 unpadded rows and the
  padded zero domain passed;
- row-owner mutations rejected by the exact row comparator: 144;
- column/public-owner mutations rejected by the exact row comparator: 77;
- semantic input mutations: 16;
- total exact-package mutations: 237;
- strict package loader: 14/14 in 91.51 seconds;
- compact-plan loader: 10/10 in 36.69 seconds;
- pilot parity and mutations: 3/3 in 20.72 seconds;
- complete PiCCS Lean / PaperExact / optimized parity: 4/4 in 3.76 seconds;
- complete indexed PiRLC parity and handoff: 3/3 in 2.34 seconds;
- complete PiDEC Lean / PaperExact / optimized parity: 3/3 in 33.10 seconds;
- PiRLC sampler parity and fail-closed decoding: 2/2;
- identity-bound complete typed package consumer: 1/1 in 194.22 seconds;
  it consumed PiCCS, PiDEC, and the Lean-authored running-transition output,
  and rejected a changed public input.
- `nifs_engine_crosscheck`: 10/10 in 179.49 seconds, including the 270-word
  state-preimage bridge, PaperExact/optimized equality, carried accumulator,
  and Nebula auxiliary commitments;
- Poseidon2 Lean vectors: 2/2;
- all `nightstream-fprime` test targets compiled in 5.55 seconds;
- all `neo-fold-clean` test targets compiled in 57.18 seconds on the incremental
  retry. The first cold aggregate compile reached the five-minute cap without
  a compiler diagnostic, so it is not a passing cold-build result.

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
allowlist one final identity. `PerApplicationPackage.packageRows_imply_applicationHolds`
proves that the generic final package enforces the selected `F`, and
`packageRows_imply_baseOrdinaryRows` proves the ordinary-row part of prefix
column-shift preservation. No concrete application has been selected, and
template/hash/compact preservation is still open.

The remaining owner-ordered work is:

1. finish hash, permutation, and compact-row preservation for the generic
   per-application package;
2. select one concrete Lean application, instantiate its package, and prove
   the exact final `2^28` fit;
3. prove exact cross-phase wiring, deterministic soundness, the recursive fixed
   point, the complete `2^28` bound, and the separate security composition;
4. make the validated package the only reachable Rust production relation
   and remove the alternate radix relation;
5. rerun all exact-cut reviews and every PiCCS, PiRLC, PiDEC, matrix,
   assignment, parity, and mutation gate on the final package identity;
6. after separate owner approval of a production backend, execute the final
   production `prove → verify` obligation.

No backend is authorized on this cut. Backend acceptance cannot replace any
semantic or conformance gate in this file.
