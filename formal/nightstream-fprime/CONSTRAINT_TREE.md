# Nightstream F′ Stage 1 constraint tree

This file is the concise audit map for the one production circuit. It defines
no relation. `✓` means present and validated; `○` means required and open.

The audit path is `paper formula → Lean predicate → leaf FormalCircuit →
phase assembler → Stage 1 assembler → proved layout → Lean package → Rust`.

PiCCS is formally **status open**. The current phase-local evidence can unlock
PiRLC under the accepted
[`piccs-phase-local-conformance-order`](../../decisions/piccs-phase-local-conformance-order.md)
decision. It does not close PiCCS. Every PiCCS gate must run again on the final
package identity after the package-to-`LogicalRelation`, application, key, and
recursive fixed-point edge exists.

## Exact SuperNeo v1_1 authority

```text
Spec/Folding/PiCCS/
├── Statement.lean       ✓ prior point and separate input Eval_K / Eval_A
├── Transcript.lean      ✓ verifier-owned Fiat–Shamir schedule
├── TranscriptReplay.lean ✓ complete statement-and-message replay reduction
├── EvalK.lean           ✓ Pad evaluation family
├── EvalA.lean           ✓ 14 CCS-matrix evaluation families
├── FinalIdentity.lean   ✓ v1_1 terminal joint identity
└── Accepted.lean        ✓ sole PiCCS acceptance predicate

Spec/ProductionRelation/SelectivePolynomial.lean
└── polynomial           ✓ 13 selective matrices + zero; 74 terms; degree 8
```

`Lifecycle/PaperAlgebra.lean` constructs Pad separately from the 14 CCS
matrices. No Pad-as-matrix-zero relation and no combined `Eval` value remain.
Equality gating gives the production PiCCS degree bound 9. Each of the 25
SumCheck messages therefore has 10 extension-field coefficients.

`Layout/Stage1/PiCCSSecurity.lean` identifies a canonical context, a complete
well-formed running statement, and an exact transcript replay unless a named
context-hash, state-hash, transcript-challenge, or transcript-state collision
occurs. This is a deterministic reduction. It does not assign a probability
to a failure event and does not replace the final Stage 1 security composition.

## Pilot conformance closure

The pilot is **Conformance-closed** on the current source cut. Its semantic
path is `Lifecycle.Pilot.phase_soundness` and
`Lifecycle.Pilot.builders_imply_hash_slots`; its emitted-row path reaches
`Export.Pilot.canonicalPackage_implies_recursive_hash_slots` and
`Export.Stage1.Package.circuitPackage_implies_pilotSpec`.

The pilot has exactly 12,574,138 rows, 12,659,088 source columns, 58 public
columns, and joint domain 12,659,088. `PilotProduction.jointDomain_matches`
connects that domain to the exact row/column maximum, and
`PilotProduction.jointDomain_le_twoPow25` proves the production bound.

`Export.Stage1.PilotParity` emits schema 1 with two distinct 42,475-word
preimages, both four-word digests, all 58 public values, and the relative
public segments `[4, 0, 54]` and `[5, 54, 4]`. The artifact SHA-256 is
`50e42cbb83b8f2b0b287e21f818f714b079d0f8c575fde7925ab44ef416d575e`.
The native-task emitter computes each digest once and emitted this artifact
in 10.25 seconds; the prior external-review run took 464 seconds.

Executed release evidence:

- `pilot_lean_nonzero_parity`: 3/3 passed in 18.58 seconds. Independent Rust
  Poseidon2 recomputation equaled the complete Lean result. Mutations covered
  both preimages, every serialization family, all 16 running sources, every
  source's 14 separate `Eval_A` families, all 58 public values, wrong lengths,
  trailing-zero extension, and noncanonical field words.
- `poseidon2_lean_vectors`: 2/2 passed. The Rust permutation and sponge equal
  the Lean primitive vectors.
- `package_matrix_conformance`: 1/1 passed in 38.09 seconds. It compared the
  final padded Rust `A/B/C` matrices with every canonical Lean row and used a
  separate evaluator for all 25,556,958 current-prefix rows and the padded
  zero domain. This includes every pilot row.

This status is not Production-closed. The relation-identity gate must rerun
these applicable checks on the final complete Stage 1 package identity before
Pilot, PiCCS, or Stage 1 production closure.

## PiCCS leaf circuits

Logical owners are under `Lifecycle/PiCCS/v1_1/`. Matching physical owners
are under `Layout/PiCCS/v1_1/Leaves/`.

| Leaf | Mathematical constraint | Rows | Stage 1 column delta |
|---|---|---:|---:|
| `StatementBinding.lean` | Bind state, fresh claim, and context | 160 | 0 |
| `StatementAbsorption.lean` | Absorb digest and fresh claim | 160,432 | 160,432 |
| `ChallengeDerivation.lean` | Derive 25 `α` coordinates and `γ` | 46,176 | 46,176 |
| `RoundTranscript.lean` | Absorb 25 degree-9 messages; derive `r′` | 133,200 | 133,200 |
| `InitialClaim.lean` | `T = T_K + γ^864 T_A` | 116,631 | 116,631 |
| `SumcheckChain.lean` | `pᵢ(0)+pᵢ(1)=cᵢ`; `cᵢ₊₁=pᵢ(rᵢ)` | 378,610 | 378,560 |
| `EvalKTerminal.lean` | Pad-family terminal evaluation `E_K` | 8,458 | 8,458 |
| `EvalATerminal.lean` | 14-matrix terminal evaluations `E_A` | 109,546 | 109,546 |
| `CcsTerminal.lean` | Evaluate the 74-term selective CCS polynomial | 20,794 | 20,794 |
| `NormTerminal.lean` | Low-norm residual `N` | 752 | 752 |
| `FinalIdentity.lean` | Final v1_1 identity and terminal equality | 130,419 | 130,417 |
| `OutputBinding.lean` | Absorb 17 separate `Eval_K` / `Eval_A` outputs | 4,076,512 | 4,076,512 |

Reusable operations remain under `Gadgets/Poseidon2/`, `Gadgets/SumCheck/`,
`Gadgets/Polynomial/`, and `Gadgets/Multilinear/`. Parents use child
contracts; they do not unfold child operations.

## PiCCS phase assembly

```text
Lifecycle/PiCCS/v1_1/
├── [twelve leaves]      ✓ local soundness, completeness, and footprint
├── Formal.lean          ✓ sole phase assembler; theorem `soundness`
└── Completeness.lean    ✓ theorem `completeness`
Layout/PiCCS/v1_1/
├── Lowering.lean        ✓ one logical-to-physical lowering
├── Composition.lean     ✓ exact child order and footprint sum
├── Ownership.lean       ✓ row and column ownership
└── Preservation.lean    ✓ `physical_implies_phaseHolds`, `physical_complete`
```

Exact PiCCS production totals are 5,181,690 physical rows, 685,348 lowering
fresh columns, and a 5,181,478 physical-column delta from the parent offset.
`Composition.jointDomain_le_twoPow25` proves the zero-based phase domain
bound. `PilotPiCCS.cumulativeFootprints_eq` transports every leaf delta to
the Stage 1 prefix and proves every cumulative row, column, and joint-domain
endpoint.

## Current Stage 1 package prefix through PiRLC

```text
Layout/Stage1/
├── PiCCSInputs.lean       ✓ four public context words + 29,012 proof words
├── PiCCSProofInputs.lean  ✓ typed 25×10 proof-message decoder
├── PiCCSStarts.lean       ✓ one set of logical and physical starts
├── PiCCSSecurity.lean     ✓ deterministic committed-statement reduction
└── PilotPiCCS.lean        ✓ pilot → PiCCS composition and full ledger
Export/Stage1/
├── PiCCSInvocations.lean  ✓ PiCCS Poseidon2 invocations
├── PiCCSArithmetic.lean   ✓ 765,370 arithmetic rows
├── PiCCSCompleteness.lean ✓ semantic witness → emitted PiCCS rows
├── PiRLC*Invocations.lean ✓ sampler, First54, and combination rows
├── PiRLC*Completeness.lean ✓ every PiRLC child reaches package rows
├── PackageCompleteness.lean ✓ cumulative `complete_piRlcPackageRows`
├── WitnessProgram.lean    ✓ Rust-interpreted expression IR
├── VerifierContextCandidate.lean ✓ canonical prefix context recipe
├── PiCCSNonzero.lean      ✓ complete deterministic nonzero fixture
├── PiCCSParity.lean       ✓ complete Lean PiCCS result vector
├── PiRLCParity.lean       ✓ all 17 indexed partial and final results
├── PackagePlan.lean       ✓ proved schema-8 generative plan expansion
├── Data.lean              ✓ sole package data assembler
└── Package.lean           ✓ rows → `PhaseHolds`; exact row coverage
```

The 29,012 private proof-input words are 972 commitment words, 500 words for
25×10 extension coefficients, and 27,540 output words. Each of the 17 outputs
has 108 `Eval_K` / Pad words and 1,512 `Eval_A` / matrix words. Four additional
public words hold the verifier-owned context digest.

| Emitted package value | Exact value |
|---|---:|
| Rows | 25,556,958 |
| Source columns / joint domain | 25,669,063 |
| Private columns / constant column | 25,669,001 |
| Public columns | 62 |
| Total unpadded columns | 25,669,064 |
| Witness instructions | 850,180 |
| Assertion rows | 136,129 |
| Poseidon2 permutation invocations | 7,613 |
| Compact templates / invocations | 326 / 163,574 |

The joint domain is `25,669,063 ≤ 2^25`. The final backend-neutral R1CS
layout pads the row and private-column domains to `2^25`: 33,554,432 rows,
constant column 33,554,432, 62 public columns, and 33,554,495 total columns.
Every padded row is empty and every padded private assignment value is zero.

This package uses the accepted digest-only schedule. Its 160,432-row statement
absorption starts a fresh domain-separated transcript, absorbs the constrained
prior-state digest, and then absorbs the fresh commitment and public input.
The values below identify the current phase-local candidate. They do not give
PiCCS Conformance-closed status.

The canonical artifact is a schema-8 generative plan whose proved expansion
is the schema-7 semantic package. It carries source tags
`[Bit, GeneralSelector, A, B, C, SboxInput, CenteredUnit, EvalSelector,
Class0, Class1, Class2, Class3, Class4, Zero]`, degree bound 9, and all 74
polynomial terms. Its verifier-owned Poseidon2 relation identifier is:

```text
[2880828118570533443, 12363340834605518522,
 17891354081046714225, 8467327743520570474]
```

The plan SHA-256 is
`d5b43c7e2d1586475dffdf0c70a23688521b7811ee91b9cb095f9a2b56322f16`.
The exact expanded reference SHA-256 is
`39036d30ff3ee064a16f152a6749a89c85167fe3ee34b248110103f4f6bdabed`.
The complete nonzero PiCCS parity artifact uses schema 7 and has SHA-256
`8f46ed7afbe0fe0c030ee8af03541851a087e49331bf1c4edae60c8bc9608899`.
The indexed PiRLC schema-2 artifact has SHA-256
`a7ea997cc19e79c08328f5aa610536db75d2be35e6ddb1247c529e4bbe65c29e`.

The phase-local verifier context contains canonical relation, application,
NIFS-key, and commitment-key word lists of lengths `[4, 4, 68, 36]`. Rust
derives a typed context from the loaded package identity and commitment-key
words. It rejects raw or noncanonical authority words. The current relation
and application lists use reserved package-identity words. This is not the
final theorem that the package constructs and selects the exact 14-matrix
`LogicalRelation` and application used by `ProductionKey.key` and
`StepHolds`.

The transported cumulative ledger is:

| Endpoint | Rows | Source columns / joint domain |
|---|---:|---:|
| Pilot plus proof and context inputs | 12,574,138 | 12,688,104 |
| Statement binding | 12,574,298 | 12,688,104 |
| Statement absorption | 12,734,730 | 12,848,536 |
| Challenge derivation | 12,780,906 | 12,894,712 |
| Round transcript | 12,914,106 | 13,027,912 |
| Initial claim | 13,030,737 | 13,144,543 |
| SumCheck chain | 13,409,347 | 13,523,103 |
| `Eval_K` terminal | 13,417,805 | 13,531,561 |
| `Eval_A` terminal | 13,527,351 | 13,641,107 |
| CCS terminal | 13,548,145 | 13,661,901 |
| Norm terminal | 13,548,897 | 13,662,653 |
| Final identity | 13,679,316 | 13,793,070 |
| PiCCS output binding | 17,755,828 | 17,869,582 |
| PiRLC input binding | 17,755,828 | 17,869,582 |
| PiRLC sampler chain | 18,764,676 | 18,876,781 |
| PiRLC commitment combination | 21,259,800 | 21,371,905 |
| PiRLC public-input combination | 21,398,418 | 21,510,523 |
| PiRLC `Eval_K` combination | 21,675,654 | 21,787,759 |
| PiRLC `Eval_A` / current prefix | 25,556,958 | 25,669,063 |

`PilotPiCCSPiRLC.cumulativeFootprints_eq` states these endpoints in Lean, and
`PilotPiCCSPiRLC.jointDomain_le_twoPow25` proves the prefix bound.

## Rust v1_1 path and evidence

```text
crates/
├── nightstream-fprime/src/package/{r1cs,relation,v1_1,pi_ccs_v1_1_transcript}.rs
│   └── exact final matrices, tags, typed inputs, and transcript replay ✓
├── neo-reductions/src/engines/{paper_exact_engine,optimized_engine}/
│   └── literal v1_1 formulas / optimized byte-equivalent formulas      ✓
├── neo-fold-clean/tests/nifs/pi_ccs_lean_nonzero_parity.rs
│   └── complete Lean / paper_exact / optimized nonzero parity          ✓
└── neo-fold-clean production lifecycle uses only package rows              ○
```

Executed release evidence:

- `poseidon2_lean_vectors`: 2/2 passed. Rust permutation and sponge hashing
  match the Lean reference used by both nonzero pilot hash chains.
- `package_loader`: 14/14 passed in 71.12 seconds. It checked the schema-8
  plan, schema-7 package identity, verifier-context authority, complete nonzero
  transcript replay, separate `Eval_K` / `Eval_A`, canonical coefficients,
  raw commitment-key mutation, noncanonical authority rejection, and package
  mutation rejection.
- `package_schema7_compact`: 5/5 passed in 26.94 seconds. It checked plan
  positions, exact generated coverage, malformed optional outputs,
  output-recipe self-reference, and incomplete input coverage.
- `package_matrix_conformance`: 1/1 passed in 38.09 seconds. An independent
  expander compared every final A/B/C row and term with the Lean-lowered rows.
  An independent evaluator checked all 25,556,958 unpadded rows, all padded
  empty rows, the relocated constant, all public values, and zero private pad.
  Fifteen row, column, coefficient, input, transcript, output, and context
  mutations were rejected.
- `nifs_pi_ccs_lean_nonzero_parity`: 4/4 passed in 3.24 seconds. Lean,
  `paper_exact`, and `optimized` matched byte-for-byte for acceptance,
  challenges, every intermediate state and claim, all six terminal components,
  all 17 output claims and evaluation families, and the outgoing state. The
  test also rejected mutations across every input, proof, output, and result
  family.
- `pi_ccs_v1_1_engine_parity`: 15/15 passed in 42.01 seconds.
- `nifs_pi_rlc_lean_nonzero_parity`: 3/3 passed in 1.91 seconds. It compared
  all 17 challenges and membership bits, every indexed partial commitment,
  public input, `Eval_K`, and `Eval_A`, the final claim, handoff, outgoing
  state, both engines, and every indexed mutation family.
- `nifs_engine_crosscheck`: 10/10 passed in 88.68 seconds, including the
  nonzero 25×10 proof bridge and package offsets.
- `validate.sh axioms`: passed 3,270 jobs in 10.49 seconds. The audited
  theorems use only `propext`, `Classical.choice`, and `Quot.sound`.

These gates prove the matrix, assignment, value, transcript, context, and
mutation properties of the current digest-only phase-local candidate. PiCCS
status remains open until the final authority edge below exists and the same
gate set passes again on that final package identity. Production package-only
`prove → verify` also remains open and is not inferred from backend
acceptance.

## Development feedback and multicore execution

The sampler completeness proof uses opaque child segments and a proved
equivalence to the unchanged canonical monolithic lowering. The exact theorem
`Sampler.physical_complete` still concludes agreement outside the same 59,247
columns, while `Sampler.physicalHolds_iff_rowsHold` proves that its segmented
predicate is exactly `RowsHold` for the package plan. The focused completeness
build is 1.0–1.7 seconds; the prior monolithic proof check was 472 seconds.
The exact `First54` selector uses the same boundary for its 64-round prefix
and final fail-closed assertion. Its focused build is 9.2 seconds, down from
63–95 seconds. An affected PiRLC export and axiom rebuild is 85 seconds, and a
current full axiom gate is 13 seconds. Every new bridge passes the
allowed-axiom audit.

The owner approved per-process multicore use for development. `validate.sh`
selects all 10 available workers. The Lean emitter prepares independent
witness groups, ordinary-row blocks, and permutation blocks as tasks, then
writes their results in the one canonical order. Rust expands witness plans
with Rayon, builds final A/B/C matrices with `rayon::join`, and evaluates the
independent conformance schedule in parallel. Assignment execution and the
Poseidon2 identity sponge remain ordered because later values depend on the
earlier prefix.

The current canonical schema-8 emitter completed in 123 seconds. Its output
was byte-equal to the stored 68,815,188-byte artifact and kept SHA-256
`d5b43c7e2d1586475dffdf0c70a23688521b7811ee91b9cb095f9a2b56322f16`.

## PiRLC sampler migration

The pre-phase Rust sampler migration is complete. Lean remains the semantic
authority in `Lifecycle/Transcript.lean` and the production sampler modules
under `Spec/Folding/Nifs/NonInteractive/PiRlcSampler/`.

The exact schedule is: absorb `[4, coordinate]`; consume eight complete
four-lane Poseidon2 digest windows; decode low then high 16-bit candidates
from every lane; reject only `65535`; map accepted candidates modulo five to
`[-2, -1, 0, 1, 2]`; keep the first 54; and reject the batch on shortfall.
Every scalar consumes all eight windows before the next scalar begins.

`Export/Stage1/PiRlcSamplerParity.lean` emits schema 1 with decoder boundary
cases, injected exact-bound success, injected shortfall, and two complete
transcript-chained samples. The artifact SHA-256 is
`e1ee42037d7750725c9442d7693b93eb60dd56c5507577370d4f06e65aad88a3`.

Rust uses `Poseidon2Transcript::squeeze_digest_v1_1` and
`decode_pi_rlc_v1_1_coefficients`. Both optimized and PaperExact paths consume
this decoder. They reject a nonzero absorb cursor instead of panicking.

Executed release evidence:

- `nifs_pi_rlc_sampler_lean_parity`: 2/2 passed. It compared all constants,
  decoder cases, exact 54-of-64 success, injected shortfall, all digest lanes,
  every candidate, both complete transcript state transitions, all 54 ring
  coefficients, and the complete PaperExact and optimized rotation matrices.
- `rot_rho_tests`: 9/9 passed, including fail-closed cursor rejection.
- `rlc_dec_k_gt1`: 11/11 passed in 1.18 seconds, including sampled commitment,
  public-input, evaluation, and witness combinations.
- `validate.sh static`: passed after the new export and emitter mode.

This closes only the required sampler migration. The PiRLC logical assembler,
physical layout, and phase-local package compiler now exist. The legacy native
R1CS sampler has a different transcript schedule and is not evidence for these
claims; the final package-only production migration must make that alternate
relation unreachable.

## PiRLC phase-local compiler

`Lifecycle/PiRLC/v1_1/Formal.lean` is the sole seven-child logical assembler.
Its `circuit` is sound and constructively complete for
`Semantics.PhaseHolds`. `Layout/PiRLC/v1_1/Preservation.lean` proves both
directions through the physical rows. The cumulative Stage 1 layout through
PiRLC has 25,556,958 rows, 25,669,063 source columns, and joint domain
25,669,063, which is below `2^25`.

The canonical package representation partitions the 7,801,130 PiRLC rows as
follows:

- 220,881 ordinary digest-lane and fail-closed selector rows;
- 153 canonical Poseidon2 invocations, with 592 rows each;
- 697,391 exact compact First54 rows;
- 6,792,282 compact combination rows.

`PiRLCFirst54Conformance.packageInvocations_imply_spec` proves the full
First54 package semantics. `Package.circuitPackage_implies_piRlcSamplerChain`
proves the complete 17-source sampler chain. The generic combination schedule,
input mapping, row semantics, and package membership are proved.
`Package.circuitPackage_implies_piRlcPhaseHolds` assembles the exact
seven-child phase unconditionally. `Data.circuitPackage_compactRowInvocations`
and the combination-count theorems prove the exact selection and totals.
`PackageCompleteness.complete_piRlcPackageRows` constructs one completed
environment from semantic PiRLC, preserves Pilot and PiCCS rows, and proves
the complete canonical package `RowsHold` predicate.

The PiRLC compiler is closed on this prefix. Its phase-local conformance gates
are green: exact schema-8 expansion, entry-for-entry final `A/B/C`, an
independent evaluation of all 25,556,958 rows, schema-2 three-way nonzero
parity with every indexed partial, and the required mutations. The current
external reviews map all PiRLC leaves but predate these final artifact runs;
PiRLC is therefore a validated conformance candidate until an external review
records this exact source-and-artifact cut. This does not change PiCCS status
open or authorize PiDEC, Production-closed, or Stage 1 closure.

## Open assembly levels

```text
Lifecycle/PiRLC/v1_1/InputBinding.lean ✓ 17 inputs; 0 rows / 0 columns
Lifecycle/PiRLC/v1_1/Formal.lean       ✓ seven-child phase assembler
Lifecycle/PiDEC/v1_1/Formal.lean       ○ 16-child phase assembler
Lifecycle/Stage1/Formal.lean           ○ sole full Stage 1 assembler
Layout/PiRLC/v1_1/Leaves/InputBinding.lean ✓ proved 0-row footprint
Layout/PiRLC/v1_1/{Lowering,Preservation}.lean ✓ phase layout and both directions
Layout/PiDEC/v1_1/                     ○ lowering and preservation
Layout/Stage1/                         ○ full ownership and preservation
Export/Stage1/                         ✓ PiRLC package soundness and completeness;
                                         PiDEC and later phases open
```

The final path is `pilot → PiCCS → PiRLC → PiDEC → application → output hash → terminal`; it lowers into one package, with no second production relation.

The exact PiCCS closure edge that remains open is:

```text
final canonical package
  -> canonical 14-matrix LogicalRelation
  -> ProductionKey.key
  -> exact application F and StepHolds
  -> recursive fixed point
  -> rerun every PiCCS gate on this final identity
```

The owner-order exception permits PiRLC work before this edge closes. It does
not permit a PiCCS or Stage 1 Conformance-closed claim.
