# Nightstream F′ Stage 1 constraint tree

This file is the concise audit map for the one production circuit. It defines
no relation. `✓` means present and validated; `○` means required and open.

The audit path is `paper formula → Lean predicate → leaf FormalCircuit →
phase assembler → Stage 1 assembler → proved layout → Lean package → Rust`.

## Exact SuperNeo v1_1 authority

```text
Spec/Folding/PiCCS/
├── Statement.lean       ✓ prior point and separate input Eval_K / Eval_A
├── Transcript.lean      ✓ verifier-owned Fiat–Shamir schedule
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
`ba1420c80bacb4ea2e744ebae42fa4aef5ab2effa97e59a01e4b60fac818e905`.

Executed release evidence:

- `pilot_lean_nonzero_parity`: 3/3 passed in 18.59 seconds. Independent Rust
  Poseidon2 recomputation equaled the complete Lean result. Mutations covered
  both preimages, every serialization family, all 16 running sources, every
  source's 14 separate `Eval_A` families, all 58 public values, wrong lengths,
  trailing-zero extension, and noncanonical field words.
- `poseidon2_lean_vectors`: 2/2 passed. The Rust permutation and sponge equal
  the Lean primitive vectors.
- `package_matrix_conformance`: 1/1 passed in 48.52 seconds. It compared the
  final padded Rust `A/B/C` matrices with every canonical Lean row and used a
  separate evaluator for all 27,893,668 current-prefix rows and the padded
  zero domain. This includes every pilot row.

This status is not Production-closed. The relation-identity gate must rerun
these applicable checks after the selected digest-only PiCCS rows replace the
superseded candidate and before the new identity is pinned.

## PiCCS leaf circuits

Logical owners are under `Lifecycle/PiCCS/v1_1/`. Matching physical owners
are under `Layout/PiCCS/v1_1/Leaves/`.

| Leaf | Mathematical constraint | Rows | R1CS-fresh columns |
|---|---|---:|---:|
| `StatementBinding.lean` | Bind the prior point and claims | 0 | 0 |
| `StatementAbsorption.lean` | Absorb running and fresh statements | 10,298,432 | 0 |
| `ChallengeDerivation.lean` | Derive 25 `α` coordinates and `γ` | 46,176 | 0 |
| `RoundTranscript.lean` | Absorb 25 degree-9 messages; derive `r′` | 133,200 | 0 |
| `InitialClaim.lean` | `T = T_K + γ^864 T_A` | 116,631 | 90,713 |
| `SumcheckChain.lean` | `pᵢ(0)+pᵢ(1)=cᵢ`; `cᵢ₊₁=pᵢ(rᵢ)` | 378,610 | 378,560 |
| `EvalKTerminal.lean` | Pad-family terminal evaluation `E_K` | 8,458 | 6,634 |
| `EvalATerminal.lean` | 14-matrix terminal evaluations `E_A` | 109,546 | 85,258 |
| `CcsTerminal.lean` | Evaluate the 74-term selective CCS polynomial | 20,794 | 20,792 |
| `NormTerminal.lean` | Low-norm residual `N` | 752 | 720 |
| `FinalIdentity.lean` | Final v1_1 identity and terminal equality | 130,419 | 102,671 |
| `OutputBinding.lean` | Absorb 17 separate `Eval_K` / `Eval_A` outputs | 4,076,512 | 0 |

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

Exact PiCCS totals are 14,634,130 logical private variables, 14,634,182
logical rows, 685,348 lowering columns, 15,319,478 source columns, and
15,319,530 physical rows. `Composition.jointDomain_le_twoPow25` proves the
phase domain bound.

## Current Stage 1 package prefix

```text
Layout/Stage1/
├── PiCCSInputs.lean       ✓ 972 commitment + 500 rounds + 27,540 outputs
├── PiCCSProofInputs.lean  ✓ typed 25×10 proof-message decoder
├── PiCCSStarts.lean       ✓ one set of logical and physical starts
└── PilotPiCCS.lean        ✓ pilot → PiCCS composition
Export/Stage1/
├── PiCCSInvocations.lean  ✓ 24,585 Poseidon2 invocations
├── PiCCSArithmetic.lean   ✓ 765,210 non-transcript rows
├── PiCCSCompleteness.lean ✓ semantic witness → emitted PiCCS rows
├── PackageCompleteness.lean ✓ `complete_piCcsRows`
├── WitnessProgram.lean    ✓ Rust-interpreted expression IR
├── PiCCSNonzero.lean      ✓ complete deterministic nonzero fixture
├── PiCCSParity.lean       ✓ complete Lean PiCCS result vector
├── Data.lean              ✓ sole package data assembler
└── Package.lean           ✓ rows → `PhaseHolds`; exact row coverage
```

The 29,012 proof-input words are 972 commitment words, 500 words for 25×10
extension coefficients, and 27,540 output words. Each of the 17 outputs has
108 `Eval_K` / Pad words and 1,512 `Eval_A` / matrix words.

| Emitted package value | Exact value |
|---|---:|
| Rows | 27,893,668 |
| Source columns / joint domain | 28,007,578 |
| Private columns / constant column | 28,007,520 |
| Public columns | 58 |
| Total unpadded columns | 28,007,579 |
| Witness instructions | 685,348 |
| Assertion rows | 79,920 |

The joint domain is `28,007,578 ≤ 2^25`. The final backend-neutral R1CS
layout pads the row and private-column domains to `2^25`: 33,554,432 rows,
constant column 33,554,432, 58 public columns, and 33,554,491 total columns.
Every padded row is empty and every padded private assignment value is zero.

This candidate still uses the full 10,298,432-row statement absorption. The
owner selected the digest-only replacement in
`decisions/piccs-prior-state-digest.md`. The values below identify and validate
the now-superseded full-absorption candidate; they do not give it
Conformance-closed status.

Schema 6 carries source tags
`[Bit, GeneralSelector, A, B, C, SboxInput, CenteredUnit, EvalSelector,
Class0, Class1, Class2, Class3, Class4, Zero]`, degree bound 9, and all 74
polynomial terms. Its verifier-owned Poseidon2 relation identifier is:

```text
[2056683603671309374, 6478784752624371706,
 16274825114146670905, 1848990277754397221]
```

The exact package artifact SHA-256 is
`f2d49ddfb2c1aa8d673284b7b6df57e0e08c09218aa7ce01c952b997982b5f5e`.
The complete nonzero PiCCS parity artifact uses schema 4 and has SHA-256
`51cbbc20bb3b2db58e3523245676da9310879d246a5bcdb020d2fd7b1c9e1ab0`.

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
- `package_loader`: 13/13 passed in 20.92 seconds. It checked the schema-6
  identity, complete nonzero transcript replay, separate `Eval_K` / `Eval_A`,
  canonical coefficients, and package mutation rejection.
- `package_matrix_conformance`: 1/1 passed in 49.05 seconds. An independent
  expander compared every final A/B/C row and term with the Lean-lowered rows.
  An independent evaluator checked all 27,893,668 unpadded rows, all padded
  empty rows, the relocated constant, all public values, and zero private pad.
  Row, column, and coefficient mutations were rejected.
- `nifs_pi_ccs_lean_nonzero_parity`: 4/4 passed in 7.93 seconds. Lean,
  `paper_exact`, and `optimized` matched byte-for-byte for acceptance,
  challenges, every intermediate state and claim, all six terminal components,
  all 17 output claims and evaluation families, and the outgoing state. The
  test also rejected mutations across every input, proof, output, and result
  family.
- `pi_ccs_v1_1_engine_parity`: 15/15 passed in 40.81 seconds.
- `nifs_engine_crosscheck`: 10/10 passed in 89.56 seconds, including the
  nonzero 25×10 proof bridge and package offsets.

These tests prove the matrix, assignment, value, and mutation properties of
the superseded full-absorption PiCCS candidate. PiCCS Conformance-closed status
remains open until the selected digest-only relation has the same complete
evidence. Production package-only `prove → verify` also remains open and is
not inferred from backend acceptance.

## Open assembly levels

```text
Lifecycle/PiRLC/v1_1/InputBinding.lean ✓ 17 inputs; 0 rows / 0 columns
Lifecycle/PiRLC/v1_1/Formal.lean       ○ remaining 17-input phase assembler
Lifecycle/PiDEC/v1_1/Formal.lean       ○ 16-child phase assembler
Lifecycle/Stage1/Formal.lean           ○ sole full Stage 1 assembler
Layout/PiRLC/v1_1/Leaves/InputBinding.lean ✓ proved 0-row footprint
Layout/PiRLC/v1_1/{Lowering,Preservation}.lean ○ remaining phase layout
Layout/PiDEC/v1_1/                     ○ lowering and preservation
Layout/Stage1/                         ○ full ownership and preservation
Export/Stage1/                         ○ PiRLC, PiDEC, application, terminal
```

The final path is `pilot → PiCCS → PiRLC → PiDEC → application → output hash → terminal`; it lowers into one package, with no second production relation.
