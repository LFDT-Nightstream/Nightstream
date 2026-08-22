# Nightstream F′ Stage 1 constraint tree
This file is the audit index for the one production circuit. It maps paper
formulas to Lean owners and assembly levels. It does not define a relation.

Status: `✓` present and gated; `◐` present but exact phase completion open;
`○` required and absent; `△` closed pilot code that Stage 1 must subsume.
Required names change only when the exact v1.1 formula gives a better boundary.

```text
leaf FormalCircuits
  → phase Formal.lean
    → Lifecycle/Stage1/Formal.lean
      → Layout/Stage1/Lowering.lean
        → Export/Stage1/Package.lean
          → Rust prove → verify
```

## Reading a leaf interface

Leaf interfaces use functions such as `Nat → KExpr` because a
`FormalCircuit` can start at any symbolic offset. The phase assembler uses
`atOffset` to freeze each semantic input at the phase entry. A child offset
selects only the child's private variable interval; it must not select a new
semantic value or give the witness authority over that value.

Use these ownership rules:

| Value | Required circuit boundary |
|---|---|
| Internal derived value | The child owns and exports its symbolic output. Do not add an expected-output equality row. |
| Value passed between children | The parent passes the same symbolic expression. A file boundary does not add a copy row. |
| Prover claim or public value | Constrain the claimed value against the derived value. This equality is a protocol check. |

A multi-child leaf composes opaque children through `Circuit.Sequence`.
Its proof must use child contracts, builders, and footprint theorems. It must
not unfold a child's operations, recipes, or compiler state to preserve rows.

Every parent start offset comes from the preceding child's footprint theorem.
Fixed exponents such as `864` and `12960` can occur in the paper formula and
as proved theorem results. A parent allocation definition must not repeat a
child row-count literal.

## Semantic authority

```text
Spec/Folding/PiCCS/v1_1/
├── Statement.lean       ✓ prior point, Eval_K, and Eval_A input binding
├── Transcript.lean      ✓ verifier-owned pre-SumCheck replay semantics
├── EvalK.lean           ✓ Pad MLE equations; k*d coordinates
├── EvalA.lean           ✓ CCS-matrix MLE equations; k*t*d coordinates
├── FinalIdentity.lean   ✓ Eval_K + γ^(k*d) Eval_A + constraint terms
└── Accepted.lean        ✓ exact production check and conjunct coverage

Lifecycle/PaperAlgebra.lean
├── padMatrix            ✓ canonical Pad from ColumnLayout
├── padEvaluation        ✓ independent Eval_K evaluator
└── evaluationFamily     ✓ Eval_K plus all 14 Eval_A matrices
```

Evidence: [`Statement.eval_K` and `eval_A`](NightstreamFPrime/Spec/Folding/PiCCS/v1_1/Statement.lean#L45),
[`EvalK.Holds`](NightstreamFPrime/Spec/Folding/PiCCS/v1_1/EvalK.lean#L49),
[`EvalA.Holds`](NightstreamFPrime/Spec/Folding/PiCCS/v1_1/EvalA.lean#L50),
[`FinalIdentity.Holds`](NightstreamFPrime/Spec/Folding/PiCCS/v1_1/FinalIdentity.lean#L31),
[`Accepted`](NightstreamFPrime/Spec/Folding/PiCCS/v1_1/Accepted.lean#L46), and
[`accepted_iff_coverage`](NightstreamFPrime/Spec/Folding/PiCCS/v1_1/Accepted.lean#L120).
Pad is not a CCS matrix index.
## Reusable leaf circuits

```text
Gadgets/
├── Poseidon2/Duplex/Formal.lean  ✓ ordered absorb/squeeze trace
├── SumCheck/FixedChain.lean      ✓ generic round recurrence and terminal equality
├── Polynomial/Horner.lean        ✓ causal polynomial evaluation with owned output
├── Polynomial/{Power,Sparse}.lean ✓ owned powers and sparse evaluation
└── Multilinear/{PointEquality,PointWeightedHorner}.lean ✓ owned multilinear outputs
```
The Duplex child owns Poseidon2 constraints, soundness, completeness, and
footprint. The fixed-chain child owns
`p_i(0) + p_i(1) = claim_i` and `claim_(i+1) = p_i(r_i)` for indexed rounds.
Neither child owns protocol labels, transcript order, or challenge authority.
## PiCCS logical circuit

```text
Lifecycle/PiCCS/v1_1/
├── StatementBinding.lean      ✓ zero-row shared statement/input wiring
├── StatementAbsorption.lean   ✓ ordered public-statement Poseidon2 absorption
├── ChallengeDerivation.lean   ✓ 24 labelled α squeezes, then labelled γ
├── RoundTranscript.lean       ✓ indexed message absorb, then derive each r_i
├── InitialClaim.lean          ✓ T_K + γ^(k*d) T_A via causal Horner
├── SumcheckChain.lean         ✓ fixed 24-round chain; child-owned final claim
├── EvalKTerminal.lean         ✓ eq(r′,r) times 864-term Pad Horner sum
├── EvalATerminal.lean         ✓ eq(r′,r) times 12,096-term matrix Horner sum
├── CcsTerminal.lean           ✓ one-source relation-owned sparse CCS term
├── NormTerminal.lean          ✓ 17 strict-b=2 cubic residuals in gamma order
├── FinalIdentity.lean         ✓ eq(r′,α), gamma shifts, and complete Q(r′)
├── OutputBinding.lean         ✓ reduced claims and outgoing transcript state
└── Formal.lean                ✓ exact PhaseHolds soundness and completeness
```

The closed statement leaf exports `circuit`, `soundness`, `completeness`, and
footprint theorems at
[`StatementAbsorption.circuit`](NightstreamFPrime/Lifecycle/PiCCS/v1_1/StatementAbsorption.lean#L856).
Its exact footprint is 54 actions, 17,396 rate-4 chunks, 10,298,432 private
recipe variables and rows, one witness operation, and no boundary-copy row.

The challenge leaf has 50 ordered label/squeeze actions, 44,400 private
recipe variables and rows, one witness operation, and no challenge or state
copy row. Its builder derives all 24 `α` coordinates, `γ`, and the final state.

The round-transcript leaf accepts only prover message coefficients. It derives
all 24 `r_i` values and its final state from the constrained Poseidon2 trace.
Its footprint theorem is `24 * perRoundRecipeCount degreeBound`; it has one
witness operation and no challenge or state copy row.

The initial-claim leaf owns the Horner output consumed by the SumCheck chain.
`initialClaimStart_atOffset` proves the exact parent wiring. The leaf has
25,918 private variables, 25,918 rows, one witness operation, and no result
copy row. Its unconditional builder is `InitialClaim.build`.

The SumCheck chain owns the 24 equations
`p_i(0) + p_i(1) = claim_i` and exports the final `p_i(r_i)` expression.
`sumcheckStart_atOffset` proves the parent wiring. It has zero private
variables, 48 assertion rows, and no terminal copy row. `FinalIdentity` owns
the one `v = Q(r′)` obligation.

The Eval_K leaf accepts only `r′`, prior `r`, `γ`, and the 864 v1_1 Pad
coefficients. Its two opaque children own `eq(r′,r)` and the gamma-weighted
Horner sum. `EvalKTerminal.output` is their symbolic product, and
`evalKStart_atOffset` proves the parent wiring. The leaf has 1,820 private
variables, 1,820 rows, two subcircuit operations, and no intermediate or
result copy row.

The Eval_A leaf accepts only `r′`, prior `r`, `γ`, and the 12,096 v1_1 CCS
matrix coefficients. It independently owns its point equality and Horner
sum; it does not reuse or merge the Eval_K value. `EvalATerminal.output` is
the product, and `evalAStart_atOffset` proves the parent wiring. The leaf has
24,284 private variables, 24,284 rows, two subcircuit operations, and no
intermediate or result copy row.

The CCS terminal leaf accepts only the 14 fresh CCS matrix values. Its sparse
polynomial comes from the production relation, and `CcsTerminal.output`
computes the one-source CCS term as a symbolic expression. The parent passes
that expression to the final identity, and `ccsStart_atOffset` proves the
handoff location. The leaf has zero private variables, zero operations, zero
rows, and no result copy row.

The norm terminal leaf accepts `γ` and the 17 output source assignments in
exact `K + k` order. It computes
`Σ_i γ^i (x_i + 1) x_i (x_i - 1)` for strict `b = 2` and exposes
`NormTerminal.output`; no caller supplies `N`. `normStart_atOffset` proves
the parent handoff. The leaf has 32 private variables, 32 rows, one witness
operation, and no result copy row.

The final-identity leaf enforces
`v = E_K + γ^864 E_A + γ^12960 eq(r′,α) (F + γ N)`.
Its opaque children own `eq(r′,α)`, `γ^864`, and `γ^12960`; the PiCCS
parent supplies none of those values. The leaf has 27,742 private variables,
27,744 rows, five parent operations, and only the two required extension-cell
terminal assertions. `FinalIdentity.spec_implies_keyTerminal` and
`FinalIdentity.keyTerminal_implies_spec` prove both directions for the exact
production `terminalFromMessage` predicate.

The output-binding leaf absorbs the complete 27,540-word `y′` family in
17-source `K + k` order, with `Eval_K` before all 14 `Eval_A` matrices for
each source. Its owned Duplex child computes the outgoing state; no caller
supplies that state. The leaf has 4,076,512 private variables and rows, one
witness operation, and no state-copy row. `OutputBinding.build` constructs
the trace, and `OutputBinding.spec_implies_keyOutgoingState` covers the
verifier handoff.

Assembler reading order:
`Formal.lean` defines the shared carrier, child wiring, offsets, phase
predicate, and soundness composition; `Completeness/Core.lean` owns opaque
child append rules; `Completeness/Transcript.lean`,
`Completeness/Evaluation.lean`, and `Completeness/Terminal.lean` build the
three ordered child groups; `Completeness.lean` exports the one PiCCS
`FormalCircuit`. Its specification is exact `PhaseHolds`; soundness maps all
rows to `PhaseHolds`, and completeness builds all rows from `PhaseHolds`
without a caller-supplied internal child predicate.

The complete logical assembler has this proved footprint:

| Measure | Exact value |
|---|---:|
| Private symbolic variables | `14,499,140 + 24 * perRoundRecipeCount degreeBound` |
| Flattened logical rows | `14,499,190 + 24 * perRoundRecipeCount degreeBound` |
| Private variables if `degreeBound = 4` | `14,584,388` |
| Logical rows if `degreeBound = 4` | `14,584,438` |

`Formal.localLength_eq` and `Formal.flatConstraints_length_eq` prove the
parameterized values. `Formal.privateCount_eq_of_degreeBound_eq_four` and
`Formal.rowCount_eq_of_degreeBound_eq_four` prove the conditional numeric
values. The parent sums certified child metadata. It does not evaluate the
14-million-row list in the kernel.

These are logical counts. They are not a physical layout, a column-reuse
proof, or a domain theorem. `Layout/PiCCS/v1_1/` must prove those results and
connect them to the Stage 1 `2^25` ledger.

## PiRLC logical circuit

```text
Lifecycle/PiRLC/v1_1/
├── InputBinding.lean           ○ bind 17 PiCCS output claims
├── TranscriptAbsorption.lean   ○ absorb the complete y′ family
├── StrongSetSampling.lean      ○ derive ρ₁…ρ₁₇ from the transcript
├── ChallengeMembership.lean    ○ prove every ρ_i is in the allowed set
├── CommitmentCombination.lean ○ Σ_i ρ_i · C_i
├── PublicInputCombination.lean ○ Σ_i ρ_i · x_i
├── EvalKCombination.lean       ○ indexed combination of 17 Eval_K values
├── EvalACombination.lean       ○ indexed combination of 17 Eval_A families
├── OutputBinding.lean          ○ bind the one combined parent claim
└── Formal.lean                 ○ only PiRLC FormalCircuit and coverage composition
```

The assembler uses indexed composition. It does not copy 17 constraint blocks.

## PiDEC logical circuit

```text
Lifecycle/PiDEC/v1_1/
├── InputBinding.lean        ○ bind the PiRLC parent
├── ParentNormCheck.lean     ○ reject a parent public input outside CE(B′)
├── SplitB.lean              ○ construct 16 signed base-2 components
├── DigitRange.lean          ○ digit range and low-norm checks
├── Recombination.lean       ○ parent = Σ_j 2^j child_j
├── CommitmentRelation.lean ○ commitment recombination
├── EvalKRelation.lean       ○ separate Eval_K recombination
├── EvalARelation.lean       ○ separate Eval_A recombination
├── OutputChildren.lean      ○ indexed construction of 16 children
├── OutputBinding.lean       ○ bind all child claims
└── Formal.lean              ○ only PiDEC FormalCircuit and coverage composition
```

The assembler uses indexed composition. It does not copy 16 constraint blocks.

## Stage 1 assembly, layout, and export

```text
Lifecycle/
├── PriorStateHash.lean          △ prior-state and recursive-input binding
├── OutputHash.lean              △ output-state and public-output binding
├── Pilot.lean                   △ closed hash-slot composition only
└── Stage1/
    ├── Interface.lean           ○ cross-phase symbolic interfaces
    ├── Formal.lean              ○ the only production logical FormalCircuit
    └── Soundness.lean           ○ circuit rows imply StepHolds/TerminalHolds

Layout/
├── PiCCS/v1_1/{Lowering,Preservation}.lean ◐ structural lowering and soundness
├── PiRLC/v1_1/{Lowering,Preservation}.lean ○
├── PiDEC/v1_1/{Lowering,Preservation}.lean ○
└── Stage1/{Lowering,Ownership,Preservation}.lean ○

Export/Stage1/
├── WitnessProgram.lean ○ one Rust-interpreted witness IR
├── Package.lean        ○ one package from the proved Stage 1 lowering
└── Emit.lean           ○ the sole production emitter
```

`Stage1/Formal.lean` owns cross-phase wiring only: transcript states, public
bindings, PiCCS outputs to 17 PiRLC inputs, the PiRLC parent to PiDEC, 16
PiDEC children to the next running accumulator, application state, hashes,
and terminal checks. A parent uses child contracts and footprint theorems; it
must not unfold child operations. File boundaries add no automatic copy rows.

The current PiCCS layout has one `R1CS.LoweringPlan` over the sole logical
circuit. `physical_implies_phaseHolds` proves that its physical rows imply the
exact PiCCS `PhaseHolds`. Its row and fresh-column costs remain structural
functions. Numeric physical costs, leaf row ownership, multiplication-witness
completeness, reuse, and the cumulative `2^25` ledger are still open.

Physical leaf packets close in logical order. Statement binding is complete:
`Leaves.StatementBinding.freshColumnCount_eq` proves zero fresh columns and
`Leaves.StatementBinding.physicalRowCount_eq` proves zero physical rows.
Statement absorption is also complete under its explicit affine-input
boundary. `Layout.Poseidon2.Duplex.compile_recipes_direct` proves that each
Duplex recipe lowers to one direct R1CS row without schedule evaluation.
`Leaves.StatementAbsorption.freshColumnCount_eq` proves zero fresh columns,
and `Leaves.StatementAbsorption.physicalRowCount_eq` proves exactly
10,298,432 physical rows. Challenge derivation is complete under its incoming
state-affinity boundary. `Layout.Poseidon2.Duplex.compile_samples_affine`
proves that all 25 expected values are compiler-derived affine samples;
there are no expected-sample copy rows.
`Leaves.ChallengeDerivation.freshColumnCount_eq` proves zero fresh columns,
and `Leaves.ChallengeDerivation.physicalRowCount_eq` proves exactly 44,400
physical rows. The fixed round transcript is also complete under its incoming
state and prover-message affinity boundary. One generic round proof composes
over 24 indices. Its physical row count is
`24 * perRoundRecipeCount degreeBound`, with zero fresh columns; when
`degreeBound = 4`, `physicalRowCount_eq_of_degreeBound_eq_four` proves exactly
85,248 rows. Initial claim is complete under its production wire-shape
boundary. Its 12,959 extension multiplications retain 25,918 logical recipe
variables. The current R1CS lowering adds 90,713 intermediate columns and
uses 116,631 physical rows. `physicalPrivateColumnCount_eq` proves 116,631
total private columns for this leaf. The fixed SumCheck chain is complete
under its production
wire-shape boundary. Its 48 logical equality rows contain no logical witness
variables. Structural lowering of one generic degree-4 round, composed over
24 indices, proves 11,053 intermediate columns and 11,101 physical rows.
The separate `Eval_K` terminal is also complete. Its reusable physical tree is:

```text
Eval_K terminal
├── point equality over 24 coordinates
│   └── 94 logical + 569 lowering columns = 663 rows
├── Horner over 864 Pad-family coefficients
│   └── 1,726 logical + 6,041 lowering columns = 7,767 rows
└── parent wiring
    └── 0 copy columns and 0 copy rows
```

`Leaves.EvalKTerminal.freshColumnCount_eq` proves 6,610 lowering
columns. `physicalPrivateColumnCount_eq` and `physicalRowCount_eq` each prove
8,430. The Pad-family value stays separate from every `Eval_A` matrix value.
The separate `Eval_A` terminal reuses the same point-equality child, then
evaluates only the 12,096 CCS-matrix-family coefficients:

```text
Eval_A terminal
├── point equality over 24 coordinates
│   └── 94 logical + 569 lowering columns = 663 rows
├── Horner over 12,096 matrix-family coefficients
│   └── 24,190 logical + 84,665 lowering columns = 108,855 rows
└── parent wiring
    └── 0 copy columns and 0 copy rows
```

`Leaves.EvalATerminal.freshColumnCount_eq` proves 85,234 lowering
columns. `physicalPrivateColumnCount_eq` and `physicalRowCount_eq` each prove
109,518. The final-identity leaf, not this leaf, owns the `gamma^(k*d)` shift.
The CCS terminal is a shared symbolic-expression leaf:

```text
CCS terminal F
├── relation-owned sparse constraint polynomial
├── 14 fresh-source matrix inputs
└── symbolic output consumed by final identity
    └── 0 logical columns, 0 lowering columns, 0 rows
```

`Leaves.CcsTerminal.physicalPrivateColumnCount_eq` and
`physicalRowCount_eq` prove zero. This is not an omitted check: the later
final-identity rows constrain the returned sparse-polynomial expression.
The strict base-2 norm terminal has this physical tree:

```text
Norm terminal N
├── 17 cubic residual expressions: (x_i + 1) x_i (x_i - 1)
├── 16 indexed gamma-Horner transitions
│   └── each: 2 logical columns + 45 lowering columns = 47 rows
└── parent output sharing
    └── 0 copy columns and 0 copy rows
```

`Leaves.NormTerminal.compile_totalFreshCount` and
`compile_totalRowCount` prove the indexed composition without expanding the
17-source schedule in the kernel. The fixed leaf has 32 logical columns, 720
lowering columns, 752 total private columns, and 752 rows. The remaining 2
physical leaf packets are open.

The current pilot preservation and package proofs are
[`Layout.Pilot.physical_implies_spec`](NightstreamFPrime/Layout/Pilot.lean#L173)
and [`Export.Pilot.canonicalPackage_implies_spec`](NightstreamFPrime/Export/Pilot.lean#L797).
They are evidence for the closed pilot, not the final Stage 1 assembler.
