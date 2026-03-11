# ArithmeticObligations Spec

## Purpose

- **What it is**: A structure bundling arithmetic obligations from Sections 4–5 (decomposition, matrix transform, eval homomorphism, module homomorphisms, sampling, MLE, interpolation) for consumption by Section 7 protocol layers.
- **Key property**: Under the bundle, `evalHom` is derivable from `P10` and module-hom assumptions (`of_p10`), with a compatibility constructor retaining `(P10 + P11)` signature (`of_p10_p11`); `mleIdentityAtR` follows from table-size precondition; `splitTerminalZero` from scalar bound.
- **Protocol role**: ProtocolTarget and downstream reductions depend on ArithmeticObligations to derive `protocolTargetProp` and compose the protocol skeleton.

## Target Formulas (Paper → Lean)

- `ArithmeticObligations bar m r ... → splitBase2TerminalZeroProp splitScalar kSplit`
- `splitBase2LowPartNat + (2^kSplit) * splitBase2TerminalQuot = splitScalar.val` (split decomposition)
- `qVals.size = 2^r.size → mleEval qVals r = mleInnerProductForm qVals r`
- `evalHomAssumption_of_p10_and_moduleAssumptions → evalHomAssumption`

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Anchors:
  - Section 7 (Neo's folding scheme for CCS), lines 447–467: Relations and structure (Definitions 11–12)
  - Section 4–5 preliminaries: decomposition, matrix transform, eval homomorphism, MLE, interpolation

## Module Mapping

- Implementation: `SuperNeo.ArithmeticObligations`
- Interface: `SuperNeo.ArithmeticObligationsInterface`

## Contract Surface

| Contract group | Lean surface | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Structure | `ArithmeticObligations` | None | Bundles splitScalarBelowPow, evalHom, vecModule, scalarModule, sampling, mleTableSize, mleIdentityAtR, interpolation | Definitional | ProtocolTarget |
| Constructor | `ArithmeticObligations.of_p10` | thm3, vec/scalar module, sampling, mleSize, interp | `ArithmeticObligations` with evalHom from P10 | Theorem-Target | — |
| Constructor (compat) | `ArithmeticObligations.of_p10_p11` | thm3, barLift, vec/scalar module, sampling, mleSize, interp | `ArithmeticObligations` via `of_p10` | Theorem-Target | — |
| Terminal zero | `ArithmeticObligations.splitTerminalZero` | `ArithmeticObligations` | `splitBase2TerminalZeroProp splitScalar kSplit` | Theorem-Target | ProtocolTarget |
| Split decomp | `splitDecompositionNat_of_obligations` | `ArithmeticObligations` | `splitBase2LowPartNat + (2^kSplit)*splitBase2TerminalQuot = splitScalar.val` | Theorem-Target | — |
| MLE from assumption | `mleIdentityAtR_of_assumption` | `qVals.size = 2^r.size`, `mleIdentityAssumption` | `mleEval qVals r = mleInnerProductForm qVals r` | Theorem-Target | — |
| MLE from size | `mleIdentityAtR_of_size` | `qVals.size = 2^r.size` | `mleEval qVals r = mleInnerProductForm qVals r` | Theorem-Target | — |

## Proof Obligations and Closure Plan

All obligations closed. `ArithmeticObligations.of_p10` derives evalHom from P10 and module assumptions; `of_p10_p11` is a compatibility wrapper. `splitTerminalZero` and `splitDecompositionNat_of_obligations` are proved. MLE identity follows from `mleIdentityAssumption_holds` or global `mleIdentityAssumption`.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/Decomp.lean`: imports decomposition and split predicates
  - `SuperNeo/MatrixTransform.lean`: imports matrix transform
  - `SuperNeo/EvalHom.lean`: imports evalHomAssumption
  - `SuperNeo/ModuleHom.lean`: imports vec/scalar module assumptions
  - `SuperNeo/SamplingSet.lean`: imports samplingExpansionProp
  - `SuperNeo/MLE.lean`: imports mleEval, mleInnerProductForm, mleIdentityAssumption_holds
  - `SuperNeo/Interp.lean`: imports interpolationProp
- Downstream consumers:
  - `SuperNeo/ProtocolTarget.lean`: uses `ArithmeticObligations` to define `ProtocolTargetAssumptions` and derive `protocolTargetProp`

## Implementation Plan

1. `ArithmeticObligations` structure bundles all arithmetic obligations except direct invertibility witnesses, which now live at `ProtocolTarget`.
2. `of_p10` constructor derives evalHom via `evalHomAssumption_of_p10_and_moduleAssumptions`; `of_p10_p11` wraps `of_p10`.
3. `splitTerminalZero` and `splitDecompositionNat_of_obligations` proved from definitions.
4. `mleIdentityAtR_of_assumption` and `mleIdentityAtR_of_size` bridge MLE theorem surface.

## Quality Expectations

- No `sorry` in any theorem.
- All declarations proved natively.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. All theorems exported through the interface.

## Out of Scope

- Concrete instantiation of upstream assumptions (Thm3, barLift, etc.); those belong to their respective modules.
