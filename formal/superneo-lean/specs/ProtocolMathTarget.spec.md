# ProtocolMathTarget Spec

## Purpose

- **What it is**: The first protocol-facing math target extracted from the arithmetic bundle; the bridge interface that the eventual protocol proof consumes. Defines `protocolMathTargetProp` (arithmetic-only) and `protocolMathTargetWithThm3Prop` (P10 + arithmetic).
- **Key property**: `arithmeticBundleProp → protocolMathTargetProp`; `p10CoreProp ∧ arithmeticBundleProp → protocolMathTargetWithThm3Prop`; check-driven preconditions imply both via soundness.
- **Protocol role**: ProtocolReduction and ProtocolRelations use these targets to derive CE valid and compose the protocol skeleton.

## Target Formulas (Paper → Lean)

- `protocolMathTargetProp bar m z z1 z2 zDecomp r ... ↔ arithmeticDecompProp ∧ MatrixRowsCompatible ∧ matrixVecDirect = matrixVecCtBar ∧ arithmeticEvalHomProp ∧ invertibilityPreconditionsProp ∧ arithmeticSamplingProp ∧ arithmeticPolyProp ∧ arithmeticInterpProp`
- `protocolMathTargetWithThm3Prop bar a b m z ... ↔ p10CoreProp bar a b ∧ protocolMathTargetProp bar m z ...`
- `arithmeticBundleProp ... → protocolMathTargetProp ...`
- `p10CoreProp ∧ arithmeticBundleProp → protocolMathTargetWithThm3Prop`
- `p10CoreCheck = true ∧ checks ... → protocolMathTargetWithThm3Prop` (soundness)

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Anchors:
  - Section 7 (Neo's folding scheme for CCS), lines 447–467: Relations (Definitions 11–13)
  - Section 7.2–7.5, lines 467–596: Folding scheme via interactive reductions (Π_CCS, Π_RLC, Π_DEC)

## Module Mapping

- Implementation: `SuperNeo.ProtocolMathTarget`
- Interface: `SuperNeo.ProtocolMathTargetInterface`

## Contract Surface

| Contract group | Lean surface | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Math target | `protocolMathTargetProp` | None | Conjunction of arithmetic decomp, matrix, eval hom, invertibility, sampling, poly, interp | Definitional | ProtocolReduction |
| With Thm3 | `protocolMathTargetWithThm3Prop` | None | `p10CoreProp ∧ protocolMathTargetProp` | Definitional | ProtocolReduction |
| From bundle | `protocolMathTargetProp_of_arithmeticBundle` | `arithmeticBundleProp` | `protocolMathTargetProp` | Theorem-Target | — |
| From P10+bundle | `protocolMathTargetWithThm3Prop_of_p10_arithmeticBundle` | `p10CoreProp`, `arithmeticBundleProp` | `protocolMathTargetWithThm3Prop` | Theorem-Target | — |
| From preconditions | `protocolMathTargetWithThm3Prop_of_thm3_preconditions` | IsDBarMatrix, IsDVec, p10CoreCheck, arithmeticBundleProp | `protocolMathTargetWithThm3Prop` | Theorem-Target | — |
| From Thm3 assumption | `protocolMathTargetWithThm3Prop_of_thm3_assumption` | IsDBarMatrix, IsDVec, thm3CoreAssumption, arithmeticBundleProp | `protocolMathTargetWithThm3Prop` | Theorem-Target | ProtocolReduction |
| From checks | `protocolMathTargetProp_of_checks`. `protocolMathTargetWithThm3Prop_of_checks` | Check-driven assumptions (P6, P12, P14, module, P17, P18, P19) | `protocolMathTargetProp` / `protocolMathTargetWithThm3Prop` | Theorem-Target | — |

## Proof Obligations and Closure Plan

All obligations closed. `protocolMathTargetProp_of_arithmeticBundle` projects from arithmetic bundle. `protocolMathTargetWithThm3Prop_of_*` variants compose P10 with arithmetic. Check-driven variants use `arithmeticBundleProp_of_checks` and `p10CoreCheck_sound`.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/ArithmeticBundle.lean`: imports `arithmeticBundleProp`
  - `SuperNeo/Thm3Core.lean`: imports `p10CoreProp`, `p10CoreCheck`, `thm3CoreAssumption`
- Downstream consumers:
  - `SuperNeo/ProtocolReduction.lean`: uses `protocolMathTargetProp`, `protocolMathTargetWithThm3Prop`, `protocolMathTargetProp_of_arithmeticBundle`, `protocolMathTargetWithThm3Prop_of_checks`, `protocolMathTargetWithThm3Prop_of_thm3_assumption`
  - `SuperNeo/ProtocolRelations.lean`: uses `protocolMathTargetProp_to_CEValid`, `protocolMathTargetWithThm3Prop_to_CEValid`

## Implementation Plan

1. `protocolMathTargetProp` and `protocolMathTargetWithThm3Prop` defined as conjunctions.
2. `protocolMathTargetProp_of_arithmeticBundle` projects from arithmetic bundle.
3. `protocolMathTargetWithThm3Prop_of_*` variants compose P10 with arithmetic.
4. Check-driven variants use `arithmeticBundleProp_of_checks` and `p10CoreCheck_sound`.

## Quality Expectations

- No `sorry` in any theorem.
- All declarations proved natively.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. All surfaces exported through the interface.

## Out of Scope

- Concrete instantiation of arithmetic bundle or P10; those belong to their respective modules.
