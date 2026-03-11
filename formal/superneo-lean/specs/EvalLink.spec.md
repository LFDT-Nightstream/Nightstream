# EvalLink Spec

## Purpose

- **What it is**: The eval-link layer formalizes the matrix-vector product evaluation identity (Remark 2 style). It defines `evalLinkIdentity` / `evalLinkIdentityProp` (delegating to `matrixTransformIdentity`), the universal contract `evalLinkAssumption`, and the check-facing `evalLinkCheckAssumption`.
- **Key property**: `∀ z, MatrixRowsCompatible m z → evalLinkIdentityProp bar m z` — i.e. the matrix-vector product evaluation holds for all compatible inputs.
- **Protocol role**: EvalHom depends on `evalLinkAssumption`. ProtocolTarget uses eval-link for the evaluation-hom pipeline.

## Target Formulas (Paper → Lean)

- Paper formula: `Mz = ct(M̄z)` (Remark 2, Matrix-Vector Product Evaluation)
- Lean mapping:
  - `evalLinkIdentity bar m z` : executable check (delegates to `matrixTransformIdentity`)
  - `evalLinkIdentityProp bar m z` : proposition counterpart
  - `evalLinkAssumption bar m` : `∀ z, MatrixRowsCompatible m z → evalLinkIdentityProp bar m z`
  - `evalLinkCheckAssumption bar m` : `∀ z, MatrixRowsCompatible m z → evalLinkIdentity bar m z = true`
- Target statement: `evalLinkIdentity bar m z = true ↔ evalLinkIdentityProp bar m z` (sound/complete/iff_prop proved).

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Anchors:
  - Remark 2 (Matrix-vector Product Evaluation), Section 5, lines 388-389

## Module Mapping

- Implementation: `SuperNeo.EvalLink`
- Interface: `SuperNeo.EvalLinkInterface`

## Contract Surface

| Contract group | Lean surface | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Eval-link identity | `evalLinkIdentity`, `evalLinkIdentityProp` | None | Delegates to `matrixTransformIdentity` | Theorem-Target | `EvalHom.lean` |
| Theorem-facing boundary | `evalLinkAssumption bar m` | None | `∀ z, MatrixRowsCompatible m z → evalLinkIdentityProp bar m z` | Theorem-Target | `EvalHom.lean`, `ProtocolTarget` |
| Check-facing boundary | `evalLinkCheckAssumption bar m` | None | `∀ z, MatrixRowsCompatible m z → evalLinkIdentity bar m z = true` | Theorem-Target | — |
| Sound/complete | `evalLinkIdentity_sound`, `evalLinkIdentity_complete`, `evalLinkIdentity_iff_prop` | Check true / Prop holds | Bidirectional bridge | Theorem-Target | — |
| From MatrixTransform | `evalLinkAssumption_of_matrixTransformAssumption` | `matrixTransformAssumption bar m` | `evalLinkAssumption bar m` | Theorem-Target | — |
| From Thm3Core | `evalLinkAssumption_of_thm3CoreAssumption` | `thm3CoreAssumption bar` | `evalLinkAssumption bar m` | Theorem-Target | — |
| From P10 | `evalLinkAssumption_of_p10` | `thm3CoreAssumption bar` | `evalLinkAssumption bar m` | Theorem-Target | — |
| From P10+P11 | `evalLinkAssumption_of_p10_p11` | `thm3CoreAssumption bar`, `barLiftLinearityAssumption bar` | `evalLinkAssumption bar m` (compatibility path) | Theorem-Target | — |

## Proof Obligations and Closure Plan

All obligations closed. Eval-link delegates to MatrixTransform; all bridge theorems (`_of_checkAssumption`, `_of_assumption`, `_iff`, `_of_matrixTransformAssumption`, `_of_thm3CoreAssumption`, `_of_p10`, `_of_p10_p11`) are proved.

## Assumption Ledger

No open boundary assumptions in this module.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/MatrixTransform.lean`: imports `matrixTransformIdentity`, `matrixTransformIdentityProp`, `matrixTransformAssumption` for delegation and constructors.
- Downstream consumers:
  - `SuperNeo/EvalHom.lean`: uses `evalLinkAssumption` to derive `evalHomAssumption` (Theorem 5).
  - `SuperNeo/ProtocolTarget.lean`: depends on eval-link for the evaluation-hom pipeline.

## Implementation Plan

1. `evalLinkIdentity` / `evalLinkIdentityProp` delegate to `matrixTransformIdentity` / `matrixTransformIdentityProp`.
2. Sound/complete/iff_prop proved via MatrixTransform theorems.
3. Assumption bridges proved via universal quantification and sound/complete.
4. Constructors from MatrixTransform, Thm3Core, and P10/P10+P11 chains flow through `evalLinkAssumption_of_matrixTransformAssumption`.

## Quality Expectations

- No `sorry` in any theorem.
- Eval-link is the single semantic surface for matrix-vector evaluation; check/prop duality is explicit.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. All eval-link theorems exported through the interface.

## Out of Scope

- Full evaluation-hom instantiation (belongs to EvalHom).
- Non-identity bar-lift (belongs to embedding/bar-lift extension).
