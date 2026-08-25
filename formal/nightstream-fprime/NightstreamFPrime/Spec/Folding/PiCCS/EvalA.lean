import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

/-!
Paper authority: SuperNeo v1.1, Section 7.3 `Eval_A`; Appendix B.2,
Equation (10) and Item 4.
Obligation: Every carried CCS-matrix coefficient equals the multilinear
evaluation of that genuine matrix image at the prior point.

Inputs:
- the prior point;
- running assignments;
- verifier-owned CCS-matrix coefficient matrices;
- public `Eval_A` coefficients.

Outputs:
- the canonical `MatrixEvaluationResidual.AllClaimsHold` predicate.

Parent coverage:
- `UnifiedSources.UnifiedInputs.SemanticTruth`, fourth conjunct.

`Pad` is not in the matrix index. This module owns the canonical named
CCS-matrix-evaluation contract. It copies no formula and emits no circuit
constraints.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.EvalA

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uExtension

/-- The existing exact data for the paper's CCS-matrix evaluation family. -/
abbrev Data (Extension : Type uExtension) (shape : Shape) (columns : Nat) :=
  MatrixEvaluationResidual.EvaluationData F Extension shape columns

/-- One exact `Eval_A` coordinate equation. -/
abbrev CoordinateHolds
    {Extension : Type uExtension}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F → Extension)
    {shape : Shape}
    {columns : Nat}
    (data : Data Extension shape columns)
    (coordinate : MatrixCoordinate shape) : Prop :=
  MatrixEvaluationResidual.EvaluationClaimHolds
    baseOps extensionOps lift data coordinate

/-- All `k*t*d` exact `Eval_A` coordinate equations. -/
abbrev Holds
    {Extension : Type uExtension}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F → Extension)
    {shape : Shape}
    {columns : Nat}
    (data : Data Extension shape columns) : Prop :=
  MatrixEvaluationResidual.AllClaimsHold baseOps extensionOps lift data

/-- The complete PiCCS semantic truth contains this exact `Eval_A` leaf. -/
theorem of_semanticTruth
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedSources.UnifiedInputs Extension shape columns)
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F → Extension)
    (truth : data.SemanticTruth baseOps extensionOps lift) :
    Holds baseOps extensionOps lift data.matrixData :=
  truth.2.2.2

/-- The exact circuit-facing residual orientation, claimed minus computed,
is zero if and only if every `Eval_A` equation holds. -/
theorem allResidualsZero_iff_holds
    {Extension : Type uExtension}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (lift : F → Extension)
    {shape : Shape}
    {columns : Nat}
    (data : Data Extension shape columns) :
    (∀ coordinate,
        MatrixEvaluationResidual.residual
          baseOps extensionOps lift data coordinate = extensionOps.zero) ↔
      Holds baseOps extensionOps lift data :=
  MatrixEvaluationResidual.allResidualsZero_iff_allClaimsHold
    baseOps extensionOps extensionLaws lift data

/-- The canonical `Eval_A` traversal has exactly `k*t*d` coordinates. -/
theorem coordinateCount
    {Extension : Type uExtension}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F → Extension)
    {shape : Shape}
    {columns : Nat}
    (data : Data Extension shape columns) :
    (MatrixEvaluationResidual.orderedResiduals
      baseOps extensionOps lift data).length = shape.matrixEvaluationCount :=
  MatrixEvaluationResidual.orderedResiduals_length
    baseOps extensionOps lift data

end NightstreamFPrime.Spec.Folding.PiCCS.EvalA
