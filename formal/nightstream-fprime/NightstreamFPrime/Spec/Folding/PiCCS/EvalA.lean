import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

/-!
Paper authority: SuperNeo v1.2, Section 7.3 `Eval_A`; Appendix B.2,
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
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.EvalA

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uExtension

abbrev Data (Extension : Type uExtension) (shape : Shape) (columns : Nat) :=
  MatrixEvaluationResidual.EvaluationData F Extension shape columns

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

abbrev Holds
    {Extension : Type uExtension}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F → Extension)
    {shape : Shape}
    {columns : Nat}
    (data : Data Extension shape columns) : Prop :=
  MatrixEvaluationResidual.AllClaimsHold baseOps extensionOps lift data

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
