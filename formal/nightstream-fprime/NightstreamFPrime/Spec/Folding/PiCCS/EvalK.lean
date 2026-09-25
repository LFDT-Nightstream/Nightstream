import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

/-!
Paper authority: SuperNeo v1.1, Section 7.3 `Eval_K`; Appendix B.2,
Equation (9) and Item 3.
Obligation: Every carried Pad coefficient equals the multilinear evaluation
of the canonical Pad image at the prior point.

Inputs:
- the prior point;
- running assignments;
- verifier-owned Pad coefficient matrices;
- public `Eval_K` coefficients.

Outputs:
- the canonical `PadEvaluationResidual.AllClaimsHold` predicate.

Parent coverage:
- `UnifiedSources.UnifiedInputs.SemanticTruth`, third conjunct.

This module owns the canonical named Pad-evaluation contract. It copies no
formula and emits no circuit constraints.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.EvalK

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uExtension

/-- The existing exact data for the paper's Pad evaluation family. -/
abbrev Data (Extension : Type uExtension) (shape : Shape) (columns : Nat) :=
  PadEvaluationResidual.EvaluationData F Extension shape columns

/-- One exact `Eval_K` coordinate equation. -/
abbrev CoordinateHolds
    {Extension : Type uExtension}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F → Extension)
    {shape : Shape}
    {columns : Nat}
    (data : Data Extension shape columns)
    (coordinate : PadCoordinate shape) : Prop :=
  PadEvaluationResidual.EvaluationClaimHolds
    baseOps extensionOps lift data coordinate

/-- All `k*d` exact `Eval_K` coordinate equations. -/
abbrev Holds
    {Extension : Type uExtension}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F → Extension)
    {shape : Shape}
    {columns : Nat}
    (data : Data Extension shape columns) : Prop :=
  PadEvaluationResidual.AllClaimsHold baseOps extensionOps lift data

/-- The complete PiCCS semantic truth contains this exact `Eval_K` leaf. -/
theorem of_semanticTruth
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedSources.UnifiedInputs Extension shape columns)
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F → Extension)
    (truth : data.SemanticTruth baseOps extensionOps lift) :
    Holds baseOps extensionOps lift data.padData :=
  truth.2.2.1

/-- The exact circuit-facing residual orientation, claimed minus computed,
is zero if and only if every `Eval_K` equation holds. -/
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
        PadEvaluationResidual.residual
          baseOps extensionOps lift data coordinate = extensionOps.zero) ↔
      Holds baseOps extensionOps lift data :=
  PadEvaluationResidual.allResidualsZero_iff_allClaimsHold
    baseOps extensionOps extensionLaws lift data

/-- The canonical `Eval_K` traversal has exactly `k*d` coordinates. -/
theorem coordinateCount
    {Extension : Type uExtension}
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F → Extension)
    {shape : Shape}
    {columns : Nat}
    (data : Data Extension shape columns) :
    (PadEvaluationResidual.orderedResiduals
      baseOps extensionOps lift data).length = shape.padEvaluationCount :=
  PadEvaluationResidual.orderedResiduals_length
    baseOps extensionOps lift data

end NightstreamFPrime.Spec.Folding.PiCCS.EvalK
