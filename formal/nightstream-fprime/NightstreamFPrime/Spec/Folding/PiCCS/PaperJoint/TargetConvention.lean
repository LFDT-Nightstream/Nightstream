import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Coefficients

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/TargetConvention.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Owns the exact SuperNeo v1.1 target exponent convention. Pad coefficients use
`I_K`; matrix coefficients use `k*d + I_A`. There is no selectable v1.0
layout.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Sign orientation for both evaluation residual families. -/
inductive EvaluationResidualOrientation where
  | targetMinusEvaluation
  | evaluationMinusTarget
deriving Repr, DecidableEq

/-- The v1.1 residual orientation from Appendix B.2. -/
def selectedEvaluationResidualOrientation : EvaluationResidualOrientation :=
  .targetMinusEvaluation

/-- Pad target coefficients start at exponent zero. -/
theorem padTargetExponent_eq_local
    {shape : Shape}
    (coordinate : PadCoordinate shape) :
    coordinate.gammaExponent = coordinate.localGammaExponent := by
  rfl

/-- Matrix target coefficients start after all `k*d` Pad coefficients. -/
theorem matrixTargetExponent_eq_shifted
    {shape : Shape}
    (coordinate : MatrixCoordinate shape) :
    coordinate.gammaExponent =
      shape.padEvaluationCount + coordinate.localGammaExponent := by
  rfl

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
