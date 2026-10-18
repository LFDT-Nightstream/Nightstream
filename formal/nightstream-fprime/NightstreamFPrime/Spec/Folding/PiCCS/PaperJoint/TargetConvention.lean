import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Coefficients

/-!
Owns the exact SuperNeo v1.2 target exponent convention. Pad coefficients use
`I_K`; matrix coefficients use `k*d + I_A`.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

inductive EvaluationResidualOrientation where
  | targetMinusEvaluation
  | evaluationMinusTarget
deriving Repr, DecidableEq

/-- The v1.2 residual orientation from Appendix B.2. -/
def selectedEvaluationResidualOrientation : EvaluationResidualOrientation :=
  .targetMinusEvaluation

theorem padTargetExponent_eq_local
    {shape : Shape}
    (coordinate : PadCoordinate shape) :
    coordinate.gammaExponent = coordinate.localGammaExponent := by
  rfl

theorem matrixTargetExponent_eq_shifted
    {shape : Shape}
    (coordinate : MatrixCoordinate shape) :
    coordinate.gammaExponent =
      shape.padEvaluationCount + coordinate.localGammaExponent := by
  rfl

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
