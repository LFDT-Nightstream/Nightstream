import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Coefficients

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

inductive EvaluationResidualOrientation where
  | targetMinusEvaluation
  | evaluationMinusTarget
deriving Repr, DecidableEq

/-- SuperNeo v1.2 Appendix B.2. -/
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
