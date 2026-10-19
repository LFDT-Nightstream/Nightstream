import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CausalExecution
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.SignedMixingProbability

/-!
Owns `testError`, the PiCCS test error of SuperNeo v1.2 Appendix B.2,
equation (14), for one execution over the extension field: the round losses
`cubeVariables · width / p²` and the α and γ losses
`(cubeVariables + jointCoefficientCount − 1) / p²`. `RandomOracleTest`
charges it once for each oracle query.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.IndependentExecution

open scoped BigOperators
open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open ConcreteCarrier StrongReduction CausalExecution

attribute [local instance] Classical.propDecidable

universe uCommitment uPublicInput

/-- The two interactive algebraic losses in the paper's test bound. -/
noncomputable def testError (shape : Shape) (width : Nat) : ℝ :=
  (shape.cubeVariables : ℝ) * width / (goldilocksModulus ^ 2 : Nat) +
    ((shape.jointCoefficientCount - 1 + shape.cubeVariables : Nat) : ℝ) /
      (goldilocksModulus ^ 2 : Nat)

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.IndependentExecution
