import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring.Basic
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Coefficients
import NightstreamFPrime.Spec.Profile
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.IndependentExecution

/-! Per-call error counts of the light fold, SuperNeo v1.2 Appendix B.2, equation (14), at the
light shape: one fresh source and one running source.

- Event: the light Π_CCS accepts while `Light.accepted_implies_securityOutcome` gives a
  mixing root or a sum-check failure. Experiment: `α`, `γ` and the round challenges drawn
  from `K`. Per-call bound: `testNumerator / |K|`, 524 at the production cube against 4,589
  for the SuperNeo fold.
- Event: the two `ρ` of a light fork collide. Experiment: one coefficient drawn from the
  strong set `C`. Per-call bound: `forkCoordinates / |C|`, one coordinate against SuperNeo's 17.

A chain of light folds adds both bounds once per fold; the fold count stays a parameter. -/

namespace NightstreamFPrime.Spec.Folding.Light.ErrorBudget

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def shape (cubeVariables : Nat) : Shape where
  cubeVariables := cubeVariables
  freshCount := 1
  runningCount := 1
  matrixCount := productionProfile.ccsMatrices
  coefficientCount := productionProfile.ringDegree

/-- SuperNeo v1.2 Appendix B.2, equation (14). -/
def testNumerator (shape : Shape) (width : Nat) : Nat :=
  shape.cubeVariables * width + (shape.jointCoefficientCount - 1 + shape.cubeVariables)

def forkCoordinates : Nat := (shape 0).freshCount

theorem production_testNumerator : testNumerator (shape 28) 8 = 524 := by
  rfl

theorem production_forkCoordinates : forkCoordinates = 1 := by
  rfl

/-- The per-call test error of SuperNeo v1.2 Appendix B.2 is `testNumerator / |K|`. -/
theorem testError_eq (shape : Shape) (width : Nat) :
    IndependentExecution.testError shape width =
      (testNumerator shape width : ℝ) / (goldilocksModulus : ℝ) ^ 2 := by
  unfold IndependentExecution.testError testNumerator
  push_cast
  ring

theorem production_testError :
    IndependentExecution.testError (shape 28) 8 = (524 : ℝ) / (goldilocksModulus : ℝ) ^ 2 := by
  rw [testError_eq, production_testNumerator]
  norm_num

end NightstreamFPrime.Spec.Folding.Light.ErrorBudget
