import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-! SuperNeo v1.2 Section 5, Theorems 8 and 9. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RingCoefficientKernel

open NightstreamFPrime.Spec
open MatrixCoefficientSource

def nativeBarEntry (output input : Fin ringDegree) : F :=
  if output.val = 0 then
    if input.val = 0 then 1 else 0
  else if output.val < ringMiddleDegree then
    if input.val = ringMiddleDegree - output.val ∨
        input.val = ringDegree - output.val then
      -1
    else
      0
  else if input.val = ringDegree - output.val then
    -1
  else
    0

def barBasis (input : Fin ringDegree) : RingF :=
  fun output => nativeBarEntry output input

def constant : Fin ringDegree :=
  ⟨0, by decide⟩

theorem barBasis_constant_eq_one :
    barBasis constant = ringFOne := by
  funext output
  by_cases outputZero : output.val = 0
  · simp [barBasis, nativeBarEntry, constant, ringFOne, ringFMonomial,
      outputZero]
  · have outputPositive : 0 < output.val := Nat.pos_of_ne_zero outputZero
    have outputLt : output.val < ringDegree := output.isLt
    by_cases outputLow : output.val < ringMiddleDegree
    · simp only [barBasis, nativeBarEntry, constant, outputZero, ite_false,
        outputLow, ite_true, ringFOne, ringFMonomial]
      erw [ite_eq_right (by omega)]
    · simp only [barBasis, nativeBarEntry, constant, outputZero, ite_false,
        outputLow, ringFOne, ringFMonomial]
      erw [ite_eq_right (by omega)]

def ringKernel : CoefficientKernel F ringDegree where
  constant := constant
  weight := fun output row assignment =>
    ringFMul (barBasis row) (ringFMonomial assignment.val 1) output

theorem basisConstantTerm :
    forall row assignment : Fin ringDegree,
      ringKernel.weight ringKernel.constant row assignment =
        if row = assignment then 1 else 0 := by
  decide +kernel

theorem ringConstantTermLaw :
    ConstantTermLaw ConcreteCarrier.baseOps ringKernel := by
  constructor
  exact basisConstantTerm

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RingCoefficientKernel
