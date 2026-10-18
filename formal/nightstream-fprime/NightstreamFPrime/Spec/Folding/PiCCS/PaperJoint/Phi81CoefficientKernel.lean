import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Concrete Phi81 coefficient kernel for the paper matrix embedding.

Protocol: SuperNeo coefficient embedding (v1.2 Section 5, Theorems 8 and 9).
Phase: concrete cyclotomic instantiation of the single-matrix source.
Constraint family: transformed matrix / assignment ring-product coefficients.

Owns: the closed-form Phi81 bar transform on coefficient bases; the exact
coefficient kernel obtained by multiplying that transformed basis by an
assignment basis with the independently defined Phi81 ring multiplication;
and the constant-term Kronecker law required by Theorem 8.

Does not own: proof that Rust's runtime Gram-matrix inversion returns this
closed form, protocol-level `ConcreteCarrier` refinement or verifier
composition, production matrix construction or padding, R1CS lowering, row
removal, or constraint counts.

Emits constraints: no.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel

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
    · simp only [barBasis, nativeBarEntry, constant, outputZero, if_false,
        outputLow, if_true, ringFOne, ringFMonomial]
      rw [if_neg (by omega)]
    · simp only [barBasis, nativeBarEntry, constant, outputZero, if_false,
        outputLow, ringFOne, ringFMonomial]
      rw [if_neg (by omega)]

/-- Exact coefficient kernel of `bar(e_row) * e_assignment` modulo Phi81. -/
def phi81Kernel : CoefficientKernel F ringDegree where
  constant := constant
  weight := fun output row assignment =>
    ringFMul (barBasis row) (ringFMonomial assignment.val 1) output

set_option maxRecDepth 100000 in -- fixed-size: ring degree 54, not artifact data
set_option maxHeartbeats 2000000 in -- fixed-size: ring degree 54, not artifact data
theorem basisConstantTerm :
    forall row assignment : Fin ringDegree,
      phi81Kernel.weight phi81Kernel.constant row assignment =
        if row = assignment then 1 else 0 := by
  decide

theorem phi81ConstantTermLaw :
    ConstantTermLaw ConcreteCarrier.baseOps phi81Kernel := by
  constructor
  exact basisConstantTerm

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
