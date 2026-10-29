import NightstreamFPrime.Export.Stage1.PiRLCWitnessAction
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.StoredSplit
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericCompletionSum

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECParentScalarRead

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment)
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

private def signedTerm (coefficient value : F) : F :=
  if coefficient = 0 then 0 else if coefficient = 1 then value else -value

private theorem signedTerm_eq_mul (coefficient value : F)
    (signed : coefficient = 0 ∨ coefficient = 1 ∨ coefficient = -1) :
    signedTerm coefficient value = value * coefficient := by
  rcases signed with zero | positive | negative
  · subst coefficient
    simp only [signedTerm]
    exact (baseLaws.mul_zero value).symm
  · subst coefficient
    simp only [signedTerm, ite_eq_right (show (1 : F) ≠ 0 by decide)]
    exact (baseLaws.mul_one value).symm
  · subst coefficient
    simp only [signedTerm, ite_eq_right (show (-1 : F) ≠ 0 by decide),
      ite_eq_right (show (-1 : F) ≠ 1 by decide)]
    calc
      -value = -(1 * value) :=
        congrArg (fun entry : F => -entry) (baseLaws.one_mul value).symm
      _ = (-1) * value := (baseLaws.neg_mul 1 value).symm
      _ = value * (-1) := baseLaws.mul_comm _ _

private theorem numericSum_eq_sumRange (count : Nat) (term : Nat → F) :
    NumericCompletionSum.numericSum baseOps count term =
      sumRange baseOps count term := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [NumericCompletionSum.numericSum, Nat.fold_succ]
      change baseOps.add (NumericCompletionSum.numericSum baseOps count term)
          (term count) =
        baseOps.add (sumRange baseOps count term) (term count)
      rw [inductionHypothesis]

def coefficient (table : FixedArray MaterializedRingF ringDegree)
    (source : RingF) (output : Fin ringDegree) : F :=
  NumericCompletionSum.numericSum baseOps ringDegree fun index =>
    if live : index < ringDegree then
      signedTerm (source ⟨index, live⟩)
        ((table.get ⟨index, live⟩).toRing output)
    else 0

theorem coefficient_value (challenge : MaterializedRingF) (source : RingF)
    (signed : ∀ input, source input = 0 ∨ source input = 1 ∨ source input = -1)
    (output : Fin ringDegree) :
    coefficient (PiRLCWitnessAction.prepare challenge) source output =
      ringFMul challenge.toRing source output := by
  rw [coefficient, numericSum_eq_sumRange,
    CarrierAction.ringFMul_apply_eq_rightLinear]
  apply sumRange_congr
  intro index live
  simp only [dite_eq_left live, PiRLCWitnessAction.prepare, FixedArray.get_ofFn,
    MaterializedRingF.toRing_ofRing, CarrierAction.rightCoefficient]
  exact signedTerm_eq_mul _ _ (signed ⟨index, live⟩)

private theorem splitScalar_signed (value : F) (child : Radix.ChildIndex)
    (bounded : centeredMagnitude value < Radix.combinedBound) :
    Radix.splitScalar value child = 0 ∨
      Radix.splitScalar value child = 1 ∨ Radix.splitScalar value child = -1 := by
  simp only [Radix.splitScalar, ite_eq_left bounded]
  have digitBound := Radix.magnitudeDigit_lt_two value child
  have casesDigit : Radix.magnitudeDigit value child = 0 ∨
      Radix.magnitudeDigit value child = 1 := by omega
  rcases casesDigit with zero | one
  · left
    simp [Radix.boundedDigit, zero]
  · by_cases nonnegative : Radix.isNonnegative value
    · right; left
      simp [Radix.boundedDigit, nonnegative, one]
    · right; right
      simp [Radix.boundedDigit, nonnegative, one]

def prepare (_unit : Unit := ()) :
    FixedArray (FixedArray MaterializedRingF ringDegree) ringDegree :=
  FixedArray.ofFn fun basis =>
    PiRLCWitnessAction.prepare
      (MaterializedRingF.ofRing (Phi81CoefficientKernel.barBasis basis))

def read (tables : FixedArray (FixedArray MaterializedRingF ringDegree) ringDegree)
    (parent : StoredAssignment ringDegree) (basis : Fin ringDegree)
    (child : Radix.ChildIndex) (output : Fin ringDegree) : F :=
  coefficient (tables.get basis)
    (fun input => Radix.splitScalar (parent.get input) child) output

theorem read_eq_splitScalar (parent : StoredAssignment ringDegree)
    (bounded : ∀ input, centeredMagnitude (parent.get input) < Radix.combinedBound)
    (basis : Fin ringDegree) (child : Radix.ChildIndex) (output : Fin ringDegree) :
    read (prepare ()) parent basis child output =
      CarrierAction.kernelImage basis
        (fun input => Radix.splitScalar (parent.get input) child) output := by
  have signed : ∀ input : Fin ringDegree,
      Radix.splitScalar (parent.get input) child = 0 ∨
        Radix.splitScalar (parent.get input) child = 1 ∨
        Radix.splitScalar (parent.get input) child = -1 :=
    fun input => splitScalar_signed (parent.get input) child (bounded input)
  rw [read, prepare, FixedArray.get_ofFn,
    coefficient_value
      (MaterializedRingF.ofRing (Phi81CoefficientKernel.barBasis basis))
      (fun input => Radix.splitScalar (parent.get input) child) signed output,
    MaterializedRingF.toRing_ofRing]
  exact (congrFun (CarrierAction.kernelImage_eq_ringFMul basis _) output).symm

theorem read_eq_checkedChild (parent : StoredAssignment ringDegree)
    (children : Vector (StoredAssignment ringDegree) productionGlobalParams.k)
    (success : StoredSplit.splitChecked parent = some children)
    (basis : Fin ringDegree) (child : Radix.ChildIndex) (output : Fin ringDegree) :
    read (prepare ()) parent basis child output =
      CarrierAction.kernelImage basis (children.get child).get output := by
  have bounded := ((StoredSplit.splitChecked_eq_some_iff parent children).mp success).1
  rw [read_eq_splitScalar parent bounded basis child output]
  apply congrArg (fun source : RingF => CarrierAction.kernelImage basis source output)
  funext input
  exact (StoredSplit.splitChecked_value parent children success child input).symm

end NightstreamFPrime.Export.Stage1.PiDECParentScalarRead
