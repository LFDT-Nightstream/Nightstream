import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFLaws
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Challenge

/-!
Owns the exact production scalar-ring and strong-set unit bridge used by the
PiRLC coordinate-fork extractor.

All quotient-ring laws are deterministic. The only hypothesis is the
explicit `LowNormInvertibility` statement isolated by the Φ₈₁ strong-set
development.
-/

namespace NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiRLC
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkAlgebra
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism

def ring : CommutativeRingOps RingF where
  zero := ringFZero
  one := ringFOne
  add := ringFAdd
  mul := ringFMul
  neg := fun value lane => -value lane

theorem ring_sub_eq (left right : RingF) :
    ring.sub left right = Phi81StrongSet.ringFSub left right := by
  funext lane
  exact (Fin.sub_eq_add_neg _ _).symm

theorem ringLaws : CommutativeRingLaws ring where
  add_assoc := by
    intro left middle right
    funext lane
    exact baseLaws.add_assoc _ _ _
  add_comm := by
    intro left right
    funext lane
    exact baseLaws.add_comm _ _
  zero_add := by
    intro value
    funext lane
    exact baseLaws.zero_add _
  add_zero := by
    intro value
    funext lane
    exact baseLaws.add_zero _
  add_neg := by
    intro value
    funext lane
    exact baseLaws.add_neg _
  mul_assoc := EvaluationHomomorphism.RingFLaws.ringFMul_assoc
  mul_comm := EvaluationHomomorphism.RingFLaws.ringFMul_comm
  one_mul := EvaluationHomomorphism.RingFLaws.ringFMul_one_left
  mul_one := EvaluationHomomorphism.RingFLaws.ringFMul_one_right
  left_distrib := EvaluationHomomorphism.CarrierAction.ringFMul_add_right
  right_distrib := EvaluationHomomorphism.CarrierAction.ringFMul_add_left

/-- The production challenge predicate supplies a two-sided inverse for every
nonzero fork difference. -/
noncomputable def strongSetUnits
    (theorem8 : Phi81StrongSet.LowNormInvertibility) :
    StrongSetUnits ring Challenge.challengeValid where
  differenceUnit := by
    intro left right leftValid rightValid different
    have invertible := Challenge.pairwiseSecure_of_lowNormInvertibility
      theorem8 leftValid rightValid different
    let inverse := Classical.choose invertible
    have inverseLaws := Classical.choose_spec invertible
    refine {
      inverse := inverse
      inverse_mul := ?_
      mul_inverse := ?_
    }
    · rw [ring_sub_eq]
      exact inverseLaws.2
    · rw [ring_sub_eq]
      exact inverseLaws.1

end NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet
