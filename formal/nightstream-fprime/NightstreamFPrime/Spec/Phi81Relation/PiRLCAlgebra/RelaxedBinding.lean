import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Binding
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Product
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFLaws
import NightstreamFPrime.Spec.Profile

/-! The selected `(2B,C)` relaxed-binding collision yields a nonzero integer
kernel vector below `8TB`. Both difference challenges come from the actual
production set, and the matrix is unchanged. Hardness remains external. -/

namespace NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Binding

open EvaluationHomomorphism

def differenceChallenge (delta : RingF) : Prop :=
  ∃ left right : RingF, Challenge.challengeValid left ∧ Challenge.challengeValid right ∧
    delta = Phi81StrongSet.ringFSub left right

def relaxedOps {shape : Shape} {rows : Nat} :
    Folding.PiRLC.RelaxedBindingOps (Assignment shape) (Commitment.Value rows) RingF where
  scaleAssignment := CarrierAction.act
  scaleCommitment := Commitment.commitmentAct
  differenceChallenge := differenceChallenge

private theorem ringFMul_sub_left (left right value : RingF) :
    ringFMul (Phi81StrongSet.ringFSub left right) value =
      Phi81StrongSet.ringFSub (ringFMul left value) (ringFMul right value) := by
  have reconstruct : ringFAdd (Phi81StrongSet.ringFSub left right) right = left := by
    funext lane
    exact sub_add_cancel (left lane : ZMod goldilocksModulus) (right lane)
  have equation := CarrierAction.ringFMul_add_left
    (Phi81StrongSet.ringFSub left right) right value
  rw [reconstruct] at equation
  funext lane
  have atLane := congrFun equation lane
  change ringFMul (Phi81StrongSet.ringFSub left right) value lane =
    (ringFMul left value lane : ZMod goldilocksModulus) - ringFMul right value lane
  exact (eq_sub_iff_add_eq).mpr atLane.symm

/-- A difference of two production challenges expands a coordinate bound by
at most `2T=432`. The bound applies to every complete-carrier coordinate. -/
theorem difference_action_bounded {shape : Shape} {bound : Nat}
    (delta : RingF) (assignment : Assignment shape)
    (valid : differenceChallenge delta)
    (bounded : ∀ column, centeredMagnitude (assignment column) ≤ bound)
    (column : Fin shape.carrierWidth) :
    centeredMagnitude (CarrierAction.act delta assignment column) ≤ 2 * 216 * bound := by
  obtain ⟨left, right, leftValid, rightValid, rfl⟩ := valid
  let packed := Folding.PiCCS.PaperJoint.Phi81ColumnLayout.decode column
  let block := CarrierAction.assignmentBlock assignment packed.1
  change centeredMagnitude (ringFMul (Phi81StrongSet.ringFSub left right) block packed.2) ≤ _
  rw [ringFMul_sub_left]
  have blockBounded : ∀ lane, centeredMagnitude (block lane) ≤ bound :=
    fun lane => bounded (CarrierAction.carrierColumn packed.1 lane)
  have leftBound := Norm.Product.ringFMul_le_expansion_of_bounded
    left block leftValid blockBounded packed.2
  have rightBound := Norm.Product.ringFMul_le_expansion_of_bounded
    right block rightValid blockBounded packed.2
  exact (Norm.Centered.centeredMagnitude_sub_le _ _).trans (by omega)

/-- The literal relaxed collision event maps to the paper's MSIS norm for
the selected `b=2`, `k_rho=16` profile. No injectivity assumption is used. -/
def relaxedBindingCollision_to_shortKernel {shape : Shape} {rows : Nat}
    (key : Commitment.Key shape rows) (commitment : Commitment.Value rows)
    (collision : Folding.PiRLC.RelaxedBindingCollision
      (relationSemantics (Commitment.commit key)) productionGlobalParams relaxedOps commitment) :
    ShortKernelVector key productionGlobalParams.msisNormBound := by
  let left := CarrierAction.act collision.delta₁ collision.opening₂
  let right := CarrierAction.act collision.delta₂ collision.opening₁
  refine liftKernel key (difference left right)
    (difference_nonzero left right collision.crossDifferent) ?_ ?_
  · intro column
    let bound := 2 * productionGlobalParams.bigB - 1
    have firstBound : ∀ i, centeredMagnitude (collision.opening₁ i) ≤ bound := by
      intro i
      have strict := collision.firstNorm i
      change centeredMagnitude (collision.opening₁ i) < 2 * productionGlobalParams.bigB at strict
      dsimp [bound]
      omega
    have secondBound : ∀ i, centeredMagnitude (collision.opening₂ i) ≤ bound := by
      intro i
      have strict := collision.secondNorm i
      change centeredMagnitude (collision.opening₂ i) < 2 * productionGlobalParams.bigB at strict
      dsimp [bound]
      omega
    have leftBound := difference_action_bounded collision.delta₁ collision.opening₂
      collision.delta₁Valid secondBound column
    have rightBound := difference_action_bounded collision.delta₂ collision.opening₁
      collision.delta₂Valid firstBound column
    have triangle := Norm.Centered.centeredMagnitude_sub_le (left column) (right column)
    change centeredMagnitude (left column - right column) < _
    -- These values reduce from `2T(2B-1)` and `8TB` in the selected profile.
    change centeredMagnitude (left column) ≤ 56622672 at leftBound
    change centeredMagnitude (right column) ≤ 56622672 at rightBound
    change centeredMagnitude (left column - right column) < 113246208
    omega
  · apply equal_commitments_difference_kernel key left right
    dsimp only [left, right]
    have firstEquation : Commitment.commitmentAct collision.delta₁ commitment =
        Commitment.commit key collision.opening₁ := collision.firstEquation
    have secondEquation : Commitment.commitmentAct collision.delta₂ commitment =
        Commitment.commit key collision.opening₂ := collision.secondEquation
    rw [Commitment.commit_act, Commitment.commit_act,
      ← secondEquation, ← firstEquation]
    funext row
    change ringFMul collision.delta₁ (ringFMul collision.delta₂ (commitment row)) =
      ringFMul collision.delta₂ (ringFMul collision.delta₁ (commitment row))
    rw [← RingFLaws.ringFMul_assoc, ← RingFLaws.ringFMul_assoc,
      RingFLaws.ringFMul_comm collision.delta₁ collision.delta₂]

end NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Binding
