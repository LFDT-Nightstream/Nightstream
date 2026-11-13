import NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.RelaxedBinding
import NightstreamFPrime.Spec.Folding.PiRLC.ChainBinding
import NightstreamFPrime.Spec.Folding.Schedule

/-! SuperNeo v1.2 Section 8 at the chain norm. A relaxed binding collision with openings below
`2β` gives a kernel vector of the Ajtai key below `8 · T · β`. A chain recovers every
accumulator at `β ≤ B + cap · T · (b − 1)` (`Schedule.recoveredBounds_le`), so the vector is
below `Schedule.msisChainBound`. `relaxedBindingCollision_to_shortKernel` is the case `β = B`. -/

namespace NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.Binding

open EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding.PiRLC.ChainBinding (CollisionAt)

def collisionAt_to_shortKernel {shape : Shape} {rows : Nat}
    (key : Commitment.Key shape rows) (commitment : Commitment.Value rows) (bound : Nat)
    (collision : CollisionAt (relationSemantics (Commitment.commit key)) productionGlobalParams
      relaxedOps (2 * bound) commitment) :
    ShortKernelVector key (8 * 216 * bound) := by
  let pair := collision.collision
  let left := CarrierAction.act pair.delta₁ pair.opening₂
  let right := CarrierAction.act pair.delta₂ pair.opening₁
  refine liftKernel key (difference left right)
    (difference_nonzero left right pair.crossDifferent) ?_ ?_
  · intro column
    have firstStrict : ∀ i, centeredMagnitude (pair.opening₁ i) < 2 * bound :=
      fun i => collision.firstNorm i
    have secondStrict : ∀ i, centeredMagnitude (pair.opening₂ i) < 2 * bound :=
      fun i => collision.secondNorm i
    have firstBound : ∀ i, centeredMagnitude (pair.opening₁ i) ≤ 2 * bound - 1 := fun i => by
      have := firstStrict i
      omega
    have secondBound : ∀ i, centeredMagnitude (pair.opening₂ i) ≤ 2 * bound - 1 := fun i => by
      have := secondStrict i
      omega
    have positive := firstStrict column
    have leftBound := difference_action_bounded pair.delta₁ pair.opening₂ pair.delta₁Valid
      secondBound column
    have rightBound := difference_action_bounded pair.delta₂ pair.opening₁ pair.delta₂Valid
      firstBound column
    have triangle := Norm.Centered.centeredMagnitude_sub_le (left column) (right column)
    change centeredMagnitude (left column) ≤ 2 * 216 * (2 * bound - 1) at leftBound
    change centeredMagnitude (right column) ≤ 2 * 216 * (2 * bound - 1) at rightBound
    change centeredMagnitude (left column - right column) < 8 * 216 * bound
    omega
  · apply equal_commitments_difference_kernel key left right
    dsimp only [left, right]
    have firstEquation : Commitment.commitmentAct pair.delta₁ commitment =
        Commitment.commit key pair.opening₁ := pair.firstEquation
    have secondEquation : Commitment.commitmentAct pair.delta₂ commitment =
        Commitment.commit key pair.opening₂ := pair.secondEquation
    rw [Commitment.commit_act, Commitment.commit_act, ← secondEquation, ← firstEquation]
    funext row
    change ringFMul pair.delta₁ (ringFMul pair.delta₂ (commitment row)) =
      ringFMul pair.delta₂ (ringFMul pair.delta₁ (commitment row))
    rw [← RingFLaws.ringFMul_assoc, ← RingFLaws.ringFMul_assoc,
      RingFLaws.ringFMul_comm pair.delta₁ pair.delta₂]

/-- A collision on an accumulator that a chain with this cap recovers gives a kernel vector
below `Schedule.msisChainBound`. -/
def chainCollision_to_shortKernel {shape : Shape} {rows : Nat}
    (key : Commitment.Key shape rows) (commitment : Commitment.Value rows) {cap bound : Nat}
    (recovered : bound ≤ productionGlobalParams.bigB +
      cap * (productionGlobalParams.expansionT * (productionGlobalParams.b - 1)))
    (collision : CollisionAt (relationSemantics (Commitment.commit key)) productionGlobalParams
      relaxedOps (2 * bound) commitment) :
    ShortKernelVector key (Folding.Schedule.msisChainBound productionGlobalParams cap) :=
  let kernel := collisionAt_to_shortKernel key commitment bound collision
  { kernel with
    bounded := fun column => Nat.lt_of_lt_of_le (kernel.bounded column)
      (Nat.mul_le_mul_left (8 * productionGlobalParams.expansionT) recovered) }

end NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.Binding
