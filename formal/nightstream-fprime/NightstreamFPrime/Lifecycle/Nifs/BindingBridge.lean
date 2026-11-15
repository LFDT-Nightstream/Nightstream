import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Lifecycle.PaperExtractionAlgebra
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkBinding
import NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.RelaxedBinding

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.BindingBridge

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open _root_.NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

theorem subtraction_eq
    (left right : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (PaperExtractionAlgebra.extractionAlgebra
      (logicalWidth := logicalWidth) (publicFits := publicFits) ajtai).assignmentModule.sub left right =
      Binding.difference (shape := FullShape logicalWidth publicFits) left right := by
  funext column
  change left column + -right column = left column - right column
  exact (Fin.sub_eq_add_neg _ _).symm

/-- Two openings below `bound` differ by less than `2 · bound`. Relaxed binding takes this at `B`
(`compatible`) and at the chain bound (`PiRLC.ChainBinding.forkCollision`,
`Light.Rlc.collisionAbove`). -/
theorem differenceBounded (bound : Nat)
    (left right : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
    (leftNorm : (ProductionKey.key relation ajtai).piRlcSemantics.normBounded bound left)
    (rightNorm : (ProductionKey.key relation ajtai).piRlcSemantics.normBounded bound right) :
    (ProductionKey.key relation ajtai).piRlcSemantics.normBounded (2 * bound)
      ((PaperExtractionAlgebra.extractionAlgebra
        (logicalWidth := logicalWidth) (publicFits := publicFits) ajtai).assignmentModule.sub left right) := by
  change RingRelation.assignmentNormBounded (shape := FullShape logicalWidth publicFits) (2 * bound)
    ((PaperExtractionAlgebra.extractionAlgebra
      (logicalWidth := logicalWidth) (publicFits := publicFits) ajtai).assignmentModule.sub left right)
  rw [subtraction_eq (logicalWidth := logicalWidth) (publicFits := publicFits) ajtai left right]
  have leftBound : RingRelation.assignmentNormBounded (shape := FullShape logicalWidth publicFits)
      bound left := leftNorm
  have rightBound : RingRelation.assignmentNormBounded (shape := FullShape logicalWidth publicFits)
      bound right := rightNorm
  simpa only [two_mul] using Binding.difference_bounded (shape := FullShape logicalWidth publicFits)
    left right leftBound rightBound

theorem compatible :
    PiRLC.PaperForkBinding.Compatible
      (algebra := (ProductionKey.key relation ajtai).piRlcAlgebra)
      (PaperExtractionAlgebra.extractionAlgebra
        (logicalWidth := logicalWidth) (publicFits := publicFits) ajtai)
      (Binding.relaxedOps (shape := FullShape logicalWidth publicFits)
        (rows := productionProfile.commitmentWidth)) := by
  refine {
    differenceValid := ?_
    assignmentAction := rfl
    commitmentAction := rfl
    differenceBounded := ?_
  }
  · intro left right leftValid rightValid
    refine ⟨left, right, leftValid, rightValid, ?_⟩
    exact ForkStrongSet.ring_sub_eq left right
  · intro left right leftNorm rightNorm
    exact differenceBounded relation ajtai _ left right leftNorm rightNorm

end NightstreamFPrime.Lifecycle.Nifs.BindingBridge
