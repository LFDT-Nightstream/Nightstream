import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Lifecycle.PaperExtractionAlgebra
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkBinding
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.RelaxedBinding

/-!
The selected fork extractor and relaxed-binding game use the same scalar
and assignment operations. Two B-bounded response assignments have a strict
2B-bounded difference. No statement about arbitrary ambient openings is used.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.BindingBridge

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open _root_.NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

private theorem subtraction_eq
    (left right : PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (PaperExtractionAlgebra.extractionAlgebra
      (logicalWidth := logicalWidth) (publicFits := publicFits) ajtai).assignmentModule.sub left right =
      Binding.difference (shape := FullShape logicalWidth publicFits) left right := by
  funext column
  change left column + -right column = left column - right column
  exact (Fin.sub_eq_add_neg _ _).symm

/-- All four compatibility facts are derived from the selected deterministic
operations and the response norm bound. A collision conclusion is not a premise. -/
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
    change Phi81Relation.assignmentNormBounded (shape := FullShape logicalWidth publicFits)
      (2 * productionGlobalParams.bigB)
      ((PaperExtractionAlgebra.extractionAlgebra
        (logicalWidth := logicalWidth) (publicFits := publicFits) ajtai).assignmentModule.sub left right)
    rw [subtraction_eq (logicalWidth := logicalWidth) (publicFits := publicFits) ajtai left right]
    have leftBound : Phi81Relation.assignmentNormBounded (shape := FullShape logicalWidth publicFits)
        productionGlobalParams.bigB left := leftNorm
    have rightBound : Phi81Relation.assignmentNormBounded (shape := FullShape logicalWidth publicFits)
        productionGlobalParams.bigB right := rightNorm
    simpa only [two_mul] using Binding.difference_bounded (shape := FullShape logicalWidth publicFits)
      left right leftBound rightBound

end NightstreamFPrime.Lifecycle.Nifs.BindingBridge
