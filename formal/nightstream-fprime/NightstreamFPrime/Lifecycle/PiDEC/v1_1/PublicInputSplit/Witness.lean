import NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.SignedSplitScalar.Witness

/-! Read support of the indexed PiDEC sign-hint batches. -/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra

theorem witnesses_main_readsSatisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth publicFits) (offset : Nat)
    (allowed : Nat → Prop)
    (parentsSupported : ∀ coordinate,
      (interface.parent offset coordinate).VarsSatisfy allowed) :
    ∀ batch ∈ witnesses (Circuit.ops (main interface) offset),
      batch.ReadsSatisfy allowed := by
  intro batch member
  change batch ∈ ((List.range (coordinateCount logicalWidth publicFits)).map
    (childOp interface offset)).flatMap Op.witnesses at member
  rcases List.mem_flatMap.mp member with ⟨operation, operationMember, batchMember⟩
  rcases List.mem_map.mp operationMember with ⟨source, sourceMember, rfl⟩
  have sourceLt := List.mem_range.mp sourceMember
  rw [childOp, dif_pos sourceLt] at batchMember
  change batch ∈ witnesses (Circuit.ops
    (SignedSplitScalar.main (childInterface interface offset source sourceLt))
    (sourceOffset offset source)) at batchMember
  exact SignedSplitScalar.witnesses_main_readsSatisfy _ _ allowed
    (parentsSupported ⟨source, sourceLt⟩) batch batchMember

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit
