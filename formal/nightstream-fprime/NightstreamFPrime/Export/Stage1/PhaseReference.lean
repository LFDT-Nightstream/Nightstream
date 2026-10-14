import NightstreamFPrime.Lifecycle.PaperAlgebra

/-! Reference shape for constructing phase circuits. Shape-invariance proofs
connect these recipes to the selected application's fixed point. This width
is not a package identity, commitment setup, or supported runtime layout. -/

namespace NightstreamFPrime.Export.Stage1.PhaseReference

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra

def logicalWidth : Nat := 27420587

theorem publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth := by
  apply Nat.le_trans (m := logicalWidth)
  · norm_num [logicalWidth, ringDegree, publicRingColumns]
  · exact Phi81CarrierLayout.logicalWidth_le_carrierWidth logicalWidth

end NightstreamFPrime.Export.Stage1.PhaseReference
