import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
import NightstreamFPrime.Spec.AjtaiSetupV1.Prefix

/-! The two binding reductions instantiated at the verifier's exact public
seed, matrix dimensions and `8TB` MSIS norm. This is a deterministic link;
hardness of the resulting public-seed instance is an explicit premise. -/

namespace NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

open NightstreamFPrime.Spec
open Folding.PiCCS.PaperJoint
open Phi81Relation
open PiRLCAlgebra

/-- Ordinary collisions yield the stronger `2B` norm; this return type widens
that bound to the same `8TB` instance used by the selected security analysis. -/
def productionBindingCollision_to_shortKernel
    (commitment : Commitment.Value verifierRows)
    (collision : Opening.BindingCollision
      (relationSemantics (Commitment.commit productionAjtaiKey))
      productionGlobalParams.bigB commitment) :
    Binding.ShortKernelVector productionAjtaiKey productionGlobalParams.msisNormBound := by
  let witness := Binding.bindingCollision_to_shortKernel productionAjtaiKey commitment collision
  exact {
    vector := witness.vector
    nonzero := witness.nonzero
    bounded := fun column => (witness.bounded column).trans_le (by decide)
    kernel := witness.kernel }

/-- Shape of the larger fixed-seed instance named in the approved
`PUBLIC_SEED_MSIS_ASSUMPTION.md`, before unused allocation removal. -/
def approvedMsisShape : Phi81Relation.Shape :=
  Lifecycle.PaperAlgebra.fullShape 254260583 (by decide)

/-- Keep the approved seed and original number of message blocks. This key
is only the target of the reduction; the verifier uses `productionSetup`. -/
def approvedMsisSetup : AjtaiSetupV1.Setup verifierRows
    (Phi81ColumnLayout.blockCount approvedMsisShape.carrierWidth) where
  seed := productionSeed

/-- A selected short kernel solves the previously approved fixed instance
by appending zero blocks after the complete selected carrier. The strict
norm is unchanged. This does not assume hardness of a new matrix, restore
removed interior coordinates, or evaluate a setup entry. -/
def productionShortKernel_to_approvedMsis
    (witness : Binding.ShortKernelVector productionAjtaiKey
      productionGlobalParams.msisNormBound) :
    Binding.ShortKernelVector (shape := approvedMsisShape) approvedMsisSetup.verifierKey
      productionGlobalParams.msisNormBound := by
  apply AjtaiSetupV1.Prefix.extendShortKernel
    (smallShape := Lifecycle.PaperAlgebra.FullShape
      (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
      (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application))
    (largeShape := approvedMsisShape)
    (small := productionSetup) (large := approvedMsisSetup) _ rfl witness
  change Phi81CarrierLayout.carrierWidth
      (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application) ≤
    Phi81CarrierLayout.carrierWidth 254260583
  rw [carrierWidth_eq]
  decide

/-- The deterministic prefix reduction reaches the exact approved vector
length and keeps its strict `8TB` norm bound. -/
theorem approvedMsis_carrierWidth :
    approvedMsisShape.carrierWidth = 254260620 := by
  decide

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
