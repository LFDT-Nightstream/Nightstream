import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.RelaxedBinding
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

def productionRelaxedBindingCollision_to_shortKernel
    (commitment : Commitment.Value verifierRows)
    (collision : Folding.PiRLC.RelaxedBindingCollision
      (relationSemantics (Commitment.commit productionAjtaiKey))
      productionGlobalParams Binding.relaxedOps commitment) :
    Binding.ShortKernelVector productionAjtaiKey productionGlobalParams.msisNormBound :=
  Binding.relaxedBindingCollision_to_shortKernel productionAjtaiKey commitment collision

/-- Shape of the larger fixed-seed instance named in the approved
`PUBLIC_SEED_MSIS_ASSUMPTION.md`. -/
def approvedMsisShape : Phi81Relation.Shape :=
  Lifecycle.PaperAlgebra.fullShape 254260583 (by decide)

def approvedMsisSetup : AjtaiSetupV1.Setup verifierRows
    (Phi81ColumnLayout.blockCount approvedMsisShape.carrierWidth) where
  seed := productionSeed

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

theorem approvedMsis_carrierWidth :
    approvedMsisShape.carrierWidth = 254260620 := by
  decide

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
