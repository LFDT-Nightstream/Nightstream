import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.RelaxedBinding
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

/-! The two binding reductions instantiated at the verifier's exact public
seed, matrix dimensions and `8TB` MSIS norm. This is a deterministic link;
hardness of the resulting public-seed instance is an explicit premise. -/

namespace NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup

open NightstreamFPrime.Spec
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

/-- The relaxed collision uses actual `C-C` challenges and strict `2B`
openings, and returns a short integer kernel vector under the selected key. -/
def productionRelaxedBindingCollision_to_shortKernel
    (commitment : Commitment.Value verifierRows)
    (collision : Folding.PiRLC.RelaxedBindingCollision
      (relationSemantics (Commitment.commit productionAjtaiKey))
      productionGlobalParams Binding.relaxedOps commitment) :
    Binding.ShortKernelVector productionAjtaiKey productionGlobalParams.msisNormBound :=
  Binding.relaxedBindingCollision_to_shortKernel productionAjtaiKey commitment collision

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Setup
