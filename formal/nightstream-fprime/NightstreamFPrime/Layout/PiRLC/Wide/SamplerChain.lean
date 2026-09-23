import NightstreamFPrime.Layout.PiRlcWideSampler.BatchPhysical

/-! Physical footprint adapter for the candidate wide sampler phase. -/
namespace NightstreamFPrime.Layout.PiRLC.Wide.SamplerChain
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.PiRLC.Wide

structure InputsAffine (interface : ProjectedBatch.Interface) (offset : Nat) : Prop where
  initialState : Poseidon2.StateAffine (interface.initialState offset)

def logicalConstraints (interface : ProjectedBatch.Interface) (offset : Nat) : List Expr :=
  flatConstraints (ProjectedBatch.operations interface offset)

theorem totalFreshCount_eq (interface : ProjectedBatch.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalFreshCount (logicalConstraints interface offset) = 26316 :=
  (PiRlcWideSampler.BatchPhysical.counts interface offset inputs.initialState).1

theorem totalRowCount_eq (interface : ProjectedBatch.Interface) (offset : Nat)
    (inputs : InputsAffine interface offset) :
    R1CS.totalRowCount (logicalConstraints interface offset) = 58939 :=
  (PiRlcWideSampler.BatchPhysical.counts interface offset inputs.initialState).2

end NightstreamFPrime.Layout.PiRLC.Wide.SamplerChain
