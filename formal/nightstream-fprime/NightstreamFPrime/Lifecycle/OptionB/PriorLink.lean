import NightstreamFPrime.Lifecycle.OptionB.Step
import NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthSecurity

/-! Option B's prior link: the Construction 2 link of a fold to its prior hash preimage
(`BoundedDepthSecurity.StepLink` for `machine`), for a well-formed prior state. The light and
refresh transcripts absorb the fresh instance and not the running claim. The fresh public input
is `encHash` of the prior state hash, and the state hash binds the slot that the prior program
counter selects. So two linked folds with one fresh instance start from one running claim, or
their two prior states give a state-hash collision (`priorLink_binds`). This is the PR #155
`PriorLink` for option B: the prior state is an adversary output, so the collision names a pair
of states. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.OptionB

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.BoundedDepthSecurity (StepLink)
open NightstreamFPrime.Lifecycle.PaperAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

abbrev StepFold (logicalWidth : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :=
  BoundedDepth.Fold (CircuitKey logicalWidth publicFits) (SlotClaim logicalWidth publicFits)
    (StepFresh logicalWidth publicFits) StepProof

abbrev PriorPreimage (logicalWidth : Nat)
    (publicFits :
      ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :=
  Paper.HashPreimage (CircuitKey logicalWidth publicFits) AppState (SlotData logicalWidth publicFits) 2

/-- The fold's fresh instance commits to `prior`, whose selected slot holds `start`, and `prior`
is well formed. -/
def PriorLink (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (prior : PriorPreimage logicalWidth publicFits) (start : SlotData logicalWidth publicFits)
    (fold : StepFold logicalWidth publicFits) : Prop :=
  StepLink (machine step control) prior start fold ∧ (toState prior).WellFormed

/-- Two linked folds with one fresh instance start from one running claim, or their prior
states collide. -/
theorem priorLink_binds (step : AppState → AppWitness → AppState)
    (control : AppState → AppWitness → Fin 2) {prior prior' : PriorPreimage logicalWidth publicFits}
    {start start' : SlotData logicalWidth publicFits} {fold fold' : StepFold logicalWidth publicFits}
    (link : PriorLink step control prior start fold)
    (link' : PriorLink step control prior' start' fold') (sameFresh : fold.fresh = fold'.fresh) :
    start = start' ∨ StateCollision (toState prior) (toState prior') := by
  obtain ⟨⟨digest, valid, startEq⟩, well⟩ := link
  obtain ⟨⟨digest', valid', startEq'⟩, well'⟩ := link'
  change fold.fresh.publicInputs ⟨0, by decide⟩ = encHash (stateHash (toState prior)) at digest
  change fold'.fresh.publicInputs ⟨0, by decide⟩ = encHash (stateHash (toState prior')) at digest'
  rw [sameFresh] at digest
  have hashes : stateHash (toState prior) = stateHash (toState prior') :=
    encHash_injective_fixed (Poseidon2.hash_length _) (Poseidon2.hash_length _)
      (digest.symm.trans digest')
  by_cases same : toState prior = toState prior'
  · have pc : prior.pc = prior'.pc := congrArg State.pc same
    have running : prior.running = prior'.running := congrArg State.slots same
    have index : Paper.selectedIndex valid = Paper.selectedIndex valid' :=
      Fin.ext (by simp only [Paper.selectedIndex, pc])
    left
    rw [← startEq, ← startEq', index, running]
  · exact Or.inr (stateHash_collision well well' same hashes)

end NightstreamFPrime.Lifecycle.OptionB
