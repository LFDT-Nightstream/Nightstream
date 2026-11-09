import NightstreamFPrime.Spec.HyperNova.Construction2.Paper

/-! HyperNova Construction 2 with one folding verifier per step circuit. A step folds the prior
circuit's instance into the prior circuit's slot, as Construction 2 does, with the verifier of
the circuit that runs now. With one verifier for every circuit this is Construction 2 itself. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.HyperNova.Construction2.PerCircuit

open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold

universe uKey uDigest uState uWitness uRunning uFresh uProof uEncoded

structure Setup
    (Key : Type uKey)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Proof : Type uProof)
    (slotCount : Nat) where
  verifierKeys : Fin slotCount -> Key
  nifs : Fin slotCount -> Verifier Key Running Fresh Proof
  defaultRunning : Running

namespace Setup

variable {Key : Type uKey} {Running : Type uRunning} {Fresh : Type uFresh}
  {Proof : Type uProof} {slotCount : Nat}

/-- The Construction 2 setup that a step of circuit `circuit` runs. -/
def forCircuit (setup : Setup Key Running Fresh Proof slotCount) (circuit : Fin slotCount) :
    Paper.Setup Key Running Fresh Proof slotCount where
  verifierKeys := setup.verifierKeys
  nifs := setup.nifs circuit
  defaultRunning := setup.defaultRunning

def ofPaper (setup : Paper.Setup Key Running Fresh Proof slotCount) :
    Setup Key Running Fresh Proof slotCount where
  verifierKeys := setup.verifierKeys
  nifs := fun _ => setup.nifs
  defaultRunning := setup.defaultRunning

end Setup

def Transition
    {Key : Type uKey} {Digest : Type uDigest} {State : Type uState} {Witness : Type uWitness}
    {Running : Type uRunning} {Fresh : Type uFresh} {Proof : Type uProof}
    {Encoded : Type uEncoded} {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Paper.Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Paper.Input Key State Witness Running Fresh Proof slotCount)
    (output : Paper.Output Digest State Running slotCount) : Prop :=
  Paper.Transition (setup.forCircuit functionIndex) machine functionIndex input output

/-- With one verifier for every circuit, the per-circuit transition is Construction 2. -/
theorem transition_ofPaper_iff
    {Key : Type uKey} {Digest : Type uDigest} {State : Type uState} {Witness : Type uWitness}
    {Running : Type uRunning} {Fresh : Type uFresh} {Proof : Type uProof}
    {Encoded : Type uEncoded} {slotCount : Nat}
    (setup : Paper.Setup Key Running Fresh Proof slotCount)
    (machine : Paper.Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Paper.Input Key State Witness Running Fresh Proof slotCount)
    (output : Paper.Output Digest State Running slotCount) :
    Transition (Setup.ofPaper setup) machine functionIndex input output ↔
      Paper.Transition setup machine functionIndex input output :=
  Iff.rfl

/-- In a recursive step, the prior slot is updated by the running circuit's verifier and the
other slots are unchanged (HyperNova Construction 2, step 4). -/
theorem recursive_selected
    {Key : Type uKey} {Digest : Type uDigest} {State : Type uState} {Witness : Type uWitness}
    {Running : Type uRunning} {Fresh : Type uFresh} {Proof : Type uProof}
    {Encoded : Type uEncoded} {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Paper.Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Paper.Input Key State Witness Running Fresh Proof slotCount)
    (output : Paper.Output Digest State Running slotCount)
    (holds : Transition setup machine functionIndex input output)
    (recursive : 0 < input.iteration) :
    ∃ priorPcValid : Paper.InRange slotCount input.priorPc,
      Accepts (setup.nifs functionIndex)
          (setup.verifierKeys (Paper.selectedIndex priorPcValid))
          (input.running (Paper.selectedIndex priorPcValid)) input.fresh input.nifsProof
          (output.runningNext (Paper.selectedIndex priorPcValid)) ∧
        ∀ slot, slot ≠ Paper.selectedIndex priorPcValid →
          output.runningNext slot = input.running slot := by
  rcases holds with ⟨_, _, _, _, base | step⟩
  · omega
  · obtain ⟨priorPcValid, _, _, accepted, unchanged⟩ := step
    exact ⟨priorPcValid, accepted, unchanged⟩

end NightstreamFPrime.Spec.HyperNova.Construction2.PerCircuit
