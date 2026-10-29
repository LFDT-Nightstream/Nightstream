/-! HyperNova Section 3, Definitions 7 and 9; Section 6.2, Definition 12. -/

namespace NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold

universe uKey uRunning uFresh uProof

structure Verifier
    (Key : Type uKey)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Proof : Type uProof) where
  verify : Key -> Running -> Fresh -> Proof -> Option Running

def Accepts
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (key : Key)
    (running : Running)
    (fresh : Fresh)
    (proof : Proof)
    (output : Running) : Prop :=
  verifier.verify key running fresh proof = some output

theorem accepts_iff_verify
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (key : Key)
    (running : Running)
    (fresh : Fresh)
    (proof : Proof)
    (output : Running) :
    Accepts verifier key running fresh proof output <->
      verifier.verify key running fresh proof = some output := by
  rfl

theorem accepted_output_unique
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (key : Key)
    (running : Running)
    (fresh : Fresh)
    (proof : Proof)
    {left right : Running}
    (leftAccepted : Accepts verifier key running fresh proof left)
    (rightAccepted : Accepts verifier key running fresh proof right) :
    left = right := by
  rw [Accepts] at leftAccepted rightAccepted
  exact Option.some.inj (leftAccepted.symm.trans rightAccepted)

abbrev Transition
    (Key : Type uKey)
    (Running : Type uRunning)
    (Fresh : Type uFresh) :=
  Key -> Running -> Fresh -> Running -> Prop

def Sound
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (transition : Transition Key Running Fresh) : Prop :=
  forall key running fresh proof output,
    Accepts verifier key running fresh proof output ->
      transition key running fresh output

def Complete
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (transition : Transition Key Running Fresh) : Prop :=
  forall key running fresh output,
    transition key running fresh output ->
      exists proof, Accepts verifier key running fresh proof output

def Exact
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (transition : Transition Key Running Fresh) : Prop :=
  Sound verifier transition /\ Complete verifier transition

theorem exact_iff_exists_accepted
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (transition : Transition Key Running Fresh)
    (exact : Exact verifier transition)
    (key : Key)
    (running : Running)
    (fresh : Fresh)
    (output : Running) :
    (exists proof, Accepts verifier key running fresh proof output) <->
      transition key running fresh output := by
  constructor
  · rintro ⟨proof, accepted⟩
    exact exact.1 key running fresh proof output accepted
  · intro holds
    exact exact.2 key running fresh output holds

end NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold
