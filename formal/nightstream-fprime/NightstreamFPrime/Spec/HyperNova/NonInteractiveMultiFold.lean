
/-! Provenance: copied from `formal/nightstream-lean/Nightstream/HyperNova/NonInteractiveMultiFold.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/
/-!
Paper-owned non-interactive multi-fold verifier interface.

Source: HyperNova Section 3, Definitions 7 and 9, and Section 6.2,
Definition 12.

Owns: the single-message deterministic verifier surface `NIFS.V`, its exact
acceptance equation, and soundness/completeness with respect to an independently
stated public transition relation.

Does not own: a particular folding protocol, Fiat--Shamir encoding, a random
oracle instantiation, commitment security, SuperNeo, Rust, R1CS, or costs.

Emits constraints: no.

The verifier is a function, not an arbitrary acceptance proposition.  Hence a
fixed key, source pair, and prover message have at most one public output.
-/

namespace NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold

universe uKey uRunning uFresh uProof

/-- HyperNova's non-interactive verifier.  `Proof` is the sole prover message. -/
structure Verifier
    (Key : Type uKey)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Proof : Type uProof) where
  verify : Key -> Running -> Fresh -> Proof -> Option Running

/-- Exact computed acceptance of one non-interactive fold. -/
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

/-- Frozen graph equation for HyperNova's deterministic `NIFS.V`.  `proof` is
the verifier's sole prover-supplied message. -/
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

/-- A deterministic verifier cannot accept one message with two outputs. -/
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

/-- Independent public transition implemented by a NIFS verifier. -/
abbrev Transition
    (Key : Type uKey)
    (Running : Type uRunning)
    (Fresh : Type uFresh) :=
  Key -> Running -> Fresh -> Running -> Prop

/-- Every accepted prover message realizes the independent transition. -/
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

/-- Every valid public transition has a single prover message accepted by the
deterministic verifier. -/
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

/-- Extensional correctness of `NIFS.V` against an independently stated
transition.  This packages no theorem as input to the verifier. -/
def Exact
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (transition : Transition Key Running Fresh) : Prop :=
  Sound verifier transition /\ Complete verifier transition

/-- Exactness is the expected existential acceptance equivalence. -/
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
