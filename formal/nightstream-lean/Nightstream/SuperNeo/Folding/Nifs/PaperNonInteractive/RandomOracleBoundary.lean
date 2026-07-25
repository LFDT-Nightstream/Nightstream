import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Types
import Nightstream.SuperNeo.InteractiveReduction.FiatShamirContract

/-!
Exact typed boundary between the paper NIFS and a random-oracle
instantiation.

Source: SuperNeo Sections 7.3--7.4 and HyperNova Section 3.

Owns: the complete replay input after the running/fresh public pair has been
absorbed; exact equations for the `Pi_CCS` replay, post-output handoff, and
coordinate-aligned `Pi_RLC` response; and a closed family of transcript
collision or sampling failures.

Does not own: a probability bound, multi-forking/programming, a concrete
Poseidon2 encoding, numeric domain tags, Ajtai binding, Rust, R1CS, artifacts,
minimality, or costs.

Emits constraints: no.

The typed calls fix semantic domain separation and order:

1. absorb the complete running/fresh public pair into the key-owned state;
2. initialize the typed `Pi_CCS` statement;
3. squeeze alpha coordinates, then gamma;
4. absorb each SumCheck message before its indexed challenge;
5. absorb the complete `Pi_CCS` output;
6. request every `Pi_RLC` coordinate from that post-output state.

The abstract functions may still ignore or collide on their arguments.  This
module names those failures; it does not turn the function interface into a
random-oracle security theorem.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uExtension uCommitment uPublicInput uScalar uState

/-- Exact typed replay input consumed by the `Pi_CCS` transcript.  The prior
state is already bound to the complete public NIFS input. -/
def piCcsReplayInput
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    ProtocolVerifier.ReplayInput Extension State shape where
  statement := {
    priorState := key.publicInputState running fresh
    input := (key.statement running fresh).verifierInput key.lift
  }
  rounds := (key.piCcsCertificate running fresh proof).toTranscript

/-- The executable NIFS uses exactly `piCcsReplayInput` for every `Pi_CCS`
challenge. -/
theorem piCcsExecution_coins_eq_replayInput
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    (key.piCcsExecution running fresh proof).coins =
      (piCcsReplayInput key running fresh proof).derive key.oracle := by
  rfl

/-- The state handed to `Pi_RLC` is exactly the state after replaying all
`Pi_CCS` rounds and absorbing the complete output message. -/
theorem piCcsExecution_outgoingState_eq_postOutput
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    (key.piCcsExecution running fresh proof).outgoingState =
      key.oracle.absorbOutput
        ((piCcsReplayInput key running fresh proof).derive key.oracle).finalState
        (key.piCcsCertificate running fresh proof).output := by
  rfl

/-- Every `Pi_RLC` coordinate uses its literal finite index and the common
post-`Pi_CCS` output state. -/
theorem piRlcChallenge_eq_response_after_piCcsOutput
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (coordinate : Fin key.arity.total) :
    key.piRlcChallenges running fresh proof coordinate =
      key.piRlcResponse
        (key.oracle.absorbOutput
          ((piCcsReplayInput key running fresh proof).derive key.oracle).finalState
          (key.piCcsCertificate running fresh proof).output)
        coordinate := by
  rfl

/-- Failure of the ideal paper sampler to return an element of the configured
strong sampling set.  The paper `Key` rules this out; a bounded production
sampler must instead reject or expose its separate shortfall event. -/
def PiRlcSamplingSetFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) : Prop :=
  exists coordinate,
    ¬ key.piRlcAlgebra.challengeValid
      (key.piRlcChallenges running fresh proof coordinate)

/-- The ideal paper key's strong-set contract excludes sampling-set failure. -/
theorem not_piRlcSamplingSetFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    ¬ PiRlcSamplingSetFailure key running fresh proof := by
  rintro ⟨coordinate, invalid⟩
  exact invalid
    (key.piRlcResponseValid
      (key.piCcsExecution running fresh proof).outgoingState coordinate)

/-- The full public NIFS inputs differ but the pre-`Pi_CCS` absorption
operation returns the same state. -/
def PublicInputBindingCollision
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (leftRunning : Running Extension Commitment PublicInput shape)
    (leftFresh : Fresh Commitment PublicInput shape)
    (rightRunning : Running Extension Commitment PublicInput shape)
    (rightFresh : Fresh Commitment PublicInput shape) : Prop :=
  (leftRunning, leftFresh) ≠ (rightRunning, rightFresh) /\
    key.publicInputState leftRunning leftFresh =
      key.publicInputState rightRunning rightFresh

/-- Closed transcript/security-refinement event family.  Each constructor is
an exact collision or sampler failure, never a generic refinement escape. -/
inductive TranscriptSecurityEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) where
  | publicInputBinding
      (otherRunning : Running Extension Commitment PublicInput shape)
      (otherFresh : Fresh Commitment PublicInput shape)
      (collision : PublicInputBindingCollision key running fresh
        otherRunning otherFresh)
  | replayChallenge
      (other : ProtocolVerifier.ReplayInput Extension State shape)
      (collision : ProtocolVerifier.TranscriptReplayCollision key.oracle
        (piCcsReplayInput key running fresh proof) other)
  | replayState
      (other : ProtocolVerifier.ReplayInput Extension State shape)
      (collision : ProtocolVerifier.TranscriptStateCollision key.oracle
        (piCcsReplayInput key running fresh proof) other)
  | outputAbsorption
      (otherState : State)
      (otherMessage : ProtocolPolynomial.OutputMessage Extension shape)
      (collision : ProtocolVerifier.OutputAbsorptionCollision key.oracle
        ((piCcsReplayInput key running fresh proof).derive key.oracle).finalState
        otherState
        (key.piCcsCertificate running fresh proof).output
        otherMessage)
  | piRlcSamplingSet
      (failure : PiRlcSamplingSetFailure key running fresh proof)

namespace TranscriptSecurityEvent

/-- Exact closed security class associated with each typed transcript event.
The coordinate-fork/multifork event belongs to the separate NIFS extraction
theorem and is not synthesized here. -/
def securityClass
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {running : Running Extension Commitment PublicInput shape}
    {fresh : Fresh Commitment PublicInput shape}
    {proof : Proof Extension Commitment shape degreeBound} :
    TranscriptSecurityEvent key running fresh proof ->
      Nightstream.SuperNeo.InteractiveReduction.Paper.FiatShamirSecurityEvent
  | .publicInputBinding _ _ _ => .publicInputBindingCollision
  | .replayChallenge _ _ => .transcriptReplayCollision
  | .replayState _ _ => .transcriptStateCollision
  | .outputAbsorption _ _ _ => .outputAbsorptionCollision
  | .piRlcSamplingSet _ => .challengeSamplingFailure

end TranscriptSecurityEvent

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
