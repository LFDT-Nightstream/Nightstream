import Nightstream.Protocol.FPrime.Paper

/-!
Cross-step binding for HyperNova Construction 2's exact typed hash preimage.

Assurance tier: model-level security partition.

Owns: the lifecycle premise that the next fresh claim carries the previous
output digest; the two exact abstract binding failures; and the theorem that
the current prior preimage—including the complete running child product—is
the previous next-output preimage or one of those failures occurred.

Does not own: serialization, Poseidon2 instantiation or collision bounds,
injectivity of a concrete instance encoding, NIFS semantics, Rust/R1CS
refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: digests are compression, never authority. Equality of
encoded digests yields preimage equality only after separately excluding an
instance-encoding collision and a hash collision. Concrete refinement must
show that the exact serialized Poseidon2 message contains every typed field
modeled by `HashPreimage`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.chain.fresh_output` | next fresh public input carries the previous output digest | lifecycle premise | `FreshCarriesPreviousOutput` |
| `fprime.chain.instance_encoding` | unequal digests must not encode to one public instance | security boundary | `InstanceEncodingCollision` |
| `fprime.chain.hash` | unequal typed preimages must not hash to one digest | security boundary | `HashCollision` |
| `fprime.chain.preimage` | current prior preimage equals previous next preimage or names a failure | exhaustive theorem | `preimage_eq_or_securityFailure` |
| `fprime.chain.running` | complete current running product equals the previous output product or names a failure | derived | `running_eq_or_securityFailure` |
-/

namespace Nightstream.Protocol.FPrime.Paper.PriorLink

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.Protocol.FPrime.Paper

universe uVerifierKey uDigest uState uWitness uStructure uPublicInput uPoint
  uEvaluation uCommitment

section

variable {VerifierKey : Type uVerifierKey}
variable {Digest : Type uDigest}
variable {State : Type uState}
variable {Witness : Type uWitness}
variable {Structure : Type uStructure}
variable {PublicInput : Type uPublicInput}
variable {Point : Type uPoint}
variable {Evaluation : Type uEvaluation}
variable {Commitment : Type uCommitment}
variable {params : GlobalParams}
variable {slotCount : Nat}

/-- A concrete public-instance encoding fails to distinguish two digests. -/
def InstanceEncodingCollision
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount) : Prop :=
  ∃ left right : Digest,
    left ≠ right ∧ machine.encodeInstance left = machine.encodeInstance right

/-- The abstract Construction-2 hash fails to distinguish two exact typed
preimages. A concrete theorem must instantiate this with the serialized
Poseidon2 message, not a caller-supplied digest. -/
def HashCollision
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount) : Prop :=
  ∃ left right : HashPreimage VerifierKey State Structure PublicInput Point
      Evaluation Commitment params slotCount,
    left ≠ right ∧ machine.hash left = machine.hash right

/-- Exhaustive binding failures at the paper cross-step compression boundary. -/
inductive SecurityFailure
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount) : Prop where
  | instanceEncoding
      (collision : InstanceEncodingCollision machine)
  | hash
      (collision : HashCollision machine)

/-- Lifecycle linkage absent from a single F-prime invocation: the next fresh
claim exposes the exact prior output digest through `encodeInstance`. -/
def FreshCarriesPreviousOutput
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (previousOutput : Output Digest State Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (currentInput : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount) : Prop :=
  currentInput.fresh.publicInput =
    machine.encodeInstance previousOutput.x

/-- The three paper equations force equality of the exact typed cross-step
preimages, unless the instance encoding or hash fails to bind. -/
theorem preimage_eq_or_securityFailure
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (previousInput : Input VerifierKey State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (previousOutput : Output Digest State Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (currentInput : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (previousOutputHash : OutputHolds machine previousInput previousOutput)
    (currentPriorPublic :
      currentInput.fresh.publicInput =
        machine.encodeInstance (machine.hash (priorHashPreimage currentInput)))
    (freshCarries :
      FreshCarriesPreviousOutput machine previousOutput currentInput) :
    priorHashPreimage currentInput =
        nextHashPreimage previousInput previousOutput ∨
      SecurityFailure machine := by
  classical
  let currentPreimage := priorHashPreimage currentInput
  let previousPreimage := nextHashPreimage previousInput previousOutput
  have encodedEq :
      machine.encodeInstance (machine.hash currentPreimage) =
        machine.encodeInstance (machine.hash previousPreimage) := by
    calc
      machine.encodeInstance (machine.hash currentPreimage) =
          currentInput.fresh.publicInput := currentPriorPublic.symm
      _ = machine.encodeInstance previousOutput.x := freshCarries
      _ = machine.encodeInstance (machine.hash previousPreimage) := by
        exact congrArg machine.encodeInstance previousOutputHash
  by_cases digestEq :
      machine.hash currentPreimage = machine.hash previousPreimage
  · by_cases preimageEq : currentPreimage = previousPreimage
    · exact Or.inl preimageEq
    · exact Or.inr (.hash ⟨currentPreimage, previousPreimage,
        preimageEq, digestEq⟩)
  · exact Or.inr (.instanceEncoding
      ⟨machine.hash currentPreimage, machine.hash previousPreimage,
        digestEq, encodedEq⟩)

/-- In particular, the complete child product—not a parent-only cache—is
carried across the paper boundary, modulo the two named failures. -/
theorem running_eq_or_securityFailure
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (previousInput : Input VerifierKey State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (previousOutput : Output Digest State Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (currentInput : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (previousOutputHash : OutputHolds machine previousInput previousOutput)
    (currentPriorPublic :
      currentInput.fresh.publicInput =
        machine.encodeInstance (machine.hash (priorHashPreimage currentInput)))
    (freshCarries :
      FreshCarriesPreviousOutput machine previousOutput currentInput) :
    currentInput.running = previousOutput.runningNext ∨
      SecurityFailure machine := by
  rcases preimage_eq_or_securityFailure machine previousInput previousOutput
      currentInput previousOutputHash currentPriorPublic freshCarries with
    preimageEq | failure
  · apply Or.inl
    exact congrArg HashPreimage.running preimageEq
  · exact Or.inr failure

end

end Nightstream.Protocol.FPrime.Paper.PriorLink
