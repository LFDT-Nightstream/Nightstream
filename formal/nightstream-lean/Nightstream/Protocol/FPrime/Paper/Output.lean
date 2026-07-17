import Nightstream.Protocol.FPrime.Paper

/-!
Canonical deterministic output construction for HyperNova Construction 2.

Assurance tier: model-level.

Owns: computation of `pcNext`, `zNext`, the next typed hash preimage, and the
public digest for any chosen next running product.

Does not own: branch acceptance, NIFS validity, output-preimage injectivity,
transcript binding, Rust, R1CS, constraints, or row removal.

Emits constraints: no.

Authority boundary: every output field is computed from the machine, input,
and explicit next running product. Equality of the abstract hash output does
not imply equality of preimages.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.output.compute` | derive `pcNext`, `zNext`, and `x` | computed | `derivedOutput` |
| `fprime.output.application` | the computed counter selects the executed application | derived | `derivedOutput_application` |
| `fprime.output.hash` | the digest uses exactly the typed next-step preimage | derived | `derivedOutput_outputHolds` |
-/

namespace Nightstream.Protocol.FPrime.Paper

open Nightstream.SuperNeo

universe uStructure uPublicInput uPoint uEvaluation uCommitment uState uWitness
  uVerifierKey uDigest

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

/-- Canonical output for a chosen next running product. -/
def derivedOutput
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (runningNext : RunningProduct Structure PublicInput Point Evaluation
      Commitment params slotCount) :
    Output Digest State Structure PublicInput Point Evaluation Commitment params
      slotCount :=
  let pcNext := machine.control input.zi input.witness
  let zNext := machine.step pcNext.index input.zi input.witness
  let preimage : HashPreimage VerifierKey State Structure PublicInput Point
      Evaluation Commitment params slotCount := {
    verifierKey := input.verifierKey
    iteration := input.iteration + 1
    z0 := input.z0
    current := zNext
    running := runningNext
    pc := pcNext.raw
  }
  {
    zNext := zNext
    runningNext := runningNext
    pcNext := pcNext
    x := machine.hash preimage
  }

/-- The canonical output satisfies deterministic control, dispatch, and
application evaluation. -/
theorem derivedOutput_application
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (runningNext : RunningProduct Structure PublicInput Point Evaluation
      Commitment params slotCount) :
    ApplicationHolds machine (machine.control input.zi input.witness).index input
      (derivedOutput machine input runningNext) := by
  constructor
  · rfl
  · exact (ProgramCounter.ofIndex_index
      (machine.control input.zi input.witness)).symm
  · rfl

/-- The canonical output hashes exactly its typed next-step preimage. -/
theorem derivedOutput_outputHolds
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (runningNext : RunningProduct Structure PublicInput Point Evaluation
      Commitment params slotCount) :
    OutputHolds machine input (derivedOutput machine input runningNext) := by
  rfl

end

end Nightstream.Protocol.FPrime.Paper
