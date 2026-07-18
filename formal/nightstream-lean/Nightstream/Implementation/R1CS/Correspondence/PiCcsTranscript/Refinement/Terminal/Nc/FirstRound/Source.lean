import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Carrier
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Execution

/-!
Typed carrier source for terminal-NC semantic round zero.

Assurance tier: conditional implementation/R1CS refinement.

Owns: selection of carrier coordinate zero; equality of the carrier and
round-zero coefficient bases; lossless raw encoding of the selected typed
polynomial; and derivation of the execution message boundary from one
carrier `RoundBound`.

Does not own: proof of the complete carrier boundary from R1CS allocation;
prologue execution; Poseidon2 execution; SumCheck algebra; costs; necessity;
or row removal.

Emits constraints: no.

Authority boundary: the message comes from the independently typed carrier.
Generated columns appear only on the decoded side of `SourceBound`; artifact
acceptance cannot establish this cross-representation equality.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.0.source.index` | select full-carrier coordinate zero | computed | `carrierIndex` |
| `nifs.pi_ccs.nc_sumcheck.round.0.source.layout` | carrier and round-zero coefficient bases agree | derived | `carrier_coefficientBase_eq` |
| `nifs.pi_ccs.nc_sumcheck.round.0.source.message` | raw message is the lossless typed encoding | computed | `typedMessage` |
| `nifs.pi_ccs.nc_sumcheck.round.0.source.bound` | typed coefficients equal the assignment-decoded five pairs | explicit source boundary | `SourceBound` |
| `nifs.pi_ccs.nc_sumcheck.round.0.message.fields` | one source equality entails all ten field equalities | derived | `messageBound_of_sourceBound` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Source

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiCcsTranscript
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private abbrev Input
    {shape : SemanticShape}
    (publicInput : PublicInput shape) :=
  Exact.Schedule.Input publicInput Carrier.domain

def carrierIndex : Fin Carrier.roundCount :=
  ⟨0, by decide⟩

theorem carrier_coefficientBase_eq :
    Carrier.coefficientBase carrierIndex = Artifact.coefficientBase := by
  rfl

def typedMessage
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput) :
    SumCheck.RoundMessage :=
  ExactMessages.encodeFixed (Carrier.typedRound input carrierIndex)

def SourceBound
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop :=
  Carrier.RoundBound input assignment canonical carrierIndex

theorem sourceBound_of_carrierBound
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (bound : Carrier.Bound input assignment canonical) :
    SourceBound input assignment canonical :=
  bound carrierIndex

theorem messageBound_of_sourceBound
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (source : SourceBound input assignment canonical) :
    Execution.MessageBound (typedMessage input) assignment canonical := by
  unfold Execution.MessageBound typedMessage
  unfold SumCheck.roundFields ExactMessages.encodeFixed
  change
    Primitives.extensionFields
        ((Carrier.typedRound input carrierIndex).coefficients.map
          Transport.toExtension) =
      Execution.messageFields assignment canonical
  rw [show
    (Carrier.typedRound input carrierIndex).coefficients =
      Carrier.artifactCoefficients assignment canonical carrierIndex by
    exact source]
  unfold Carrier.artifactCoefficients
  rw [Carrier.coefficientColumns_eq carrierIndex]
  unfold Carrier.expectedCoefficientColumns
  rw [carrier_coefficientBase_eq]
  simp [Primitives.extensionFields, Carrier.semanticCoefficientAt,
    Carrier.semanticFieldAt, Transport.toExtension, Transport.toField,
    Execution.messageFields,
    PiRlcChallenge.Transcript.CallRefinement.fieldAt]

theorem messageBound_of_carrierBound
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (bound : Carrier.Bound input assignment canonical) :
    Execution.MessageBound (typedMessage input) assignment canonical :=
  messageBound_of_sourceBound input canonical
    (sourceBound_of_carrierBound input canonical bound)

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Source
