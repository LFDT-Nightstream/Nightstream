import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Carrier
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Execution

/-!
Typed carrier source for terminal-NC rounds one through fourteen.

Assurance tier: conditional implementation/R1CS refinement.

Owns: the index embedding from uniform later-round coordinates into the
complete fifteen-round carrier; equality of their coefficient-base formulas;
lossless raw encoding of the selected typed polynomial; and derivation of
the execution message boundary from one carrier `RoundBound`.

Does not own: proof of the complete carrier boundary from R1CS allocation;
inter-round state connectivity; Poseidon2 execution; SumCheck algebra;
costs; necessity; or row removal.

Emits constraints: no.

Authority boundary: the selected message comes from the independently typed
carrier. Generated columns appear only on the decoded side of `SourceBound`;
artifact acceptance cannot establish this cross-representation equality.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.1_14.source.index` | later coordinate `r` selects full-carrier coordinate `r+1` | computed | `carrierIndex` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.source.layout` | carrier and later-round coefficient bases agree | derived | `carrier_coefficientBase_eq` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.source.message` | selected raw message is the lossless typed encoding | computed | `typedMessage` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.source.bound` | typed coefficients equal the assignment-decoded five pairs | explicit source boundary | `SourceBound` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.message.fields` | one source equality entails all ten field equalities | derived | `messageBound_of_sourceBound` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Source

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiCcsTranscript
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private abbrev Input
    {shape : SemanticShape}
    (publicInput : PublicInput shape) :=
  Exact.Schedule.Input publicInput Carrier.domain

/-- Uniform later-round coordinate `r` is semantic NC coordinate `r+1`. -/
def carrierIndex (round : Fin Artifact.roundCount) :
    Fin Carrier.roundCount :=
  ⟨round.val + 1, by
    change round.val + 1 < 15
    have roundLt := round.isLt
    change round.val < 14 at roundLt
    omega⟩

@[simp] theorem carrierIndex_val (round : Fin Artifact.roundCount) :
    (carrierIndex round).val = round.val + 1 :=
  rfl

/-- The independent carrier layout and indexed execution layout select the
same ten physical columns. -/
theorem carrier_coefficientBase_eq
    (round : Fin Artifact.roundCount) :
    Carrier.coefficientBase (carrierIndex round) =
      Artifact.coefficientBase round := by
  unfold Carrier.coefficientBase Artifact.coefficientBase
  rw [carrierIndex_val]
  omega

/-- Lossless raw transcript message selected from the typed carrier. -/
def typedMessage
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    (round : Fin Artifact.roundCount) :
    SumCheck.RoundMessage :=
  ExactMessages.encodeFixed
    (Carrier.typedRound input (carrierIndex round))

/-- Sole typed-to-assignment source boundary for one later round. -/
def SourceBound
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    (round : Fin Artifact.roundCount)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop :=
  Carrier.RoundBound input assignment canonical (carrierIndex round)

/-- The complete carrier decoder supplies every later-round source leaf. -/
theorem sourceBound_of_carrierBound
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (bound : Carrier.Bound input assignment canonical)
    (round : Fin Artifact.roundCount) :
    SourceBound input round assignment canonical :=
  bound (carrierIndex round)

/-- One typed coefficient-source equality entails the exact ten-field
execution boundary in constant-first coefficient order. -/
theorem messageBound_of_sourceBound
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    (round : Fin Artifact.roundCount)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (source : SourceBound input round assignment canonical) :
    Execution.MessageBound round (typedMessage input round)
      assignment canonical := by
  unfold Execution.MessageBound typedMessage
  unfold SumCheck.roundFields ExactMessages.encodeFixed
  change
    Primitives.extensionFields
        ((Carrier.typedRound input
          (carrierIndex round)).coefficients.map Transport.toExtension) =
      Execution.messageFields round assignment canonical
  rw [show
    (Carrier.typedRound input
      (carrierIndex round)).coefficients =
        Carrier.artifactCoefficients assignment canonical
          (carrierIndex round) by
    exact source]
  unfold Carrier.artifactCoefficients
  rw [Carrier.coefficientColumns_eq (carrierIndex round)]
  unfold Carrier.expectedCoefficientColumns
  rw [carrier_coefficientBase_eq round]
  simp [Primitives.extensionFields, Carrier.semanticCoefficientAt,
    Carrier.semanticFieldAt, Transport.toExtension, Transport.toField,
    Execution.messageFields,
    PiRlcChallenge.Transcript.CallRefinement.fieldAt]

/-- Preferred later-round message theorem exposing only the complete carrier
boundary. -/
theorem messageBound_of_carrierBound
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    (round : Fin Artifact.roundCount)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (bound : Carrier.Bound input assignment canonical) :
    Execution.MessageBound round (typedMessage input round)
      assignment canonical :=
  messageBound_of_sourceBound input round canonical
    (sourceBound_of_carrierBound input canonical bound round)

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Source
