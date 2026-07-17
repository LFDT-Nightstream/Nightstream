import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Source
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Execution

/-!
Exact connectivity from terminal-NC round zero into uniform round one.

Assurance tier: conditional implementation/R1CS refinement.

Owns: equality of the round-zero challenge output base with the first
uniform-round input base; construction of that incoming boundary from one
complete call output; and phase composition from FE successor, prologue,
typed carrier coordinate zero, and round-zero execution.

Does not own: derivation of the FE successor; complete carrier allocation;
later-round replay; SumCheck algebra; costs; necessity; or row removal.

Emits constraints: no.

Authority boundary: the successor state is one accepted Poseidon2 call
output. The next round's capacity lanes are derived from exact SSA identity,
not supplied independently or authenticated by a digest.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.0_to_1.layout` | round-zero output base equals uniform round-one input base | derived | `squeezeOutputBase_eq_laterColumnBase` |
| `nifs.pi_ccs.nc_sumcheck.round.0_to_1.incoming` | complete round-zero call output supplies round-one input | derived | `incomingBound_of_callOutputState` |
| `nifs.pi_ccs.nc_sumcheck.round.0_to_1.execution` | prologue plus carrier coordinate zero computes round-one input | conditional composition | `nextIncoming_of_prologueAndCarrier` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Connectivity

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds
open Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private abbrev Input
    {shape : SemanticShape}
    (publicInput : PublicInput shape) :=
  PiCcsTranscript.Exact.Schedule.Input publicInput Carrier.domain

def laterFirst : Fin LaterRound.Artifact.roundCount :=
  ⟨0, by decide⟩

theorem squeezeOutputBase_eq_laterColumnBase :
    Artifact.squeezeOutputBase =
      LaterRound.Artifact.columnBase laterFirst := by
  rfl

theorem incomingBound_of_callOutputState
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP) :
    LaterRound.Execution.IncomingBound laterFirst
      (callOutputState assignment canonical Artifact.squeezeCall)
      assignment canonical := by
  refine {
    cursorZero := rfl
    capacity := ?_
  }
  intro lane _high
  change
    fieldAt assignment canonical
        (Artifact.squeezeCall.columnMap (601 + lane.val)) =
      fieldAt assignment canonical
        (LaterRound.Artifact.columnBase laterFirst + lane.val)
  rw [Artifact.squeezeOutputColumn lane,
    squeezeOutputBase_eq_laterColumnBase]

theorem nextIncoming_of_runRound
    {initial : State}
    {message : RoundMessage}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (incoming : Execution.IncomingBound initial assignment canonical)
    (source : Execution.MessageBound message assignment canonical) :
    LaterRound.Execution.IncomingBound laterFirst
      (runRound initial message).1 assignment canonical := by
  rw [Execution.successor_eq_callOutputState
    canonical one accepted incoming source]
  exact incomingBound_of_callOutputState canonical

theorem nextIncoming_of_prologueAndCarrier
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput)
    (afterFe : State)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (carrierBound : Carrier.Bound input assignment canonical)
    (afterFeBound :
      Prologue.Execution.AfterFeBound afterFe assignment canonical) :
    LaterRound.Execution.IncomingBound laterFirst
      (runRound (ncPrologue afterFe) (Source.typedMessage input)).1
      assignment canonical :=
  nextIncoming_of_runRound canonical one accepted
    (Execution.incomingBound_of_prologue
      canonical one accepted afterFeBound)
    (Source.messageBound_of_carrierBound input canonical carrierBound)

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Connectivity
