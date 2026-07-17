import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Source

/-!
Exact state connectivity between uniform terminal-NC later rounds.

Assurance tier: conditional implementation/R1CS refinement.

Owns: the affine equality between one round's squeeze outputs and the next
round's incoming state columns; construction of the next minimal incoming
boundary from one complete call output; and propagation through one semantic
`runRound`.

Does not own: connectivity from the distinct round-zero layout into round
one; replay of the whole later-round list; SumCheck algebra; costs;
necessity; or row removal.

Emits constraints: no.

Authority boundary: connectivity is derived from exact SSA output identity.
It does not assume four fresh capacity equalities at every round and does not
accept a digest as state authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.1_13.next.layout` | round `r` output base equals round `r+1` input base | derived | `squeezeOutputBase_eq_nextColumnBase` |
| `nifs.pi_ccs.nc_sumcheck.round.1_13.next.incoming` | one complete call output supplies the next incoming boundary | derived | `incomingBound_of_callOutputState` |
| `nifs.pi_ccs.nc_sumcheck.round.1_13.next.execution` | one proved round supplies the next round's input | conditional composition | `nextIncoming_of_runRound` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Connectivity

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds
open Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck

/-- Thirteen edges connect the fourteen uniform later rounds. -/
def edgeCount : Nat := 13

def currentRound (edge : Fin edgeCount) :
    Fin Artifact.roundCount :=
  ⟨edge.val, by
    have edgeLt := edge.isLt
    change edge.val < 13 at edgeLt
    change edge.val < 14
    omega⟩

def nextRound (edge : Fin edgeCount) :
    Fin Artifact.roundCount :=
  ⟨edge.val + 1, by
    have edgeLt := edge.isLt
    change edge.val < 13 at edgeLt
    change edge.val + 1 < 14
    omega⟩

@[simp] theorem currentRound_val (edge : Fin edgeCount) :
    (currentRound edge).val = edge.val :=
  rfl

@[simp] theorem nextRound_val (edge : Fin edgeCount) :
    (nextRound edge).val = edge.val + 1 :=
  rfl

/-- One later-round squeeze output state is laid out exactly as the next
later round's incoming state. -/
theorem squeezeOutputBase_eq_nextColumnBase
    (edge : Fin edgeCount) :
    Artifact.squeezeOutputBase (currentRound edge) =
      Artifact.columnBase (nextRound edge) := by
  unfold Artifact.squeezeOutputBase Artifact.squeezeAllocatedColumn
    Artifact.squeezeMarkerColumn Artifact.secondAllocatedColumn
    Artifact.firstAllocatedColumn Artifact.columnBase
  rw [currentRound_val, nextRound_val]
  omega

/-- A complete accepted call output supplies the next round's cursor and
capacity lanes without independent per-lane assumptions. -/
theorem incomingBound_of_callOutputState
    {assignment : Nat → Nat}
    (edge : Fin edgeCount)
    (canonical : ∀ column, assignment column < goldilocksP) :
    Execution.IncomingBound (nextRound edge)
      (callOutputState assignment canonical
        (Artifact.squeezeCall (currentRound edge)))
      assignment canonical := by
  refine {
    cursorZero := rfl
    capacity := ?_
  }
  intro lane _high
  change
    fieldAt assignment canonical
        ((Artifact.squeezeCall (currentRound edge)).columnMap
          (601 + lane.val)) =
      fieldAt assignment canonical
        (Artifact.columnBase (nextRound edge) + lane.val)
  rw [Artifact.squeezeOutputColumn, squeezeOutputBase_eq_nextColumnBase]

/-- A semantically replayed current round provides the exact incoming
boundary for its successor. -/
theorem nextIncoming_of_runRound
    {initial : State}
    {message : RoundMessage}
    {assignment : Nat → Nat}
    (edge : Fin edgeCount)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (incoming :
      Execution.IncomingBound (currentRound edge)
        initial assignment canonical)
    (source :
      Execution.MessageBound (currentRound edge)
        message assignment canonical) :
    Execution.IncomingBound (nextRound edge)
      (runRound initial message).1 assignment canonical := by
  rw [Execution.successor_eq_callOutputState
    (currentRound edge) canonical one accepted incoming source]
  exact incomingBound_of_callOutputState edge canonical

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Connectivity
