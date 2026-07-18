import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Connectivity

/-!
Complete typed replay of terminal-NC rounds one through fourteen.

Assurance tier: conditional implementation/R1CS refinement.

Owns: the ordered fourteen-message list selected from the typed carrier;
the semantic state before every later round; induction of exact incoming
state connectivity across all thirteen edges; and equality of the complete
later-round successor with the final indexed artifact call output.

Does not own: NC prologue or round-zero replay; proof of the carrier decoder
from R1CS allocation; SumCheck algebra; costs; necessity; or row removal.

Emits constraints: no.

Authority boundary: the only state premise is the complete incoming boundary
for semantic round one. Every subsequent state is computed by typed replay
and identified with accepted call outputs. No intermediate digest or
caller-supplied lane equality is authoritative.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.1_14.messages` | exact ordered typed messages for carrier coordinates one through fourteen | computed | `messages` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.state_before` | replay the exact typed prefix before each round | computed | `stateBefore` |
| `nifs.pi_ccs.nc_sumcheck.round.1_13.replay` | every proved successor supplies the next incoming state | derived induction | `incomingBound_all` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.replay` | the complete suffix reaches the final accepted squeeze output | conditional composition | `run_eq_finalCallOutput` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Replay

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

/-- Exact ordered typed raw messages for semantic NC rounds one through
fourteen. -/
def messages
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput) :
    List RoundMessage :=
  List.ofFn fun round : Fin Artifact.roundCount =>
    Source.typedMessage input round

@[simp] theorem messages_length
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput) :
    (messages input).length = Artifact.roundCount := by
  simp [messages]

/-- Semantic state before indexed later round `round`. -/
def stateBefore
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (initial : State)
    (input : Input publicInput)
    (round : Fin Artifact.roundCount) : State :=
  (runRounds initial ((messages input).take round.val)).1

@[simp] theorem stateBefore_zero
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (initial : State)
    (input : Input publicInput) :
    stateBefore initial input ⟨0, by decide⟩ = initial := by
  rfl

/-- Extending the typed prefix by one message is exactly one semantic
`runRound` from the preceding prefix state. -/
theorem stateBefore_next
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (initial : State)
    (input : Input publicInput)
    (edge : Fin Connectivity.edgeCount) :
    stateBefore initial input (Connectivity.nextRound edge) =
      (runRound
        (stateBefore initial input (Connectivity.currentRound edge))
        (Source.typedMessage input
          (Connectivity.currentRound edge))).1 := by
  have indexLt :
      (Connectivity.currentRound edge).val <
        (messages input).length := by
    rw [messages_length]
    exact (Connectivity.currentRound edge).isLt
  unfold stateBefore
  rw [show
    (Connectivity.nextRound edge).val =
      (Connectivity.currentRound edge).val + 1 by rfl]
  rw [List.take_succ_eq_append_getElem indexLt]
  rw [runRounds_append]
  change
    (runRound
      (runRounds initial
        ((messages input).take
          (Connectivity.currentRound edge).val)).1
      ((messages input)[(Connectivity.currentRound edge).val]'indexLt)).1 =
    (runRound
      (runRounds initial
        ((messages input).take
          (Connectivity.currentRound edge).val)).1
      (Source.typedMessage input
        (Connectivity.currentRound edge))).1
  congr 2
  simp [messages]
  congr

/-- Complete carrier decoding and one round-one input boundary induce exact
incoming-state refinement for every later round. -/
theorem incomingBound_all
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (initial : State)
    (input : Input publicInput)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (carrierBound : Carrier.Bound input assignment canonical)
    (firstIncoming :
      Execution.IncomingBound ⟨0, by decide⟩
        initial assignment canonical) :
    ∀ round : Fin Artifact.roundCount,
      Execution.IncomingBound round
        (stateBefore initial input round) assignment canonical := by
  apply Fin.induction
  · simpa using firstIncoming
  · intro edge inductionHypothesis
    have currentEq :
        Connectivity.currentRound edge = Fin.castSucc edge := by
      apply Fin.ext
      rfl
    have nextEq :
        Connectivity.nextRound edge = edge.succ := by
      apply Fin.ext
      rfl
    have currentIncoming :
        Execution.IncomingBound (Connectivity.currentRound edge)
          (stateBefore initial input
            (Connectivity.currentRound edge))
          assignment canonical := by
      simpa [currentEq] using inductionHypothesis
    have currentSource :=
      Source.messageBound_of_carrierBound input
        (Connectivity.currentRound edge) canonical carrierBound
    have propagated :=
      Connectivity.nextIncoming_of_runRound edge canonical one accepted
        currentIncoming currentSource
    rw [← stateBefore_next initial input edge] at propagated
    simpa [nextEq] using propagated

def finalRound : Fin Artifact.roundCount :=
  ⟨13, by decide⟩

/-- The complete later-round list is its first thirteen messages followed by
the typed final message. -/
theorem messages_eq_prefix_append_final
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (input : Input publicInput) :
    messages input =
      (messages input).take finalRound.val ++
        [Source.typedMessage input finalRound] := by
  have indexLt : finalRound.val < (messages input).length := by
    rw [messages_length]
    decide
  have throughFinal :=
    List.take_append_getElem (l := messages input) indexLt
  have completeTake :
      (messages input).take (finalRound.val + 1) =
        messages input := by
    apply List.take_of_length_le
    rw [messages_length]
    decide
  calc
    messages input =
        (messages input).take (finalRound.val + 1) :=
      completeTake.symm
    _ =
        (messages input).take finalRound.val ++
          [(messages input)[finalRound.val]'indexLt] :=
      throughFinal.symm
    _ =
        (messages input).take finalRound.val ++
          [Source.typedMessage input finalRound] := by
      congr 2

/-- Running all fourteen typed later rounds is one final `runRound` from the
computed final prefix state. -/
theorem run_eq_finalRun
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (initial : State)
    (input : Input publicInput) :
    (runRounds initial (messages input)).1 =
      (runRound
        (stateBefore initial input finalRound)
        (Source.typedMessage input finalRound)).1 := by
  rw [messages_eq_prefix_append_final input]
  rw [runRounds_append]
  rfl

/-- Complete later-round replay reaches the exact final indexed squeeze-call
output. The only remaining state premise is the round-one incoming boundary. -/
theorem run_eq_finalCallOutput
    {shape : SemanticShape}
    {publicInput : PublicInput shape}
    (initial : State)
    (input : Input publicInput)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (carrierBound : Carrier.Bound input assignment canonical)
    (firstIncoming :
      Execution.IncomingBound ⟨0, by decide⟩
        initial assignment canonical) :
    (runRounds initial (messages input)).1 =
      callOutputState assignment canonical
        (Artifact.squeezeCall finalRound) := by
  rw [run_eq_finalRun initial input]
  exact Execution.successor_eq_callOutputState finalRound
    canonical one accepted
    (incomingBound_all initial input canonical one accepted
      carrierBound firstIncoming finalRound)
    (Source.messageBound_of_carrierBound input finalRound
      canonical carrierBound)

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Replay
