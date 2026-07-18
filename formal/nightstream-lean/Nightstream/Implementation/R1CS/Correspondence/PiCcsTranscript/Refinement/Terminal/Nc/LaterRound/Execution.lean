import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Artifact
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.RoundExecution

/-!
Indexed semantic execution for terminal-NC rounds one through fourteen.

Assurance tier: conditional implementation/R1CS refinement.

Owns: one uniform ten-field message boundary; the minimal incoming-state
surface; both message permutations; the challenge input; and equality of one
semantic `runRound` successor with the exact accepted artifact call output.

Does not own: round-zero's distinct layout; derivation of typed coefficients
from the carrier; inter-round connectivity; SumCheck algebra; costs;
necessity; or row removal.

Emits constraints: no.

Authority boundary: the caller supplies only the semantic message binding
and the four incoming capacity lanes for the selected round. Length and
squeeze-marker constants come from accepted equations, while all three
permutations come from independently accepted Poseidon2 calls.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.1_14.message.fields` | ten semantic fields equal the indexed coefficient columns | explicit source boundary | `MessageBound` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.message.incoming_capacity` | incoming lanes four through seven equal the indexed state columns | explicit connectivity boundary | `IncomingBound` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.message.permute.0` | length and first three fields form the exact first call | derived refinement | `firstBoundary_eq_callInput` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.message.permute.1` | the next four fields form the exact second call | derived refinement | `secondBoundary_eq_callInput` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.challenge.input` | the last three fields and marker form the exact challenge call | derived refinement | `squeezeBoundary_eq_callInput` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.execution` | semantic successor equals the accepted indexed call output | conditional composition | `successor_eq_callOutputState` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Execution

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives
open Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck
open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution

set_option maxHeartbeats 1000000

/-- Exact assignment-backed field list for one indexed degree-four message. -/
def messageFields
    (round : Fin Artifact.roundCount)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    List Field :=
  [fieldAt assignment canonical (Artifact.coefficientBase round),
   fieldAt assignment canonical (Artifact.coefficientBase round + 1),
   fieldAt assignment canonical (Artifact.coefficientBase round + 2),
   fieldAt assignment canonical (Artifact.coefficientBase round + 3),
   fieldAt assignment canonical (Artifact.coefficientBase round + 4),
   fieldAt assignment canonical (Artifact.coefficientBase round + 5),
   fieldAt assignment canonical (Artifact.coefficientBase round + 6),
   fieldAt assignment canonical (Artifact.coefficientBase round + 7),
   fieldAt assignment canonical (Artifact.coefficientBase round + 8),
   fieldAt assignment canonical (Artifact.coefficientBase round + 9)]

/-- Lossless field-level source binding for one indexed later round. -/
def MessageBound
    (round : Fin Artifact.roundCount)
    (message : RoundMessage)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop :=
  roundFields message = messageFields round assignment canonical

/-- Only the incoming state surface that survives until the first message
permutation. Lanes zero through three are overwritten. -/
structure IncomingBound
    (round : Fin Artifact.roundCount)
    (initial : State)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop where
  cursorZero : initial.absorbed.val = 0
  capacity : ∀ lane : Fin width, 4 ≤ lane.val →
    initial.lanes lane =
      fieldAt assignment canonical (Artifact.columnBase round + lane.val)

/-- Semantic successor is the pure permutation of one exact indexed
challenge-call input. -/
def PermutationBound
    (round : Fin Artifact.roundCount)
    (successor : State)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop :=
  ∃ permutationInput : State,
    permutationInput =
      callInputState assignment canonical (Artifact.squeezeCall round)
        ⟨rate, by decide⟩ ∧
    successor = permute permutationInput

/-- The length word, first three fields, and incoming capacity lanes form the
first exact indexed message-call input. -/
theorem firstBoundary_eq_callInput
    {initial : State}
    {assignment : Nat → Nat}
    (round : Fin Artifact.roundCount)
    (canonical : ∀ column, assignment column < goldilocksP)
    (facts : Artifact.Facts round assignment)
    (incoming : IncomingBound round initial assignment canonical) :
    absorbAll initial
        [wordField 10,
         fieldAt assignment canonical (Artifact.coefficientBase round),
         fieldAt assignment canonical (Artifact.coefficientBase round + 1),
         fieldAt assignment canonical (Artifact.coefficientBase round + 2)] =
      callInputState assignment canonical (Artifact.firstMessageCall round)
        ⟨rate, by decide⟩ := by
  have lengthField :
      fieldAt assignment canonical (Artifact.messageLengthColumn round) =
        wordField 10 :=
    fieldAt_eq_wordField canonical facts.messageLength
      (by decide) (by decide)
  rw [absorbAll_four_of_cursorZero initial incoming.cursorZero]
  apply stateExt
  · funext lane
    by_cases high : 4 ≤ lane.val
    · have capacity := incoming.capacity lane high
      rcases laneValueCases lane with
        h | h | h | h | h | h | h | h
      all_goals
        apply Fin.ext
        simp [fullBuffer, overwriteLane, callInputState,
          Artifact.firstMessageCall, Poseidon2Call.Call.columnMap,
          h, lengthField, capacity]
    · rcases laneValueCases lane with
        h | h | h | h | h | h | h | h
      all_goals
        apply Fin.ext
        simp [fullBuffer, overwriteLane, callInputState,
          Artifact.firstMessageCall, Poseidon2Call.Call.columnMap,
          h, lengthField] at *
  · apply Fin.ext
    simp [fullBuffer, callInputState]

/-- The next four fields and first accepted permutation form the second exact
indexed message-call input. -/
theorem secondBoundary_eq_callInput
    {assignment : Nat → Nat}
    (round : Fin Artifact.roundCount)
    (canonical : ∀ column, assignment column < goldilocksP) :
    absorbAll
        (callOutputState assignment canonical
          (Artifact.firstMessageCall round))
        [fieldAt assignment canonical (Artifact.coefficientBase round + 3),
         fieldAt assignment canonical (Artifact.coefficientBase round + 4),
         fieldAt assignment canonical (Artifact.coefficientBase round + 5),
         fieldAt assignment canonical (Artifact.coefficientBase round + 6)] =
      callInputState assignment canonical (Artifact.secondMessageCall round)
        ⟨rate, by decide⟩ := by
  rw [absorbAll_four_of_cursorZero
    (callOutputState assignment canonical (Artifact.firstMessageCall round))
    (by rfl)]
  apply stateExt
  · funext lane
    rcases laneValueCases lane with
      h | h | h | h | h | h | h | h
    all_goals
      apply Fin.ext
      simp [fullBuffer, overwriteLane, callInputState, callOutputState,
        Artifact.firstMessageCall, Artifact.secondMessageCall,
        Artifact.firstOutputBase,
        Poseidon2Call.Call.columnMap, h]
  · apply Fin.ext
    simp [fullBuffer, callInputState, callOutputState]

/-- The last three fields, accepted second permutation, and accepted marker
form the exact indexed challenge-call input. -/
theorem squeezeBoundary_eq_callInput
    {assignment : Nat → Nat}
    (round : Fin Artifact.roundCount)
    (canonical : ∀ column, assignment column < goldilocksP)
    (facts : Artifact.Facts round assignment) :
    absorbAll
        (callOutputState assignment canonical
          (Artifact.secondMessageCall round))
        [fieldAt assignment canonical (Artifact.coefficientBase round + 7),
         fieldAt assignment canonical (Artifact.coefficientBase round + 8),
         fieldAt assignment canonical (Artifact.coefficientBase round + 9),
         wordField 1] =
      callInputState assignment canonical (Artifact.squeezeCall round)
        ⟨rate, by decide⟩ := by
  have markerField :
      fieldAt assignment canonical (Artifact.squeezeMarkerColumn round) =
        wordField 1 :=
    fieldAt_eq_wordField canonical facts.squeezeMarker
      (by decide) (by decide)
  rw [absorbAll_four_of_cursorZero
    (callOutputState assignment canonical (Artifact.secondMessageCall round))
    (by rfl)]
  apply stateExt
  · funext lane
    rcases laneValueCases lane with
      h | h | h | h | h | h | h | h
    all_goals
      apply Fin.ext
      simp [fullBuffer, overwriteLane, callInputState, callOutputState,
        Artifact.secondMessageCall, Artifact.squeezeCall,
        Artifact.secondOutputBase,
        Poseidon2Call.Call.columnMap, h, markerField]
  · apply Fin.ext
    simp [fullBuffer, callInputState, callOutputState]

/-- Accepted artifact rows plus the two explicit input leaves prove that one
semantic later round reaches the exact indexed final permutation. -/
theorem permutationBound_of_runRound
    {initial : State}
    {message : RoundMessage}
    {assignment : Nat → Nat}
    (round : Fin Artifact.roundCount)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (incoming : IncomingBound round initial assignment canonical)
    (source : MessageBound round message assignment canonical) :
    PermutationBound round (runRound initial message).1
      assignment canonical := by
  have roundFacts := Artifact.facts canonical one accepted round
  have firstInput :=
    firstBoundary_eq_callInput round canonical roundFacts incoming
  have firstExecution :=
    callAccepted_permute canonical one (Artifact.firstMessageCall round)
      ⟨rate, by decide⟩
      (Artifact.firstMessageCallAccepted accepted round)
  have secondInput := secondBoundary_eq_callInput round canonical
  have secondExecution :=
    callAccepted_permute canonical one (Artifact.secondMessageCall round)
      ⟨rate, by decide⟩
      (Artifact.secondMessageCallAccepted accepted round)
  have squeezeInput :=
    squeezeBoundary_eq_callInput round canonical roundFacts
  have afterMessage :
      appendRaw initial (roundFields message) =
        absorbAll
          (callOutputState assignment canonical
            (Artifact.secondMessageCall round))
          [fieldAt assignment canonical
              (Artifact.coefficientBase round + 7),
           fieldAt assignment canonical
              (Artifact.coefficientBase round + 8),
           fieldAt assignment canonical
              (Artifact.coefficientBase round + 9)] := by
    rw [source]
    unfold messageFields
    rw [appendRaw_ten_of_cursorZero initial incoming.cursorZero]
    rw [firstInput, firstExecution, secondInput, secondExecution]
  refine
    ⟨callInputState assignment canonical (Artifact.squeezeCall round)
        ⟨rate, by decide⟩, rfl, ?_⟩
  unfold runRound squeezeN blocksFor squeezeBlocks digest
  rw [afterMessage]
  change
    permute
        (absorbAll
          (callOutputState assignment canonical
            (Artifact.secondMessageCall round))
          [fieldAt assignment canonical
              (Artifact.coefficientBase round + 7),
           fieldAt assignment canonical
              (Artifact.coefficientBase round + 8),
           fieldAt assignment canonical
              (Artifact.coefficientBase round + 9),
           wordField 1]) =
      permute
        (callInputState assignment canonical (Artifact.squeezeCall round)
          ⟨rate, by decide⟩)
  rw [squeezeInput]

/-- Accepted challenge-call rows identify the complete semantic successor
state, enabling exact inter-round connectivity rather than four independent
lane assumptions. -/
theorem successor_eq_callOutputState
    {initial : State}
    {message : RoundMessage}
    {assignment : Nat → Nat}
    (round : Fin Artifact.roundCount)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (incoming : IncomingBound round initial assignment canonical)
    (source : MessageBound round message assignment canonical) :
    (runRound initial message).1 =
      callOutputState assignment canonical (Artifact.squeezeCall round) := by
  rcases permutationBound_of_runRound
      round canonical one accepted incoming source with
    ⟨permutationInput, inputBound, successorBound⟩
  have callRefinement :=
    callAccepted_permute canonical one (Artifact.squeezeCall round)
      ⟨rate, by decide⟩ (Artifact.squeezeCallAccepted accepted round)
  rw [successorBound, inputBound, callRefinement]

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Execution
