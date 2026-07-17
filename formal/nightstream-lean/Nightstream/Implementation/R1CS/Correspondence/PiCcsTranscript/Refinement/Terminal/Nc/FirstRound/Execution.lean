import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Artifact
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Prologue.Execution

/-!
Semantic execution of terminal-NC round zero.

Assurance tier: conditional implementation/R1CS refinement.

Owns: the distinct cursor-one input surface; one ten-field message boundary;
all three message permutations; the cursor-one challenge input; equality of
the semantic successor with the exact accepted squeeze-call output; and
derivation of the incoming surface from prologue execution.

Does not own: typed coefficient authority; derivation of the FE successor;
SumCheck algebra; costs; necessity; or row removal.

Emits constraints: no.

Authority boundary: the prologue supplies one computed cursor-one state and
the caller supplies one lossless coefficient binding. Constants and all four
permutations come from accepted rows. No digest or challenge is accepted as
authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.0.message.fields` | ten semantic fields equal the exact coefficient columns | explicit source boundary | `MessageBound` |
| `nifs.pi_ccs.nc_sumcheck.round.0.message.incoming` | cursor-one tag and retained capacity match prologue output | derived phase boundary | `IncomingBound`, `incomingBound_of_prologue` |
| `nifs.pi_ccs.nc_sumcheck.round.0.message.permute.0` | tag, length, and first coefficient pair form the first call | derived refinement | `firstBoundary_eq_callInput` |
| `nifs.pi_ccs.nc_sumcheck.round.0.message.permute.1` | coefficient fields two through five form the second call | derived refinement | `secondBoundary_eq_callInput` |
| `nifs.pi_ccs.nc_sumcheck.round.0.message.permute.2` | coefficient fields six through nine form the third call | derived refinement | `thirdBoundary_eq_callInput` |
| `nifs.pi_ccs.nc_sumcheck.round.0.challenge.input` | marker one plus retained lanes form the challenge call | derived refinement | `squeezeBoundary_eq_callInput` |
| `nifs.pi_ccs.nc_sumcheck.round.0.execution` | semantic successor equals the accepted challenge-call output | conditional composition | `successor_eq_callOutputState` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Execution

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives
open Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck
open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution

set_option maxHeartbeats 1000000

def messageFields
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    List Field :=
  [fieldAt assignment canonical Artifact.coefficientBase,
   fieldAt assignment canonical (Artifact.coefficientBase + 1),
   fieldAt assignment canonical (Artifact.coefficientBase + 2),
   fieldAt assignment canonical (Artifact.coefficientBase + 3),
   fieldAt assignment canonical (Artifact.coefficientBase + 4),
   fieldAt assignment canonical (Artifact.coefficientBase + 5),
   fieldAt assignment canonical (Artifact.coefficientBase + 6),
   fieldAt assignment canonical (Artifact.coefficientBase + 7),
   fieldAt assignment canonical (Artifact.coefficientBase + 8),
   fieldAt assignment canonical (Artifact.coefficientBase + 9)]

def MessageBound
    (message : RoundMessage)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop :=
  roundFields message = messageFields assignment canonical

/-- Only the cursor-one surface that survives until the first round-zero
message permutation. Lanes one through three are overwritten. -/
structure IncomingBound
    (initial : State)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop where
  cursorOne : initial.absorbed.val = 1
  roundTag :
    initial.lanes ⟨0, by decide⟩ =
      fieldAt assignment canonical Prologue.Artifact.roundTagColumn
  capacity : ∀ lane : Fin width, 4 ≤ lane.val →
    initial.lanes lane =
      fieldAt assignment canonical
        (Prologue.Artifact.secondOutputBase + lane.val)

def PermutationBound
    (successor : State)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop :=
  ∃ permutationInput : State,
    permutationInput =
      callInputState assignment canonical Artifact.squeezeCall
        ⟨1, by decide⟩ ∧
    successor = permute permutationInput

theorem firstBoundary_eq_callInput
    {initial : State}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (facts : Artifact.Facts assignment)
    (incoming : IncomingBound initial assignment canonical) :
    absorbAll initial
        [wordField 10,
         fieldAt assignment canonical Artifact.coefficientBase,
         fieldAt assignment canonical (Artifact.coefficientBase + 1)] =
      callInputState assignment canonical Artifact.firstMessageCall
        ⟨rate, by decide⟩ := by
  have messageLength :
      fieldAt assignment canonical Artifact.messageLengthColumn =
        wordField 10 :=
    fieldAt_eq_wordField canonical facts.messageLength
      (by decide) (by decide)
  rw [absorbAll_three_of_cursorOne initial incoming.cursorOne]
  apply stateExt
  · funext lane
    by_cases high : 4 ≤ lane.val
    · have capacity := incoming.capacity lane high
      rcases laneValueCases lane with
        h | h | h | h | h | h | h | h
      all_goals try omega
      all_goals
        apply Fin.ext
        simp [cursorOneFullBuffer, overwriteLane, callInputState,
          Artifact.firstMessageCall, Poseidon2Call.Call.columnMap,
          h, incoming.roundTag, messageLength, capacity]
    · rcases laneValueCases lane with
        h | h | h | h | h | h | h | h
      · have laneEq : lane = ⟨0, by decide⟩ := by
          apply Fin.ext
          exact h
        rw [laneEq]
        change
          initial.lanes ⟨0, by decide⟩ =
            fieldAt assignment canonical
              Prologue.Artifact.roundTagColumn
        exact incoming.roundTag
      all_goals
        apply Fin.ext
        simp [cursorOneFullBuffer, overwriteLane, callInputState,
          Artifact.firstMessageCall, Poseidon2Call.Call.columnMap,
          h, messageLength] at *
  · apply Fin.ext
    simp [cursorOneFullBuffer, callInputState]

theorem secondBoundary_eq_callInput
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP) :
    absorbAll
        (callOutputState assignment canonical Artifact.firstMessageCall)
        [fieldAt assignment canonical (Artifact.coefficientBase + 2),
         fieldAt assignment canonical (Artifact.coefficientBase + 3),
         fieldAt assignment canonical (Artifact.coefficientBase + 4),
         fieldAt assignment canonical (Artifact.coefficientBase + 5)] =
      callInputState assignment canonical Artifact.secondMessageCall
        ⟨rate, by decide⟩ := by
  rw [absorbAll_four_of_cursorZero
    (callOutputState assignment canonical Artifact.firstMessageCall)
    (by rfl)]
  apply stateExt
  · funext lane
    rcases laneValueCases lane with
      h | h | h | h | h | h | h | h
    all_goals
      apply Fin.ext
      simp [fullBuffer, overwriteLane, callInputState, callOutputState,
        Artifact.firstMessageCall, Artifact.secondMessageCall,
        Artifact.firstOutputBase, Poseidon2Call.Call.columnMap, h]
  · apply Fin.ext
    simp [fullBuffer, callInputState, callOutputState]

theorem thirdBoundary_eq_callInput
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP) :
    absorbAll
        (callOutputState assignment canonical Artifact.secondMessageCall)
        [fieldAt assignment canonical (Artifact.coefficientBase + 6),
         fieldAt assignment canonical (Artifact.coefficientBase + 7),
         fieldAt assignment canonical (Artifact.coefficientBase + 8),
         fieldAt assignment canonical (Artifact.coefficientBase + 9)] =
      callInputState assignment canonical Artifact.thirdMessageCall
        ⟨rate, by decide⟩ := by
  rw [absorbAll_four_of_cursorZero
    (callOutputState assignment canonical Artifact.secondMessageCall)
    (by rfl)]
  apply stateExt
  · funext lane
    rcases laneValueCases lane with
      h | h | h | h | h | h | h | h
    all_goals
      apply Fin.ext
      simp [fullBuffer, overwriteLane, callInputState, callOutputState,
        Artifact.secondMessageCall, Artifact.thirdMessageCall,
        Artifact.secondOutputBase, Poseidon2Call.Call.columnMap, h]
  · apply Fin.ext
    simp [fullBuffer, callInputState, callOutputState]

theorem squeezeBoundary_eq_callInput
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (facts : Artifact.Facts assignment) :
    absorbElem
        (callOutputState assignment canonical Artifact.thirdMessageCall)
        (wordField 1) =
      callInputState assignment canonical Artifact.squeezeCall
        ⟨1, by decide⟩ := by
  have marker :
      fieldAt assignment canonical Artifact.squeezeMarkerColumn =
        wordField 1 :=
    fieldAt_eq_wordField canonical facts.squeezeMarker
      (by decide) (by decide)
  have room :
      (callOutputState assignment canonical
        Artifact.thirdMessageCall).absorbed.val < rate := by
    simp [callOutputState, rate]
  apply stateExt
  · rw [absorbElem_lanes_of_room _ _ room]
    funext lane
    rcases laneValueCases lane with
      h | h | h | h | h | h | h | h
    all_goals
      apply Fin.ext
      simp [overwriteLane, callInputState, callOutputState,
        Artifact.thirdMessageCall, Artifact.squeezeCall,
        Artifact.thirdOutputBase, Poseidon2Call.Call.columnMap, h, marker]
  · apply Fin.ext
    rw [absorbElem_absorbed_of_room _ _ room]
    simp [callOutputState, callInputState]

theorem permutationBound_of_runRound
    {initial : State}
    {message : RoundMessage}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (incoming : IncomingBound initial assignment canonical)
    (source : MessageBound message assignment canonical) :
    PermutationBound (runRound initial message).1 assignment canonical := by
  have facts := Artifact.facts canonical one accepted
  have firstInput := firstBoundary_eq_callInput canonical facts incoming
  have firstExecution :=
    callAccepted_permute canonical one Artifact.firstMessageCall
      ⟨rate, by decide⟩ (Artifact.firstMessageCallAccepted accepted)
  have secondInput := secondBoundary_eq_callInput canonical
  have secondExecution :=
    callAccepted_permute canonical one Artifact.secondMessageCall
      ⟨rate, by decide⟩ (Artifact.secondMessageCallAccepted accepted)
  have thirdInput := thirdBoundary_eq_callInput canonical
  have thirdExecution :=
    callAccepted_permute canonical one Artifact.thirdMessageCall
      ⟨rate, by decide⟩ (Artifact.thirdMessageCallAccepted accepted)
  have squeezeInput := squeezeBoundary_eq_callInput canonical facts
  have afterMessage :
      appendRaw initial (roundFields message) =
        callOutputState assignment canonical Artifact.thirdMessageCall := by
    rw [source]
    unfold messageFields
    rw [appendRaw_ten_of_cursorOne initial incoming.cursorOne]
    rw [firstInput, firstExecution, secondInput, secondExecution,
      thirdInput, thirdExecution]
  refine
    ⟨callInputState assignment canonical Artifact.squeezeCall
        ⟨1, by decide⟩, rfl, ?_⟩
  unfold runRound squeezeN blocksFor squeezeBlocks digest
  rw [afterMessage]
  change
    permute
        (absorbElem
          (callOutputState assignment canonical Artifact.thirdMessageCall)
          (wordField 1)) =
      permute
        (callInputState assignment canonical Artifact.squeezeCall
          ⟨1, by decide⟩)
  rw [squeezeInput]

theorem successor_eq_callOutputState
    {initial : State}
    {message : RoundMessage}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (incoming : IncomingBound initial assignment canonical)
    (source : MessageBound message assignment canonical) :
    (runRound initial message).1 =
      callOutputState assignment canonical Artifact.squeezeCall := by
  rcases permutationBound_of_runRound canonical one accepted incoming source with
    ⟨permutationInput, inputBound, successorBound⟩
  have callRefinement :=
    callAccepted_permute canonical one Artifact.squeezeCall
      ⟨1, by decide⟩ (Artifact.squeezeCallAccepted accepted)
  rw [successorBound, inputBound, callRefinement]

/-- Prologue execution constructs the complete round-zero incoming surface;
no caller supplies its tag or capacity lanes independently. -/
theorem incomingBound_of_prologue
    {afterFe : State}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (afterFeBound :
      Prologue.Execution.AfterFeBound afterFe assignment canonical) :
    IncomingBound (ncPrologue afterFe) assignment canonical := by
  have prologueEq :=
    Prologue.Execution.run_eq_roundTagState
      canonical one accepted afterFeBound
  have prologueFacts := Prologue.Artifact.facts canonical one accepted
  have tagField :
      fieldAt assignment canonical Prologue.Artifact.roundTagColumn =
        wordField 10 :=
    fieldAt_eq_wordField canonical prologueFacts.roundTag
      (by decide) (by decide)
  rw [prologueEq]
  have room :
      (callOutputState assignment canonical
        Prologue.Artifact.secondCall).absorbed.val < rate := by
    simp [callOutputState, rate]
  refine {
    cursorOne := ?_
    roundTag := ?_
    capacity := ?_
  }
  · rw [absorbElem_absorbed_of_room _ _ room]
    rfl
  · rw [absorbElem_lanes_of_room _ _ room]
    apply Fin.ext
    simp [overwriteLane, callOutputState, tagField]
  · intro lane high
    rw [absorbElem_lanes_of_room _ _ room]
    change
      overwriteLane
          (callOutputState assignment canonical
            Prologue.Artifact.secondCall).lanes
          0 (wordField 10) lane =
        fieldAt assignment canonical
          (Prologue.Artifact.secondOutputBase + lane.val)
    have laneNonzero : lane.val ≠ 0 := by omega
    rw [show
      overwriteLane
          (callOutputState assignment canonical
            Prologue.Artifact.secondCall).lanes
          0 (wordField 10) lane =
        (callOutputState assignment canonical
          Prologue.Artifact.secondCall).lanes lane by
      simp [overwriteLane, laneNonzero]]
    change
      fieldAt assignment canonical
          (Prologue.Artifact.secondCall.columnMap (601 + lane.val)) =
        fieldAt assignment canonical
          (Prologue.Artifact.secondOutputBase + lane.val)
    rw [Prologue.Artifact.secondOutputColumn lane]

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Execution
