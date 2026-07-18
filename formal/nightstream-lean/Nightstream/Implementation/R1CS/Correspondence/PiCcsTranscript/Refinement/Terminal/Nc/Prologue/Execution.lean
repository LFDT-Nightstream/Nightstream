import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Prologue.Artifact
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.RoundExecution

/-!
Semantic execution of the terminal-NC prologue.

Assurance tier: conditional implementation/R1CS refinement.

Owns: the minimal FE-successor input surface; both fixed prologue message
boundaries; execution of both accepted Poseidon2 calls; and the exact
cursor-one state carrying round tag `10` into semantic round zero.

Does not own: derivation of the FE successor from the FE phase; round-zero
message coefficients; SumCheck algebra; costs; necessity; or row removal.

Emits constraints: no.

Authority boundary: only the cursor and four retained capacity lanes cross
from FE. All NC-domain fields are verifier constants proved from accepted
rows, and both successor permutations are computed by accepted Poseidon2
calls. The round tag is then absorbed by the semantic machine.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe_to_nc.state` | FE ends at cursor zero and supplies retained lanes four through seven | explicit phase boundary | `AfterFeBound` |
| `nifs.pi_ccs.nc_sumcheck.prologue.permute.0.input` | raw `[8]` and raw `[9]` form the first exact call | derived refinement | `firstBoundary_eq_callInput` |
| `nifs.pi_ccs.nc_sumcheck.prologue.permute.1.input` | raw zero pair and raw `[10]` length form the second exact call | derived refinement | `secondBoundary_eq_callInput` |
| `nifs.pi_ccs.nc_sumcheck.prologue.execution` | the independent prologue reaches the exact cursor-one round-tag state | conditional composition | `run_eq_roundTagState` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.Prologue.Execution

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds
open Nightstream.Implementation.R1CS.PiCcsTranscript.Primitives
open Nightstream.Implementation.R1CS.PiCcsTranscript.SumCheck
open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.RoundExecution

set_option maxHeartbeats 1000000

/-- Minimal state surface crossing from the final FE challenge permutation
into the NC prologue. Lanes zero through three are immediately overwritten. -/
structure AfterFeBound
    (afterFe : State)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop where
  cursorZero : afterFe.absorbed.val = 0
  capacity : ∀ lane : Fin width, 4 ≤ lane.val →
    afterFe.lanes lane =
      fieldAt assignment canonical
        (Artifact.afterFeColumnBase + lane.val)

theorem firstBoundary_eq_callInput
    {afterFe : State}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (facts : Artifact.Facts assignment)
    (incoming : AfterFeBound afterFe assignment canonical) :
    absorbAll afterFe
        [wordField 1, wordField 8, wordField 1, wordField 9] =
      callInputState assignment canonical Artifact.firstCall
        ⟨rate, by decide⟩ := by
  have domainLength :
      fieldAt assignment canonical Artifact.domainLengthColumn =
        wordField 1 :=
    fieldAt_eq_wordField canonical facts.domainLength
      (by decide) (by decide)
  have domainTag :
      fieldAt assignment canonical Artifact.domainTagColumn =
        wordField 8 :=
    fieldAt_eq_wordField canonical facts.domainTag
      (by decide) (by decide)
  have initialTagLength :
      fieldAt assignment canonical Artifact.initialTagLengthColumn =
        wordField 1 :=
    fieldAt_eq_wordField canonical facts.initialTagLength
      (by decide) (by decide)
  have initialTag :
      fieldAt assignment canonical Artifact.initialTagColumn =
        wordField 9 :=
    fieldAt_eq_wordField canonical facts.initialTag
      (by decide) (by decide)
  rw [absorbAll_four_of_cursorZero afterFe incoming.cursorZero]
  apply stateExt
  · funext lane
    by_cases high : 4 ≤ lane.val
    · have capacity := incoming.capacity lane high
      rcases laneValueCases lane with
        h | h | h | h | h | h | h | h
      all_goals
        apply Fin.ext
        simp [fullBuffer, overwriteLane, callInputState,
          Artifact.firstCall, Poseidon2Call.Call.columnMap,
          h, domainLength, domainTag, initialTagLength, initialTag,
          capacity]
    · rcases laneValueCases lane with
        h | h | h | h | h | h | h | h
      all_goals
        apply Fin.ext
        simp [fullBuffer, overwriteLane, callInputState,
          Artifact.firstCall, Poseidon2Call.Call.columnMap,
          h, domainLength, domainTag, initialTagLength, initialTag] at *
  · apply Fin.ext
    simp [fullBuffer, callInputState]

theorem secondBoundary_eq_callInput
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (facts : Artifact.Facts assignment) :
    absorbAll
        (callOutputState assignment canonical Artifact.firstCall)
        [wordField 2, wordField 0, wordField 0, wordField 1] =
      callInputState assignment canonical Artifact.secondCall
        ⟨rate, by decide⟩ := by
  have zeroLength :
      fieldAt assignment canonical Artifact.zeroLengthColumn =
        wordField 2 :=
    fieldAt_eq_wordField canonical facts.zeroLength
      (by decide) (by decide)
  have zeroC0 :
      fieldAt assignment canonical Artifact.zeroC0Column =
        wordField 0 :=
    fieldAt_eq_wordField canonical facts.zeroC0
      (by decide) (by decide)
  have zeroC1 :
      fieldAt assignment canonical Artifact.zeroC1Column =
        wordField 0 :=
    fieldAt_eq_wordField canonical facts.zeroC1
      (by decide) (by decide)
  have roundTagLength :
      fieldAt assignment canonical Artifact.roundTagLengthColumn =
        wordField 1 :=
    fieldAt_eq_wordField canonical facts.roundTagLength
      (by decide) (by decide)
  rw [absorbAll_four_of_cursorZero
    (callOutputState assignment canonical Artifact.firstCall) (by rfl)]
  apply stateExt
  · funext lane
    rcases laneValueCases lane with
      h | h | h | h | h | h | h | h
    all_goals
      apply Fin.ext
      simp [fullBuffer, overwriteLane, callInputState, callOutputState,
        Artifact.firstCall, Artifact.secondCall, Artifact.firstOutputBase,
        Poseidon2Call.Call.columnMap, h, zeroLength, zeroC0, zeroC1,
        roundTagLength]
  · apply Fin.ext
    simp [fullBuffer, callInputState, callOutputState]

/-- The exact independent NC prologue ends at the accepted second-call
output with the verifier-owned round tag absorbed into lane zero. -/
theorem run_eq_roundTagState
    {afterFe : State}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (incoming : AfterFeBound afterFe assignment canonical) :
    ncPrologue afterFe =
      absorbElem
        (callOutputState assignment canonical Artifact.secondCall)
        (wordField 10) := by
  have facts := Artifact.facts canonical one accepted
  have firstInput :=
    firstBoundary_eq_callInput canonical facts incoming
  have firstExecution :=
    callAccepted_permute canonical one Artifact.firstCall
      ⟨rate, by decide⟩ (Artifact.firstCallAccepted accepted)
  have secondInput := secondBoundary_eq_callInput canonical facts
  have secondExecution :=
    callAccepted_permute canonical one Artifact.secondCall
      ⟨rate, by decide⟩ (Artifact.secondCallAccepted accepted)
  have firstStage :
      appendRaw (appendRaw afterFe [wordField 8]) [wordField 9] =
        callOutputState assignment canonical Artifact.firstCall := by
    rw [appendRaw_singletons_of_cursorZero
      afterFe incoming.cursorZero (wordField 8) (wordField 9)]
    rw [firstInput, firstExecution]
  unfold ncPrologue
  change
    appendRaw
        (appendRaw
          (appendRaw
            (appendRaw afterFe [wordField 8])
            [wordField 9])
          [wordField 0, wordField 0])
        [wordField 10] =
      absorbElem
        (callOutputState assignment canonical Artifact.secondCall)
        (wordField 10)
  rw [firstStage]
  rw [appendRaw_pair_then_singleton_of_cursorZero
    (callOutputState assignment canonical Artifact.firstCall)
    (by rfl) (wordField 0) (wordField 0) (wordField 10)]
  rw [secondInput, secondExecution]

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.Prologue.Execution
