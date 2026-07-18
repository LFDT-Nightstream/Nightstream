import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.OutputDigestPins
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.Schedule

/-!
Minimal observable post-NC boundary for the terminal `Pi_CCS` catch-up.

Assurance tier: conditional implementation/R1CS refinement.

Owns: the exact state surface observed by the catch-up permutation; separate
children for computed cursor control flow and the seven retained artifact
lanes; composition of an exact positive NC schedule into the cursor child;
and proof that post-NC lane zero is overwritten by the verifier marker.

Does not own: refinement of accepted FE/NC transcript rows into the seven
tail lanes; NC terminal algebra; catch-up permutation acceptance; output
message authority; Rust conformance; costs; or row removal.

Emits constraints: no.

Authority boundary: `Bound` intentionally omits post-NC lane zero. The next
verifier operation overwrites that lane with the equation-bound word `1`, so
requiring it as an authoritative carried value would be stronger than the
actual recursive verifier semantics.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.post_state.cursor` | positive exact NC replay ends at cursor zero | computed control flow | `CursorZero`, `Bound.ofExactSchedule` |
| `nifs.pi_ccs.nc.post_state.retained_lanes` | lanes one through seven equal the exact artifact tail columns | checked boundary | `RetainedLanesBound` |
| `nifs.pi_ccs.nc.post_state` | compose cursor control flow with retained lane authority | grouped boundary | `Bound` |
| `nifs.pi_ccs.catchup.marker` | catch-up overwrites lane zero with the verifier word `1` | computed | `refines_catchupInput` |
| `nifs.pi_ccs.catchup.lane0` | the previous lane-zero value is unobservable at this boundary | derived/eliminated | `laneZero_irrelevant` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.PostNcBoundary

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds

/-- Artifact column observed for one retained post-NC lane. Lane zero is
listed for arithmetic convenience but is deliberately outside `Bound.tail`. -/
def tailColumn (lane : Fin width) : Nat :=
  1692820 + lane.val

/-- Cursor state required by the next catch-up absorption. For exact positive
NC replay this proposition is computed by `Bound.ofExactSchedule`; it is
named separately so it cannot be confused with artifact-lane authority. -/
def CursorZero (postNc : State) : Prop :=
  postNc.absorbed.val = 0

/-- Exactly the post-NC lanes that survive the verifier marker overwrite. -/
structure RetainedLanesBound
    (postNc : State)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) : Prop where
  lane : forall lane : Fin width, lane.val ≠ 0 ->
    postNc.lanes lane =
      fieldAt assignment canonical (tailColumn lane)

/-- Smallest grouped state relation sufficient for the next catch-up
operation. Cursor control flow and retained artifact lanes remain visibly
separate children. -/
structure Bound
    (postNc : State)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) : Prop where
  cursorZero : CursorZero postNc
  retainedLanes : RetainedLanesBound postNc assignment canonical

namespace Bound

/-- Exact positive NC replay computes the cursor child. The caller supplies
only the seven retained lane equalities still awaiting row refinement. -/
theorem ofExactSchedule
    {shape :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.SemanticShape}
    {publicInput :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.PublicInput shape}
    {domain :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.FlatNcDomain}
    (input :
      PiCcsTranscript.Exact.Schedule.Input publicInput domain)
    (positive :
      0 <
        Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Transcript.Nc.roundCount
          domain)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (retainedLanes :
      RetainedLanesBound
        (PiCcsTranscript.Exact.Schedule.run input).afterNc
        assignment canonical) :
    Bound
      (PiCcsTranscript.Exact.Schedule.run input).afterNc
      assignment canonical where
  cursorZero :=
    PiCcsTranscript.Exact.Schedule.run_afterNc_absorbed_zero
      input positive
  retainedLanes := retainedLanes

end Bound

private theorem stateExt
    {left right : State}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem laneValueCases (lane : Fin width) :
    lane.val = 0 ∨ lane.val = 1 ∨ lane.val = 2 ∨ lane.val = 3 ∨
    lane.val = 4 ∨ lane.val = 5 ∨ lane.val = 6 ∨ lane.val = 7 := by
  have laneLt : lane.val < 8 := by
    simpa [width] using lane.isLt
  omega

/-- Any previous value in lane zero disappears after the verifier absorbs the
catch-up marker. This is the formal reason lane zero is absent from `Bound`. -/
theorem laneZero_irrelevant
    (postNc : State)
    (cursorZero : CursorZero postNc)
    (replacement marker : Field) :
    absorbElem
        { lanes := overwriteLane postNc.lanes 0 replacement
          absorbed := postNc.absorbed }
        marker =
      absorbElem postNc marker := by
  change postNc.absorbed.val = 0 at cursorZero
  have room : postNc.absorbed.val < rate := by
    rw [cursorZero]
    decide
  unfold absorbElem
  rw [dif_pos room, dif_pos room]
  apply stateExt
  · funext lane
    by_cases laneZero : lane.val = 0 <;>
      simp [overwriteLane, cursorZero, laneZero]
  · apply Fin.ext
    simp [cursorZero]

/-- The minimal post-NC boundary plus the independently pinned marker is
exactly the input state replayed by the accepted catch-up permutation. -/
theorem refines_catchupInput
    {postNc : State}
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (catchupAccepted :
      FPrimeFullHistoryTerminalPiCcsCatchup.Accepted assignment)
    (bound : Bound postNc assignment canonical) :
    absorbElem postNc (wordField 1) =
      callInputState assignment canonical
        PiRlcChallenge.Transcript.Terminal.OutputDigestSchedule.catchupCall
        ⟨1, by decide⟩ := by
  have cursorZero : postNc.absorbed.val = 0 := bound.cursorZero
  have room : postNc.absorbed.val < rate := by
    rw [cursorZero]
    decide
  have marker :
      assignment 1713693 = 1 :=
    PiRlcChallenge.Transcript.Terminal.OutputDigestPins.catchupSqueeze_eq_one
      canonical one catchupAccepted
  unfold absorbElem
  rw [dif_pos room]
  apply stateExt
  · funext lane
    change
      overwriteLane postNc.lanes postNc.absorbed.val (wordField 1) lane =
        (callInputState assignment canonical
          PiRlcChallenge.Transcript.Terminal.OutputDigestSchedule.catchupCall
          ⟨1, by decide⟩).lanes lane
    by_cases laneZero : lane.val = 0
    · apply Fin.ext
      simp [overwriteLane, callInputState,
        PiRlcChallenge.Transcript.Terminal.OutputDigestSchedule.catchupCall,
      Poseidon2Call.Call.columnMap, wordField, fieldValue,
        u64Modulus, goldilocksP, cursorZero, laneZero, marker]
    · have overwritten :
          overwriteLane postNc.lanes postNc.absorbed.val
              (wordField 1) lane =
            postNc.lanes lane := by
        simp [overwriteLane, cursorZero, laneZero]
      rw [overwritten, bound.retainedLanes.lane lane laneZero]
      rcases laneValueCases lane with
        h | h | h | h | h | h | h | h
      · exact (laneZero h).elim
      all_goals
        simp [callInputState, tailColumn,
          PiRlcChallenge.Transcript.Terminal.OutputDigestSchedule.catchupCall,
          Poseidon2Call.Call.columnMap, h]
  · apply Fin.ext
    simp [callInputState, cursorZero]

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.PostNcBoundary
