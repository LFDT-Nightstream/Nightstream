import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.PostNcBoundary
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.DigestRounds

/-!
Final terminal-NC permutation and retained-state boundary.

Assurance tier: conditional implementation/R1CS refinement.

Owns: proof that the final squeeze's eight output wires are exactly columns
`1692820..1692827`; and reduction of the post-NC boundary to one explicit
final-permutation input refinement.

Does not own: final-round artifact selection or call acceptance; proof that
exact semantic NC replay reaches the artifact final call input; serialization
of the final round message; the 30 final SumCheck equations;
native/gadget/Rust conformance; costs; necessity; or row removal.

Emits constraints: no.

Authority boundary: `FinalPermutationBound` is deliberately the sole remaining
conditional bridge in this module. It requires the semantic post-NC state to
be the pure permutation of exactly the artifact final-call input. Accepted
rows then prove that same permutation produced the artifact output columns.
No output digest or carried post-state is accepted as authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.14.challenge.call` | exact call and independent acceptance | imported artifact boundary | `FinalRound.Artifact` |
| `nifs.pi_ccs.nc_sumcheck.round.14.challenge.outputs` | output lane `i` is column `1692820+i` | derived structure | `finalSqueeze_outputColumn` |
| `nifs.pi_ccs.nc_sumcheck.round.14.challenge.input` | semantic final permutation input equals the exact artifact call input | explicit remaining bridge | `FinalPermutationBound` |
| `nifs.pi_ccs.nc.post_state.cursor` | the final pure permutation computes cursor zero | computed | `FinalPermutationBound.cursorZero` |
| `nifs.pi_ccs.nc.post_state.retained_lanes` | accepted final call binds lanes one through seven | conditional refinement | `retainedLanes_of_accepted` |
| `nifs.pi_ccs.nc.post_state` | compose computed cursor and retained lanes | conditional refinement | `boundary_of_accepted` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalState

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.DigestRounds
open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact

/-- The final call's eight SSA output lanes are exactly the post-NC tail
columns consumed by the catch-up boundary. -/
theorem finalSqueeze_outputColumn (lane : Fin width) :
    finalSqueezeCall.columnMap (601 + lane.val) =
      PostNcBoundary.tailColumn lane := by
  rw [LaterRound.Artifact.squeezeOutputColumn finalLaterRound lane]
  rw [finalSqueezeOutputBase_eq]
  rfl

/-- Exact remaining bridge between semantic NC replay and the final artifact
call. The input cursor is four because the squeeze marker fills the rate
before the final permutation. -/
def FinalPermutationBound
    (postNc : State)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) : Prop :=
  ∃ permutationInput : State,
    permutationInput =
      callInputState assignment canonical finalSqueezeCall
        ⟨rate, by decide⟩ ∧
    postNc = permute permutationInput

/-- A final permutation execution computes the post-NC cursor; no artifact
column is needed for this fact. -/
theorem FinalPermutationBound.cursorZero
    {postNc : State}
    {assignment : Nat -> Nat}
    {canonical : forall column, assignment column < goldilocksP}
    (bound : FinalPermutationBound postNc assignment canonical) :
    PostNcBoundary.CursorZero postNc := by
  rcases bound with ⟨permutationInput, _inputBound, outputBound⟩
  change postNc.absorbed.val = 0
  rw [outputBound]
  rfl

/-- Accepted final-call rows plus the exact input refinement bind all retained
post-NC lanes. Lane zero is included in the derived call equality but omitted
from the exported boundary because catch-up overwrites it. -/
theorem retainedLanes_of_accepted
    {postNc : State}
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (bound : FinalPermutationBound postNc assignment canonical) :
    PostNcBoundary.RetainedLanesBound postNc assignment canonical := by
  rcases bound with ⟨permutationInput, inputBound, outputBound⟩
  have callAccepted := finalSqueezeCallAccepted accepted
  have callRefinement :=
    callAccepted_permute canonical one finalSqueezeCall
      ⟨rate, by decide⟩ callAccepted
  have outputEq :
      postNc =
        callOutputState assignment canonical finalSqueezeCall := by
    rw [outputBound, inputBound, callRefinement]
  constructor
  intro lane _laneNonzero
  rw [outputEq]
  change
    fieldAt assignment canonical
        (finalSqueezeCall.columnMap (601 + lane.val)) =
      fieldAt assignment canonical (PostNcBoundary.tailColumn lane)
  rw [finalSqueeze_outputColumn]

/-- The exact final-call bridge and accepted rows construct the complete
minimal post-NC boundary consumed by catch-up. -/
theorem boundary_of_accepted
    {postNc : State}
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment)
    (bound : FinalPermutationBound postNc assignment canonical) :
    PostNcBoundary.Bound postNc assignment canonical where
  cursorZero := bound.cursorZero
  retainedLanes :=
    retainedLanes_of_accepted canonical one accepted bound

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalState
