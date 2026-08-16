import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayDigestDomain.Checkpoint2

/-! Artifact-checked third permutation checkpoint for claim-state domain framing. -/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
open Nightstream.Implementation.R1CS.Poseidon2ExtractedReference
open Nightstream.Implementation.R1CS.Poseidon2Permutation

def checkpoint3InputValues : List Nat :=
  domainBlock3 ++ checkpoint2Values.drop 4

def checkpoint3InputState : State :=
  stateFromValues checkpoint3InputValues ⟨4, by decide⟩

private theorem checkpoint3_input_exact :
    absorbWords (checkpointState checkpoint2Values) domainBlock3 =
      checkpoint3InputState := by
  apply stateView_injective
  native_decide

private theorem checkpoint3_inputs_agree :
    AgreeOn (sourceAssignment (stateLaneValues checkpoint3InputState))
      checkpoint3Assignment inputColumns := by
  intro column member
  simp only [inputColumns, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    native_decide

private theorem checkpoint3_output_values :
    ∀ lane : Fin width,
      checkpoint3Assignment (traceOutputColumn lane) =
        ((checkpointState checkpoint3Values).lanes lane).val := by
  native_decide

theorem checkpoint3_exact :
    permute (absorbWords (checkpointState checkpoint2Values) domainBlock3) =
      checkpointState checkpoint3Values := by
  rw [checkpoint3_input_exact]
  apply state_ext
    (certified_permutation_lane checkpoint3InputState
      (checkpointState checkpoint3Values) checkpoint3Assignment
      checkpoint3_inputs_agree checkpoint3_assignment_canonical
      checkpoint3_assignment_one checkpoint3_rows_satisfied
      checkpoint3_output_values)
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
