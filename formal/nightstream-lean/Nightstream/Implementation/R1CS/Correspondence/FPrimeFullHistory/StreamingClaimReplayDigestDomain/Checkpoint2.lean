import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayDigestDomain.Checkpoint1

/-! Artifact-checked second permutation checkpoint for claim-state domain framing. -/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
open Nightstream.Implementation.R1CS.Poseidon2ExtractedReference
open Nightstream.Implementation.R1CS.Poseidon2Permutation

def checkpoint2InputValues : List Nat :=
  domainBlock2 ++ checkpoint1Values.drop 4

def checkpoint2InputState : State :=
  stateFromValues checkpoint2InputValues ⟨4, by decide⟩

private theorem checkpoint2_input_exact :
    absorbWords (checkpointState checkpoint1Values) domainBlock2 =
      checkpoint2InputState := by
  apply stateView_injective
  native_decide

private theorem checkpoint2_inputs_agree :
    AgreeOn (sourceAssignment (stateLaneValues checkpoint2InputState))
      checkpoint2Assignment inputColumns := by
  intro column member
  simp only [inputColumns, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    native_decide

private theorem checkpoint2_output_values :
    ∀ lane : Fin width,
      checkpoint2Assignment (traceOutputColumn lane) =
        ((checkpointState checkpoint2Values).lanes lane).val := by
  native_decide

theorem checkpoint2_exact :
    permute (absorbWords (checkpointState checkpoint1Values) domainBlock2) =
      checkpointState checkpoint2Values := by
  rw [checkpoint2_input_exact]
  apply state_ext
    (certified_permutation_lane checkpoint2InputState
      (checkpointState checkpoint2Values) checkpoint2Assignment
      checkpoint2_inputs_agree checkpoint2_assignment_canonical
      checkpoint2_assignment_one checkpoint2_rows_satisfied
      checkpoint2_output_values)
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
