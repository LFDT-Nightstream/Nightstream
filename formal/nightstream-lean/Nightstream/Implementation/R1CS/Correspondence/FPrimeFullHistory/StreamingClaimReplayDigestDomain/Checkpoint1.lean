import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayDigestDomain.WitnessBridge

/-! Artifact-checked first permutation checkpoint for claim-state domain framing. -/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
open Nightstream.Implementation.R1CS.Poseidon2ExtractedReference
open Nightstream.Implementation.R1CS.Poseidon2Permutation

def checkpoint1InputValues : List Nat :=
  domainBlock1 ++ [0, 0, 0, 0]

def checkpoint1InputState : State :=
  stateFromValues checkpoint1InputValues ⟨4, by decide⟩

private theorem checkpoint1_input_exact :
    absorbWords emptyState domainBlock1 = checkpoint1InputState := by
  apply stateView_injective
  native_decide

private theorem checkpoint1_inputs_agree :
    AgreeOn (sourceAssignment (stateLaneValues checkpoint1InputState))
      checkpoint1Assignment inputColumns := by
  intro column member
  simp only [inputColumns, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    native_decide

private theorem checkpoint1_output_values :
    ∀ lane : Fin width,
      checkpoint1Assignment (traceOutputColumn lane) =
        ((checkpointState checkpoint1Values).lanes lane).val := by
  native_decide

theorem checkpoint1_exact :
    permute (absorbWords emptyState domainBlock1) =
      checkpointState checkpoint1Values := by
  rw [checkpoint1_input_exact]
  apply state_ext
    (certified_permutation_lane checkpoint1InputState
      (checkpointState checkpoint1Values) checkpoint1Assignment
      checkpoint1_inputs_agree checkpoint1_assignment_canonical
      checkpoint1_assignment_one checkpoint1_rows_satisfied
      checkpoint1_output_values)
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
