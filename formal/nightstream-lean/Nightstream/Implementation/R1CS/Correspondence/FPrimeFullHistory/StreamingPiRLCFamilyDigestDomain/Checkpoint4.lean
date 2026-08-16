import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyDigestDomain.Model
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyDigestDomain.Witness

/-! Artifact-checked fourth permutation checkpoint for PiRLC family-state domain framing. -/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigestDomain

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
open Nightstream.Implementation.R1CS.Poseidon2ExtractedReference
open Nightstream.Implementation.R1CS.Poseidon2Permutation

def checkpoint4InputValues : List Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyDigestWitness.checkpoint4InputValues

def checkpoint4InputState : State :=
  stateFromValues checkpoint4InputValues ⟨4, by decide⟩

private theorem checkpoint4_input_values_exact :
    checkpoint4InputValues =
      domainBlock4 ++ checkpoint3Values.drop 4 := by
  native_decide

private theorem checkpoint4_input_exact :
    absorbWords (checkpointState checkpoint3Values) domainBlock4 =
      checkpoint4InputState := by
  apply stateView_injective
  native_decide

private theorem checkpoint4_inputs_agree :
    AgreeOn (sourceAssignment (stateLaneValues checkpoint4InputState))
      checkpoint4Assignment inputColumns := by
  intro column member
  simp only [inputColumns, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    native_decide

private theorem checkpoint4_output_values :
    ∀ lane : Fin width,
      checkpoint4Assignment (traceOutputColumn lane) =
        ((checkpointState checkpoint4Values).lanes lane).val := by
  native_decide

theorem checkpoint4_exact :
    permute (absorbWords (checkpointState checkpoint3Values) domainBlock4) =
      checkpointState checkpoint4Values := by
  rw [checkpoint4_input_exact]
  apply state_ext
    (certified_permutation_lane checkpoint4InputState
      (checkpointState checkpoint4Values) checkpoint4Assignment
      checkpoint4_inputs_agree checkpoint4_assignment_canonical
      checkpoint4_assignment_one checkpoint4_rows_satisfied
      checkpoint4_output_values)
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigestDomain
