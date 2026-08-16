import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayDigestDomain.Model
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayDigestDomain.Witness

/-!
Contract: generic refinement from one satisfying generated Poseidon2 witness
to one exact transcript-state lane.

The bridge uses all 600 generated rows, canonical residues, the constant-one
wire, and exact input columns. It does not own any checkpoint constants.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
open Nightstream.Implementation.R1CS.Poseidon2ExtractedReference
open Nightstream.Implementation.R1CS.Poseidon2Permutation
open Nightstream.Implementation.R1CS.Poseidon2PermutationSound

private theorem trace_output_member :
    ∀ lane : Fin width, traceOutputColumn lane ∈ outputColumns := by
  native_decide

/-- A satisfying exact artifact witness fixes each semantic permutation lane
when its nine source columns match the transcript state. -/
theorem certified_permutation_lane
    (input target : State)
    (assignment : Nat → Nat)
    (inputsAgree :
      AgreeOn (sourceAssignment (stateLaneValues input)) assignment
        inputColumns)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment)
    (outputsExact : ∀ lane : Fin width,
      assignment (traceOutputColumn lane) = (target.lanes lane).val)
    (lane : Fin width) :
    ((permute input).lanes lane).val = (target.lanes lane).val := by
  have outputKnown :
      traceOutputColumn lane ∈ knownAfter inputColumns definitions :=
    outputs_known _ (trace_output_member lane)
  have runAgreement := run_congr definitions_wellFormed inputsAgree
  have witnessAgreement := poseidon2Permutation_sound
    canonical one satisfies
  have artifactOutput :
      execution (stateLaneValues input) (traceOutputColumn lane) =
        assignment (traceOutputColumn lane) :=
    (runAgreement _ outputKnown).trans
      (witnessAgreement _ outputKnown)
  have extractedLanes := permute_lanes_eq_ssa input
  unfold ssaPermutationValues at extractedLanes
  have extractedFunctions := List.ofFn_injective extractedLanes
  exact (congrFun extractedFunctions lane).trans
    (artifactOutput.trans (outputsExact lane))

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
