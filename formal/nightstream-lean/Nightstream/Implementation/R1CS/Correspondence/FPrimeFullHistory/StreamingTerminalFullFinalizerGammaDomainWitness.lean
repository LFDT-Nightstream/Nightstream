import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingGammaDomainWitness
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaDomainModel

/-!
Contract: exact Rust-to-Lean witness certificate for the terminal Nebula gamma
application-domain Poseidon2 checkpoint.

Owns 24 bounded certificates for the 600 ordered SSA definitions, exact
coverage with no remainder, and the five capacity-lane outputs. It does not
own transcript framing, gamma challenges, output muxes, or lifecycle closure.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536
set_option maxHeartbeats 8000000

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaDomainWitness

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
open Nightstream.Implementation.R1CS.Poseidon2ExtractedReference
open Nightstream.Implementation.R1CS.Poseidon2Permutation
open Nightstream.Implementation.R1CS.Poseidon2PermutationSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingGammaDomainWitness
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaDomainModel

private def definitionChunk (offset : Nat) : List Definition :=
  (definitions.drop offset).take 25

private def definitionChunkAccepted (offset : Nat) : Bool :=
  (definitionChunk offset).all fun definition =>
    decide (Definition.Holds gammaCheckpointAssignment definition)

private theorem definitionChunk_holds {offset : Nat}
    (accepted : definitionChunkAccepted offset = true) :
    ∀ definition ∈ definitionChunk offset,
      Definition.Holds gammaCheckpointAssignment definition := by
  intro definition member
  exact of_decide_eq_true
    ((List.all_eq_true.mp accepted) definition member)

private theorem chunk00_holds :
    ∀ definition ∈ definitionChunk 0,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk01_holds :
    ∀ definition ∈ definitionChunk 25,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk02_holds :
    ∀ definition ∈ definitionChunk 50,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk03_holds :
    ∀ definition ∈ definitionChunk 75,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk04_holds :
    ∀ definition ∈ definitionChunk 100,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk05_holds :
    ∀ definition ∈ definitionChunk 125,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk06_holds :
    ∀ definition ∈ definitionChunk 150,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk07_holds :
    ∀ definition ∈ definitionChunk 175,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk08_holds :
    ∀ definition ∈ definitionChunk 200,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk09_holds :
    ∀ definition ∈ definitionChunk 225,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk10_holds :
    ∀ definition ∈ definitionChunk 250,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk11_holds :
    ∀ definition ∈ definitionChunk 275,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk12_holds :
    ∀ definition ∈ definitionChunk 300,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk13_holds :
    ∀ definition ∈ definitionChunk 325,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk14_holds :
    ∀ definition ∈ definitionChunk 350,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk15_holds :
    ∀ definition ∈ definitionChunk 375,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk16_holds :
    ∀ definition ∈ definitionChunk 400,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk17_holds :
    ∀ definition ∈ definitionChunk 425,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk18_holds :
    ∀ definition ∈ definitionChunk 450,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk19_holds :
    ∀ definition ∈ definitionChunk 475,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk20_holds :
    ∀ definition ∈ definitionChunk 500,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk21_holds :
    ∀ definition ∈ definitionChunk 525,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk22_holds :
    ∀ definition ∈ definitionChunk 550,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem chunk23_holds :
    ∀ definition ∈ definitionChunk 575,
      Definition.Holds gammaCheckpointAssignment definition :=
  definitionChunk_holds (by rfl)

private theorem definitions_holds :
    ∀ definition ∈ definitions,
      Definition.Holds gammaCheckpointAssignment definition := by
  intro definition member
  rw [← List.take_append_drop 25 definitions] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk00_holds definition member
  rw [← List.drop_take_append_drop definitions 25 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk01_holds definition member
  rw [← List.drop_take_append_drop definitions 50 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk02_holds definition member
  rw [← List.drop_take_append_drop definitions 75 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk03_holds definition member
  rw [← List.drop_take_append_drop definitions 100 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk04_holds definition member
  rw [← List.drop_take_append_drop definitions 125 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk05_holds definition member
  rw [← List.drop_take_append_drop definitions 150 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk06_holds definition member
  rw [← List.drop_take_append_drop definitions 175 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk07_holds definition member
  rw [← List.drop_take_append_drop definitions 200 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk08_holds definition member
  rw [← List.drop_take_append_drop definitions 225 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk09_holds definition member
  rw [← List.drop_take_append_drop definitions 250 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk10_holds definition member
  rw [← List.drop_take_append_drop definitions 275 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk11_holds definition member
  rw [← List.drop_take_append_drop definitions 300 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk12_holds definition member
  rw [← List.drop_take_append_drop definitions 325 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk13_holds definition member
  rw [← List.drop_take_append_drop definitions 350 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk14_holds definition member
  rw [← List.drop_take_append_drop definitions 375 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk15_holds definition member
  rw [← List.drop_take_append_drop definitions 400 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk16_holds definition member
  rw [← List.drop_take_append_drop definitions 425 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk17_holds definition member
  rw [← List.drop_take_append_drop definitions 450 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk18_holds definition member
  rw [← List.drop_take_append_drop definitions 475 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk19_holds definition member
  rw [← List.drop_take_append_drop definitions 500 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk20_holds definition member
  rw [← List.drop_take_append_drop definitions 525 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk21_holds definition member
  rw [← List.drop_take_append_drop definitions 550 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk22_holds definition member
  rw [← List.drop_take_append_drop definitions 575 25] at member
  rcases List.mem_append.mp member with member | member
  · exact chunk23_holds definition member
  have noRemainder : definitions.drop 600 = [] :=
    List.drop_eq_nil_iff.mpr (by rw [definitions_length, rowCount])
  rw [noRemainder] at member
  exact False.elim (by simpa using member)

private theorem inputs_agree :
    AgreeOn (sourceAssignment (stateLaneValues checkpoint3InputState))
      gammaCheckpointAssignment inputColumns := by
  intro column member
  simp only [inputColumns, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    rfl

private theorem execution_agrees :
    AgreeOn (execution (stateLaneValues checkpoint3InputState))
      gammaCheckpointAssignment (knownAfter inputColumns definitions) := by
  exact run_agrees_of_holds definitions_wellFormed inputs_agree definitions_holds

private theorem trace_output_member :
    ∀ lane : Fin width, traceOutputColumn lane ∈ outputColumns := by
  intro lane
  fin_cases lane <;> decide

private theorem checkpoint3_lane_exact (lane : Fin width) :
    ((permute checkpoint3InputState).lanes lane).val =
      gammaCheckpointAssignment (traceOutputColumn lane) := by
  have semanticOutputs := permute_lanes_eq_ssa checkpoint3InputState
  unfold ssaPermutationValues at semanticOutputs
  have semanticLane := congrFun (List.ofFn_injective semanticOutputs) lane
  exact semanticLane.trans
    (execution_agrees _ (outputs_known _ (trace_output_member lane)))

/-- The exact Rust witness fixes all five capacity lanes of the third
application-domain checkpoint. -/
theorem checkpoint3_capacity_exact :
    ((permute checkpoint3InputState).lanes ⟨3, by decide⟩).val =
        17411973590883579087 ∧
      ((permute checkpoint3InputState).lanes ⟨4, by decide⟩).val =
        6939038333896971149 ∧
      ((permute checkpoint3InputState).lanes ⟨5, by decide⟩).val =
        3171679524884682263 ∧
      ((permute checkpoint3InputState).lanes ⟨6, by decide⟩).val =
        2890321166649729893 ∧
      ((permute checkpoint3InputState).lanes ⟨7, by decide⟩).val =
        13044081322747540714 := by
  constructor
  · exact (checkpoint3_lane_exact ⟨3, by decide⟩).trans (by rfl)
  constructor
  · exact (checkpoint3_lane_exact ⟨4, by decide⟩).trans (by rfl)
  constructor
  · exact (checkpoint3_lane_exact ⟨5, by decide⟩).trans (by rfl)
  constructor
  · exact (checkpoint3_lane_exact ⟨6, by decide⟩).trans (by rfl)
  · exact (checkpoint3_lane_exact ⟨7, by decide⟩).trans (by rfl)

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaDomainWitness
