import Nightstream.Implementation.R1CS.Artifacts.NebulaProgramBinding.Generated.NebulaProgramBindingPoseidon

/-!
Contract: exact-row soundness for the production Nebula base program binding.

Assurance tier: exact executable artifact correspondence. The theorems prove
that the carried binding is the output of the certified Poseidon2 sponge over
the generated input layout. They also prove the initial semantic-state and
memory links. They do not claim Poseidon2 collision resistance or cover the
complete recursive and terminal relations.
-/

namespace Nightstream.Implementation.R1CS.NebulaProgramBindingSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.NebulaProgramBinding
open Nightstream.Implementation.R1CS.NebulaProgramBindingPoseidon

private theorem linked_values_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies rows assignment)
    (outputs inputs : List Nat)
    (localRowStart : Nat)
    (exactRows : (rows.drop localRowStart).take 4 =
      (List.range 4).map fun lane =>
        builderLinearRow (outputs.getD lane 0)
          [(inputs.getD lane 0, 1)]) :
    ∀ lane, lane < 4 →
      assignment (outputs.getD lane 0) =
        assignment (inputs.getD lane 0) := by
  intro lane laneLt
  let row := builderLinearRow (outputs.getD lane 0)
    [(inputs.getD lane 0, 1)]
  have rowMember : row ∈
      (List.range 4).map fun index =>
        builderLinearRow (outputs.getD index 0)
          [(inputs.getD index 0, 1)] := by
    exact List.mem_map.mpr ⟨lane, List.mem_range.mpr laneLt, rfl⟩
  have inSlice : row ∈ (rows.drop localRowStart).take 4 := by
    rw [exactRows]
    exact rowMember
  have holds := satisfies row
    (List.mem_of_mem_drop (List.mem_of_mem_take inSlice))
  have linked := builderLinearRow_sound canonical constantOne
    (outputs.getD lane 0) [(inputs.getD lane 0, 1)]
    (by simp [CanonicalTerms, goldilocksP]) holds
  have inputCanonical := canonical (inputs.getD lane 0)
  have linkedMod :
      assignment (outputs.getD lane 0) =
        assignment (inputs.getD lane 0) % goldilocksP := by
    simpa [lcEval] using linked
  exact linkedMod.trans (Nat.mod_eq_of_lt inputCanonical)

structure Holds (assignment : Nat → Nat) : Prop where
  programBinding : ∀ lane, lane < 4 →
    assignment (carriedBindingColumns.getD lane 0) =
      runValueRounds trace.rounds
        (trace.inputColumns.map assignment) (fun _ => 0) lane
  semanticState : ∀ lane, lane < 4 →
    assignment (semanticStateColumns.getD lane 0) =
      assignment (initialSemanticStateColumns.getD lane 0)
  initialMemory : ∀ lane, lane < 4 →
    assignment (carriedMemoryColumns.getD lane 0) =
      assignment (initialMemoryDigestColumns.getD lane 0)

private theorem trace_rows_satisfied
    {assignment : Nat → Nat}
    (satisfies : Satisfies rows assignment) :
    Satisfies trace.rows assignment := by
  intro row member
  apply satisfies row
  have inSlice : row ∈
      (rows.drop rowStart).take traceRowCount := by
    rw [trace_rows_exact]
    exact member
  exact List.mem_of_mem_drop (List.mem_of_mem_take inSlice)

/-- The exact rows recompute the program binding with Poseidon2 and enforce
the two program-value links used by the base lane. -/
theorem program_binding_sound {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    Holds assignment := by
  have bindingLink := linked_values_sound canonical constantOne satisfies
    computedBindingColumns carriedBindingColumns bindingLinkRowStart
    binding_link_rows_exact
  have semanticLink := linked_values_sound canonical constantOne satisfies
    semanticStateColumns initialSemanticStateColumns semanticLinkRowStart
    semantic_link_rows_exact
  have memoryLink := linked_values_sound canonical constantOne satisfies
    carriedMemoryColumns initialMemoryDigestColumns memoryLinkRowStart
    memory_link_rows_exact
  have hashSound := trace_values_sound trace_valid canonical constantOne
    (trace_rows_satisfied satisfies)
  refine ⟨?_, semanticLink, memoryLink⟩
  intro lane laneLt
  have computedHash :
      assignment (computedBindingColumns.getD lane 0) =
        runValueRounds trace.rounds
          (trace.inputColumns.map assignment) (fun _ => 0) lane := by
    rw [← trace_output_layout]
    exact hashSound lane laneLt
  exact (bindingLink lane laneLt).symm.trans computedHash

end Nightstream.Implementation.R1CS.NebulaProgramBindingSound
