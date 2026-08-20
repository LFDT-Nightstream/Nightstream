import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalProfileSelection

/-!
Contract: exact terminal profile-selection rows force all three selected
schedule columns to one on the same canonical Goldilocks assignment.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProfileSelectionRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProfileSelection.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalProfileSelection

private theorem selector_row_sound
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    {column : Nat}
    (holds : RowHolds assignment (selectorRow column)) :
    assignment column = 1 := by
  have columnLt := canonical column
  simp only [selectorRow, RowHolds, lcEval, List.foldl, one, goldilocksP] at holds columnLt
  omega

/-- Satisfaction of the exact three Rust rows fixes the schedule, lifecycle,
and phase selectors to one. -/
theorem rows_imply_selectors_one
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment) :
    forall column, column ∈ rawArtifact.selectorColumns ->
      assignment column = 1 := by
  intro column member
  apply selector_row_sound canonical one
  exact satisfied _ (List.mem_map.mpr ⟨column, member, rfl⟩)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProfileSelectionRowSound
