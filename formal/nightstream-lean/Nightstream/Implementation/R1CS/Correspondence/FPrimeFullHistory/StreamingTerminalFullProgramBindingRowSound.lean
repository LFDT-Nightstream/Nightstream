import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullProgramBinding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound

/-!
Contract: all exact full-layout terminal program-binding rows recompute the
Poseidon2 binding from the tag-first verifier configuration and copy it to the
carried Nebula lane.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullProgramBindingRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProgramBinding.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullProgramBinding
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullProgramBinding

abbrev DigestValues := Fin 4 → Nat

def inputValues (assignment : Nat → Nat) : List Nat :=
  rawArtifact.hashRecipe.inputColumns.map assignment

def computedDigest (assignment : Nat → Nat) : DigestValues :=
  fun lane => runValueRounds rawArtifact.hashRecipe.trace.rounds
    (inputValues assignment) (fun _ => 0) lane.val

def assignedDigest (assignment : Nat → Nat) : DigestValues :=
  fun lane => assignment (rawArtifact.hashOutputColumns.getD lane.val 0)

def carriedDigest (assignment : Nat → Nat) : DigestValues :=
  fun lane => assignment (rawArtifact.carriedBindingColumns.getD lane.val 0)

private theorem all_pieces_satisfied
    (assignment : Nat → Nat)
    (satisfied : rawArtifact.Satisfied assignment) :
    ∀ piece ∈ rawArtifact.programPieces, Satisfies piece assignment := by
  apply (satisfies_flatten_iff rawArtifact.programPieces assignment).mp
  simpa [RawArtifact.Satisfied, RawArtifact.program] using satisfied

structure Sound (assignment : Nat → Nat) : Prop where
  constants : rawArtifact.hashRecipe.constantColumns.map assignment =
    rawArtifact.constantValues
  hash : assignedDigest assignment = computedDigest assignment
  carriedLink : carriedDigest assignment = assignedDigest assignment

theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment) :
    Sound assignment := by
  have pieces := all_pieces_satisfied assignment satisfied
  have constantsSatisfied :
      Satisfies (constantRows rawArtifact.hashRecipe) assignment :=
    pieces _ (by simp [RawArtifact.programPieces])
  have traceSatisfied :
      Satisfies rawArtifact.hashRecipe.trace.rows assignment :=
    pieces _ (by simp [RawArtifact.programPieces])
  have equalitySatisfied : Satisfies rawArtifact.equalityRows assignment :=
    pieces _ (by simp [RawArtifact.programPieces])
  refine {
    constants := constantRows_values rawArtifact.hashRecipe assignment
      canonical one rawArtifact_valid.constantsCanonical constantsSatisfied
    hash := ?_
    carriedLink := ?_ }
  · funext lane
    exact ownedTrace_values_sound trace_ownedValid canonical one traceSatisfied
      lane.val lane.isLt
  · funext lane
    have laneMember : lane.val ∈ List.range digestFields := by
      simp [digestFields, lane.isLt]
    have linkHolds := equalitySatisfied
      (builderLinearRow (rawArtifact.carriedBindingColumns.getD lane.val 0)
        [(rawArtifact.hashOutputColumns.getD lane.val 0, 1)])
      (List.mem_map.mpr ⟨lane.val, laneMember, by
        simp [RawArtifact.equalityRows]⟩)
    have exact := builderLinearRow_sound canonical one
      (rawArtifact.carriedBindingColumns.getD lane.val 0)
      [(rawArtifact.hashOutputColumns.getD lane.val 0, 1)]
      (by simp [CanonicalTerms, goldilocksP]) linkHolds
    have sourceCanonical :=
      canonical (rawArtifact.hashOutputColumns.getD lane.val 0)
    simp only [lcEval, List.foldl, Nat.zero_add, Nat.one_mul] at exact
    rw [Nat.mod_eq_of_lt sourceCanonical] at exact
    simpa [carriedDigest, assignedDigest] using exact

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullProgramBindingRowSound
