import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound

/-!
Contract: the exact staged lane-digest rows recompute the absent-gamma
Poseidon2 digest after replacing `D_pre` with the decoded delayed suffix.

It does not own transcript challenge derivation or the final open-state muxes.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerStagedDigestRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer

abbrev DigestValues := Fin 4 → Nat

def inputValues (assignment : Nat → Nat) : List Nat :=
  rawArtifact.stagedDigestRecipe.inputColumns.map assignment

def expectedInputValues (assignment : Nat → Nat) : List Nat :=
  rawArtifact.stagedDigestExpectedInputs.map assignment

def computedDigest (assignment : Nat → Nat) : DigestValues :=
  fun lane => runValueRounds rawArtifact.stagedDigestRecipe.trace.rounds
    (inputValues assignment) (fun _ => 0) lane.val

def assignedDigest (assignment : Nat → Nat) : DigestValues :=
  fun lane => assignment
    (rawArtifact.stagedDigestOutputColumns.getD lane.val 0)

private theorem all_pieces_satisfied
    (assignment : Nat → Nat)
    (satisfied : rawArtifact.StagedDigestSatisfied assignment) :
    ∀ piece ∈ rawArtifact.stagedDigestPieces, Satisfies piece assignment := by
  apply (satisfies_flatten_iff rawArtifact.stagedDigestPieces assignment).mp
  simpa [RawArtifact.StagedDigestSatisfied, RawArtifact.stagedDigestRows] using
    satisfied

structure Sound (assignment : Nat → Nat) : Prop where
  constants :
    rawArtifact.stagedDigestRecipe.constantColumns.map assignment =
      rawArtifact.stagedDigestConstantValues
  inputOrder : inputValues assignment = expectedInputValues assignment
  decodedDPre :
    ((inputValues assignment).drop 30).take dPreWordCount =
      (List.range dPreWordCount).map fun index =>
        assignment (rawArtifact.dPreWordColumn index)
  hash : assignedDigest assignment = computedDigest assignment

theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.StagedDigestSatisfied assignment) :
    Sound assignment := by
  have pieces := all_pieces_satisfied assignment satisfied
  have constantsSatisfied :
      Satisfies (constantRows rawArtifact.stagedDigestRecipe) assignment :=
    pieces _ (by simp [RawArtifact.stagedDigestPieces])
  have traceSatisfied :
      Satisfies rawArtifact.stagedDigestRecipe.trace.rows assignment :=
    pieces _ (by simp [RawArtifact.stagedDigestPieces])
  refine {
    constants := constantRows_values rawArtifact.stagedDigestRecipe assignment
      canonical one rawArtifact_valid.stagedDigestConstantsCanonical
      constantsSatisfied
    inputOrder := ?_
    decodedDPre := ?_
    hash := ?_ }
  · simpa [inputValues, expectedInputValues] using
      congrArg (List.map assignment) rawArtifact_valid.stagedDigestInputOrder
  · simp only [inputValues, List.map_drop, List.map_take,
      staged_digest_dPre_columns, List.map_map]
    rfl
  · funext lane
    exact ownedTrace_values_sound staged_digest_trace_ownedValid canonical one
      traceSatisfied lane.val lane.isLt

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerStagedDigestRowSound
