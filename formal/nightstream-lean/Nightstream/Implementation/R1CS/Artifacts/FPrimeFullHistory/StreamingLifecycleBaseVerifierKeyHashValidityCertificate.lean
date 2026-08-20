import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleBaseVerifierKey
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeCertificate

/-!
Contract: leaf structural certificate for the three exact Rust-emitted base
verifier-key core Poseidon2 recipes.

Owns their source ranges, input lengths, absorb counts, total round counts,
four-lane outputs, and structural sponge validity. It does not unfold the
source or final run schedules and does not prove semantic input authority.

Assurance tier: artifact-checked for
`FPRIME-STREAMING-LIFECYCLE-BASE-VERIFIER-KEY-PROVENANCE-V1`,
Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleBaseVerifierKey

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipeCertificate
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleBaseVerifierKey

theorem baseHash_sourceRows_exact :
    rawArtifact.baseVerifierKeyHash.sourceRows =
      { start := 36133, stop := 42793 } := rfl

theorem baseHash_inputLength_exact :
    rawArtifact.baseVerifierKeyHash.recipe.inputColumns.length = 37 := rfl

theorem baseHash_absorbRounds_exact :
    rawArtifact.baseVerifierKeyHash.recipe.absorbRounds = 10 := by
  rw [VariableHashRecipe.absorbRounds, baseHash_inputLength_exact]
  norm_num [rate]

theorem baseHash_roundCount_exact :
    rawArtifact.baseVerifierKeyHash.recipe.trace.rounds.length = 11 := by
  simp [VariableHashRecipe.trace, VariableHashRecipe.rounds,
    baseHash_absorbRounds_exact]

theorem baseHash_output_exact :
    rawArtifact.baseVerifierKeyHash.recipe.outputColumns =
      (rawArtifact.baseVerifierKeyHash.recipe.callOutputColumns
        rawArtifact.baseVerifierKeyHash.recipe.absorbRounds).take 4 := by
  rw [baseHash_absorbRounds_exact]
  rfl

theorem baseHash_trace_ownedValid :
    rawArtifact.baseVerifierKeyHash.recipe.trace.OwnedValid := by
  exact ownedValid rawArtifact.baseVerifierKeyHash.recipe (by
      rw [baseHash_absorbRounds_exact]
      omega) (by
      rw [baseHash_inputLength_exact, baseHash_absorbRounds_exact]
      omega) baseHash_output_exact

theorem policyHash_sourceRows_exact :
    rawArtifact.policyVerifierKeyHash.sourceRows =
      { start := 42793, stop := 45817 } := rfl

theorem policyHash_inputLength_exact :
    rawArtifact.policyVerifierKeyHash.recipe.inputColumns.length = 13 := rfl

theorem policyHash_absorbRounds_exact :
    rawArtifact.policyVerifierKeyHash.recipe.absorbRounds = 4 := by
  rw [VariableHashRecipe.absorbRounds, policyHash_inputLength_exact]
  norm_num [rate]

theorem policyHash_roundCount_exact :
    rawArtifact.policyVerifierKeyHash.recipe.trace.rounds.length = 5 := by
  simp [VariableHashRecipe.trace, VariableHashRecipe.rounds,
    policyHash_absorbRounds_exact]

theorem policyHash_output_exact :
    rawArtifact.policyVerifierKeyHash.recipe.outputColumns =
      (rawArtifact.policyVerifierKeyHash.recipe.callOutputColumns
        rawArtifact.policyVerifierKeyHash.recipe.absorbRounds).take 4 := by
  rw [policyHash_absorbRounds_exact]
  rfl

theorem policyHash_trace_ownedValid :
    rawArtifact.policyVerifierKeyHash.recipe.trace.OwnedValid := by
  exact ownedValid rawArtifact.policyVerifierKeyHash.recipe (by
      rw [policyHash_absorbRounds_exact]
      omega) (by
      rw [policyHash_inputLength_exact, policyHash_absorbRounds_exact]
      omega) policyHash_output_exact

theorem initialBoundaryHash_sourceRows_exact :
    rawArtifact.initialBoundaryHash.sourceRows =
      { start := 45821, stop := 48243 } := rfl

theorem initialBoundaryHash_inputLength_exact :
    rawArtifact.initialBoundaryHash.recipe.inputColumns.length = 12 := rfl

theorem initialBoundaryHash_absorbRounds_exact :
    rawArtifact.initialBoundaryHash.recipe.absorbRounds = 3 := by
  rw [VariableHashRecipe.absorbRounds,
    initialBoundaryHash_inputLength_exact]
  norm_num [rate]

theorem initialBoundaryHash_roundCount_exact :
    rawArtifact.initialBoundaryHash.recipe.trace.rounds.length = 4 := by
  simp [VariableHashRecipe.trace, VariableHashRecipe.rounds,
    initialBoundaryHash_absorbRounds_exact]

theorem initialBoundaryHash_output_exact :
    rawArtifact.initialBoundaryHash.recipe.outputColumns =
      (rawArtifact.initialBoundaryHash.recipe.callOutputColumns
        rawArtifact.initialBoundaryHash.recipe.absorbRounds).take 4 := by
  rw [initialBoundaryHash_absorbRounds_exact]
  rfl

theorem initialBoundaryHash_trace_ownedValid :
    rawArtifact.initialBoundaryHash.recipe.trace.OwnedValid := by
  exact ownedValid rawArtifact.initialBoundaryHash.recipe (by
      rw [initialBoundaryHash_absorbRounds_exact]
      omega) (by
      rw [initialBoundaryHash_inputLength_exact,
        initialBoundaryHash_absorbRounds_exact]) initialBoundaryHash_output_exact

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleBaseVerifierKey
