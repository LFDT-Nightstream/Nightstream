import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafCompressionRowSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound

/-!
Contract: exact row soundness for the Poseidon2 envelope of the terminal
`ops` leaf.

The verifier-owned ten-field domain prefix and all 54 rank-one compression
outputs form the exact 64-field preimage. Satisfying rows recompute the four
declared digest outputs through the complete compact Poseidon2 trace.

The digest is an output of checked rows, never an authority premise. This
module does not own sampler liveness, collision resistance, Module-SIS
security, or lifecycle closure.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafEnvelopeRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipeCertificate
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer

abbrev recipe : VariableHashRecipe := rawArtifact.opsLeaf.envelopeRecipe

theorem input_length : recipe.inputColumns.length = 64 := by
  rfl

theorem absorbRounds_exact : recipe.absorbRounds = 16 := by
  norm_num [VariableHashRecipe.absorbRounds, input_length, rate]

theorem output_exact :
    recipe.outputColumns =
      (recipe.callOutputColumns recipe.absorbRounds).take 4 := by
  rw [absorbRounds_exact]
  rfl

theorem trace_ownedValid : recipe.trace.OwnedValid := by
  exact ownedValid recipe (by rw [absorbRounds_exact]; decide) (by
    rw [input_length, absorbRounds_exact]) output_exact

theorem constants_canonical :
    ∀ value ∈ recipe.constantValues, value < goldilocksP := by
  norm_num [recipe, LeafHashArtifact.envelopeRecipe, rawArtifact, opsLeaf,
    goldilocksP]

def opsEnvelopeRows : List Row :=
  constantRows recipe ++ recipe.trace.rows

def OpsEnvelopeSatisfied (assignment : Nat → Nat) : Prop :=
  Satisfies opsEnvelopeRows assignment

private theorem constants_satisfied
    (assignment : Nat → Nat)
    (satisfied : OpsEnvelopeSatisfied assignment) :
    Satisfies (constantRows recipe) assignment := by
  intro row member
  exact satisfied row (List.mem_append_left _ member)

private theorem trace_satisfied
    (assignment : Nat → Nat)
    (satisfied : OpsEnvelopeSatisfied assignment) :
    Satisfies recipe.trace.rows assignment := by
  intro row member
  exact satisfied row (List.mem_append_right _ member)

abbrev DigestValues := Fin 4 → Nat

def computedDigest (assignment : Nat → Nat) : DigestValues :=
  fun lane => runValueRounds recipe.trace.rounds
    (recipe.inputColumns.map assignment) (fun _ => 0) lane.val

def assignedDigest (assignment : Nat → Nat) : DigestValues :=
  fun lane => assignment (recipe.outputColumns.getD lane.val 0)

structure Sound (assignment : Nat → Nat) : Prop where
  constants :
    recipe.constantColumns.map assignment = recipe.constantValues
  inputOrder :
    recipe.inputColumns =
      recipe.constantColumns ++
        rawArtifact.opsLeaf.compression.block.outputColumns
  compressionInputs :
    (recipe.inputColumns.drop leafEnvelopeConstantFields).map assignment =
      rawArtifact.opsLeaf.compression.block.outputColumns.map assignment
  hash : assignedDigest assignment = computedDigest assignment

theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : OpsEnvelopeSatisfied assignment) :
    Sound assignment := by
  have constantsRows := constants_satisfied assignment satisfied
  have traceRows := trace_satisfied assignment satisfied
  refine {
    constants := constantRows_values recipe assignment canonical one
      constants_canonical constantsRows
    inputOrder := ?_
    compressionInputs := ?_
    hash := ?_ }
  · exact rawArtifact_valid.opsLeafDigestInputs
  · change
      (rawArtifact.opsLeaf.digestInputColumns.drop
        leafEnvelopeConstantFields).map assignment = _
    rw [rawArtifact_valid.opsLeafDigestInputs]
    simp [leafEnvelopeConstantFields]
  · funext lane
    exact ownedTrace_values_sound trace_ownedValid canonical one traceRows
      lane.val lane.isLt

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafEnvelopeRowSound
