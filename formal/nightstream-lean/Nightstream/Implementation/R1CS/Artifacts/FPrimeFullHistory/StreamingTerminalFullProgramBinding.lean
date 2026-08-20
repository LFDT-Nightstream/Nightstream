import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullProgramBinding
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeCertificate

/-! Structural validation of the exact full-layout terminal program-binding artifact. -/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullProgramBinding

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipeCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProgramBinding.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullProgramBinding

structure FullValid : Prop where
  schemaVersion : rawArtifact.schemaVersion = 1
  profileId : rawArtifact.profileId =
    "nightstream/goldilocks/streaming-terminal-full-program-binding/v1"
  sourceIdentity : rawArtifact.sourceIdentity =
    "rust:nightstream/streaming-terminal-lifecycle/source-rows/v1"
  legacyShaNonAuthoritative : rawArtifact.sourceRowsSha256 = ""
  lifecycleScope : lifecycleScope = "recursive-terminal-arm-435"
  rowCount : rawArtifact.rowCount = totalRows
  sourceRowStart : rawArtifact.sourceRowStart = 352042
  finalRowStart : rawArtifact.finalRowStart = rawArtifact.sourceRowStart
  constantCount : rawArtifact.constantValues.length = constantFields
  constantsCanonical : ∀ value ∈ rawArtifact.constantValues,
    value < goldilocksP
  inputCount : rawArtifact.inputColumns.length = inputFields
  tagFirst : rawArtifact.inputColumns =
    (List.range' rawArtifact.constantStartColumn constantFields).drop 12 ++
      (List.range' rawArtifact.constantStartColumn constantFields).take 12
  hashOutputCount : rawArtifact.hashOutputColumns.length = digestFields
  carriedBindingCount : rawArtifact.carriedBindingColumns.length = digestFields
  equalityRowStart :
    rawArtifact.equalityRowStart = constantFields + traceRows
  rowStop :
    rawArtifact.equalityRowStart + equalityRowsCount = rawArtifact.rowCount

theorem rawArtifact_valid : FullValid := by
  refine {
    schemaVersion := rfl
    profileId := rfl
    sourceIdentity := rfl
    legacyShaNonAuthoritative := rfl
    lifecycleScope := rfl
    rowCount := by decide
    sourceRowStart := rfl
    finalRowStart := rfl
    constantCount := rfl
    constantsCanonical := by
      norm_num [rawArtifact, constantValues, goldilocksP]
    inputCount := rfl
    tagFirst := rfl
    hashOutputCount := rfl
    carriedBindingCount := rfl
    equalityRowStart := by decide
    rowStop := by decide }

theorem input_length :
    rawArtifact.hashRecipe.inputColumns.length = inputFields := by
  exact rawArtifact_valid.inputCount

theorem absorbRounds_exact :
    rawArtifact.hashRecipe.absorbRounds = absorbRounds := by
  norm_num [VariableHashRecipe.absorbRounds, input_length, rate,
    absorbRounds, inputFields]

theorem output_exact :
    rawArtifact.hashRecipe.outputColumns =
      (rawArtifact.hashRecipe.callOutputColumns
        rawArtifact.hashRecipe.absorbRounds).take 4 := by
  rw [absorbRounds_exact]
  rfl

theorem trace_ownedValid : rawArtifact.hashRecipe.trace.OwnedValid := by
  exact ownedValid rawArtifact.hashRecipe (by
      rw [absorbRounds_exact]
      norm_num [absorbRounds, inputFields, rate]) (by
      rw [input_length, absorbRounds_exact]
      norm_num [inputFields, absorbRounds, rate]) output_exact

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullProgramBinding
