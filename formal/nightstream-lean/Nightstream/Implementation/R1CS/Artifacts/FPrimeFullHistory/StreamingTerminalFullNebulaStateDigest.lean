import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullNebulaStateDigest
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeCertificate

/-! Structural validation of the exact full-layout terminal Nebula-state artifact. -/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullNebulaStateDigest

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipeCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullNebulaStateDigest

structure FullValid : Prop where
  schemaVersion : rawArtifact.schemaVersion = 2
  profileId : rawArtifact.profileId =
    "nightstream/goldilocks/streaming-terminal-full-nebula-state-digest/v1"
  sourceIdentity : rawArtifact.sourceIdentity =
    "rust:nightstream/streaming-terminal-lifecycle/source-rows/v1"
  legacyShaNonAuthoritative : rawArtifact.sourceRowsSha256 = ""
  lifecycleScope : lifecycleScope = "recursive-terminal-arm-435"
  rowCount : rawArtifact.rowCount = familyRows
  sourceRowStart : rawArtifact.sourceRowStart = 332689
  finalRowStart : rawArtifact.finalRowStart = rawArtifact.sourceRowStart
  absentConstantCount :
    rawArtifact.absentConstantValues.length = absentConstantFields
  absentConstantsCanonical : ∀ value ∈ rawArtifact.absentConstantValues,
    value < goldilocksP
  absentInputCount : rawArtifact.absentInputColumns.length = absentInputFields
  absentOutputCount : rawArtifact.absentOutputColumns.length = digestFields
  presentConstantCount :
    rawArtifact.presentConstantValues.length = presentConstantFields
  presentConstantsCanonical : ∀ value ∈ rawArtifact.presentConstantValues,
    value < goldilocksP
  presentInputCount : rawArtifact.presentInputColumns.length = presentInputFields
  presentOutputCount : rawArtifact.presentOutputColumns.length = digestFields
  hashOutputCount : rawArtifact.hashOutputColumns.length = digestFields
  xOutStateCount : rawArtifact.xOutStateColumns.length = digestFields
  absentRowStart : rawArtifact.absentRowStart = bitRowCount
  presentRowStart :
    rawArtifact.presentRowStart = rawArtifact.absentRowStart + absentHashRows
  muxRowStart :
    rawArtifact.muxRowStart = rawArtifact.presentRowStart + presentHashRows
  equalityRowStart :
    rawArtifact.equalityRowStart = rawArtifact.muxRowStart + muxRowCount
  rowStop :
    rawArtifact.equalityRowStart + equalityRowCount = rawArtifact.rowCount
  selectedSourceRow :
    rawArtifact.selectedSourceRow =
      rawArtifact.sourceRowStart + rawArtifact.equalityRowStart

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
    absentConstantCount := rfl
    absentConstantsCanonical := by
      norm_num [rawArtifact, absentConstantValues, goldilocksP]
    absentInputCount := rfl
    absentOutputCount := rfl
    presentConstantCount := rfl
    presentConstantsCanonical := by
      norm_num [rawArtifact, presentConstantValues, goldilocksP]
    presentInputCount := rfl
    presentOutputCount := rfl
    hashOutputCount := rfl
    xOutStateCount := rfl
    absentRowStart := rfl
    presentRowStart := by decide
    muxRowStart := by decide
    equalityRowStart := by decide
    rowStop := by decide
    selectedSourceRow := by decide }

theorem absent_input_length :
    rawArtifact.absentRecipe.inputColumns.length = absentInputFields := by
  norm_num [VariableHashRecipe.inputColumns, RawArtifact.absentRecipe,
    rawArtifact, absentInputFields]

theorem absent_absorbRounds_exact :
    rawArtifact.absentRecipe.absorbRounds = 15 := by
  norm_num [VariableHashRecipe.absorbRounds, absent_input_length, rate,
    absentInputFields]

theorem absent_output_exact :
    rawArtifact.absentRecipe.outputColumns =
      (rawArtifact.absentRecipe.callOutputColumns
        rawArtifact.absentRecipe.absorbRounds).take 4 := by
  rw [absent_absorbRounds_exact]
  rfl

theorem absent_trace_ownedValid :
    rawArtifact.absentRecipe.trace.OwnedValid := by
  exact ownedValid rawArtifact.absentRecipe (by
      rw [absent_absorbRounds_exact]
      norm_num) (by
      rw [absent_input_length, absent_absorbRounds_exact]
      norm_num [absentInputFields]) absent_output_exact

theorem present_input_length :
    rawArtifact.presentRecipe.inputColumns.length = presentInputFields := by
  norm_num [VariableHashRecipe.inputColumns, RawArtifact.presentRecipe,
    rawArtifact, presentInputFields]

theorem present_absorbRounds_exact :
    rawArtifact.presentRecipe.absorbRounds = 15 := by
  norm_num [VariableHashRecipe.absorbRounds, present_input_length, rate,
    presentInputFields]

theorem present_output_exact :
    rawArtifact.presentRecipe.outputColumns =
      (rawArtifact.presentRecipe.callOutputColumns
        rawArtifact.presentRecipe.absorbRounds).take 4 := by
  rw [present_absorbRounds_exact]
  rfl

theorem present_trace_ownedValid :
    rawArtifact.presentRecipe.trace.OwnedValid := by
  exact ownedValid rawArtifact.presentRecipe (by
      rw [present_absorbRounds_exact]
      norm_num) (by
      rw [present_input_length, present_absorbRounds_exact]
      norm_num [presentInputFields]) present_output_exact

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullNebulaStateDigest
