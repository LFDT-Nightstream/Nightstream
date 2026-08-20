import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullPhaseSemantic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeCertificate

/-! Structural validation of the exact full-layout terminal phase-semantic artifact. -/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullPhaseSemantic

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Call
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipeCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullPhaseSemantic

structure FullValid : Prop where
  schemaVersion : rawArtifact.schemaVersion = 2
  profileId : rawArtifact.profileId =
    "nightstream/goldilocks/streaming-terminal-full-phase-semantic/v1"
  sourceIdentity : rawArtifact.sourceIdentity =
    "rust:nightstream/streaming-terminal-lifecycle/source-rows/v1"
  legacyShaNonAuthoritative : rawArtifact.sourceRowsSha256 = ""
  lifecycleScope : lifecycleScope = "recursive-terminal-arm-435"
  rowCount : rawArtifact.rowCount = totalRows
  sourceRowStart : rawArtifact.sourceRowStart = 2288
  finalRowStart : rawArtifact.finalRowStart = rawArtifact.sourceRowStart
  constants : rawArtifact.constantValues = expectedConstantValues
  constantsCanonical : ∀ value ∈ rawArtifact.constantValues,
    value < goldilocksP
  localCount : rawArtifact.localColumns.length = digestFields
  payloadCount : rawArtifact.payloadColumns.length = payloadFields
  hashOutputCount : rawArtifact.hashOutputColumns.length = digestFields
  xOutSemanticCount : rawArtifact.xOutSemanticColumns.length = digestFields
  equalityRowStart : rawArtifact.equalityRowStart = hashTotalRows

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
    constants := rfl
    constantsCanonical := by
      norm_num [rawArtifact, phaseConstantValues, goldilocksP]
    localCount := rfl
    payloadCount := by simp [rawArtifact, payloadFields]
    hashOutputCount := rfl
    xOutSemanticCount := rfl
    equalityRowStart := by decide }

theorem input_length :
    rawArtifact.hashRecipe.inputColumns.length = hashInputFields := by
  norm_num [VariableHashRecipe.inputColumns, VariableHashRecipe.constantColumns,
    RawArtifact.hashRecipe, rawArtifact, phaseConstantValues,
    hashInputFields, constantFields, digestFields, payloadFields]

theorem absorbRounds_exact :
    rawArtifact.hashRecipe.absorbRounds = absorbRounds := by
  norm_num [VariableHashRecipe.absorbRounds, input_length, rate,
    absorbRounds, hashInputFields]

theorem output_exact :
    rawArtifact.hashRecipe.outputColumns =
      (rawArtifact.hashRecipe.callOutputColumns
        rawArtifact.hashRecipe.absorbRounds).take 4 := by
  have full :
      rawArtifact.hashRecipe.inputColumns.length =
        rate * rawArtifact.hashRecipe.absorbRounds := by
    rw [input_length, absorbRounds_exact]
    norm_num [rate, hashInputFields, absorbRounds,
      constantFields, digestFields, payloadFields]
  rw [finalCallOutputColumns_eq_of_fullAbsorbRounds
    rawArtifact.hashRecipe full]
  rw [absorbRounds_exact]
  norm_num [rawArtifact, RawArtifact.hashRecipe, phaseConstantValues,
    VariableHashRecipe.zeroColumn, absorbRounds,
    hashInputFields, constantFields, digestFields, payloadFields,
    rate, permutationRows]
  rfl

theorem trace_ownedValid : rawArtifact.hashRecipe.trace.OwnedValid := by
  exact ownedValid rawArtifact.hashRecipe (by
      rw [absorbRounds_exact]
      norm_num [absorbRounds, hashInputFields,
        constantFields, digestFields, payloadFields, rate]) (by
      rw [input_length, absorbRounds_exact]
      norm_num [hashInputFields, absorbRounds,
        constantFields, digestFields, payloadFields, rate]) output_exact

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullPhaseSemantic
