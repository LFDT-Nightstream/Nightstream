import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink

/-! Public facade for the exact Rust terminal Nebula-state-digest leaf. -/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestLink

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink

theorem rawArtifact_valid : rawArtifact.Valid := by
  unfold RawArtifact.Valid
  refine ⟨rfl, rfl, rfl, by decide, ?_⟩
  norm_num [rawArtifact, absentConstantValues, presentConstantValues,
    familyRows, digestFields, bitRowCount, muxRowCount, equalityRowCount,
    hashRowCount,
    absentHashRows, presentHashRows, absentConstantFields,
    absentInputFields, presentConstantFields, presentInputFields,
    rate, permutationRows,
    Nightstream.Implementation.R1CS.goldilocksP]

theorem sourceRows_exact :
    List.range' rawArtifact.sourceRowStart rawArtifact.rowCount =
      List.range' 330425 19353 := by
  rfl

theorem finalRows_exact :
    List.range' rawArtifact.finalRowStart rawArtifact.rowCount =
      List.range' 330425 19353 := by
  rfl

theorem equalityRows_length : rawArtifact.equalityRows.length = 4 := by
  norm_num [RawArtifact.equalityRows, digestFields]

theorem row_partition_exact :
    rawArtifact.absentRowStart = 1 ∧
      rawArtifact.presentRowStart = 9674 ∧
      rawArtifact.muxRowStart = 19345 ∧
      rawArtifact.equalityRowStart = 19349 := by
  norm_num [rawArtifact]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestLink
