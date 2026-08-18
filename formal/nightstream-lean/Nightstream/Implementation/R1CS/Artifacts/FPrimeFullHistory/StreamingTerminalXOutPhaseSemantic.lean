import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic

/-! Public facade for the exact Rust terminal XOut phase-semantic leaf. -/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPhaseSemantic

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic

theorem rawArtifact_valid : rawArtifact.Valid := by
  unfold RawArtifact.Valid
  refine ⟨rfl, rfl, rfl, by decide, ?_⟩
  norm_num [rawArtifact, totalRows, hashTotalRows, equalityRowCount,
    constantFields, expectedConstantValues, phaseConstantValues,
    Nightstream.Implementation.R1CS.goldilocksP]

theorem sourceRows_exact :
    List.range' rawArtifact.sourceRowStart rawArtifact.rowCount =
      List.range' 24 3636 := by
  rfl

theorem finalRows_exact :
    List.range' rawArtifact.finalRowStart rawArtifact.rowCount =
      List.range' 24 3636 := by
  rfl

theorem equalityRows_length : rawArtifact.equalityRows.length = 4 := by
  norm_num [RawArtifact.equalityRows, digestFields]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPhaseSemantic
