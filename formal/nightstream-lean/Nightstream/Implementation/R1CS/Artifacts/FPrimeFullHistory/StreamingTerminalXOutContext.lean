import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutContext

/-! Public facade for the exact 24-row Rust terminal XOut context leaf. -/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutContext

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContext.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutContext

theorem rawArtifact_valid : rawArtifact.Valid := by
  unfold RawArtifact.Valid
  refine ⟨rfl, rfl, rfl, by decide, ?_⟩
  norm_num [rawArtifact, Nightstream.Implementation.R1CS.goldilocksP]

theorem xOutColumns_exact :
    rawArtifact.xOutColumns = List.range' 1 32 := by
  rfl

theorem contextRows_length : rawArtifact.contextRows.length = 24 := by
  norm_num [RawArtifact.contextRows, copyRows, rawArtifact]

theorem changedColumn_exact : rawArtifact.xOutColumns.getD 1 0 = 2 := by
  rfl

theorem changedSource_exact : rawArtifact.vkFsSourceColumns.getD 0 0 = 91 := by
  rfl

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutContext
