import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalProfileSelection

/-! Structural validation of the exact terminal profile-selection artifact. -/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalProfileSelection

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProfileSelection.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalProfileSelection

theorem rawArtifact_valid : rawArtifact.Valid := by
  refine {
    schemaVersion := rfl
    profileId := rfl
    sourceArtifactIdentity := rfl
    lifecycleScope := rfl
    rowFamily := rfl
    rowCount := by decide
    selectorCount := rfl
    selectorsInside := by decide }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalProfileSelection
