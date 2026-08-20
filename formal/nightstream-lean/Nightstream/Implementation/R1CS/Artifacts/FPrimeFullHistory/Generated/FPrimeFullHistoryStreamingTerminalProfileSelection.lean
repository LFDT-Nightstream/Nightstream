import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalProfileSelectionSchema

/-! Generated exact Rust rows for terminal profile selection.

Emits constraints: no. Rust emits the described rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalProfileSelection

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProfileSelection.Artifact

def rawArtifact : RawArtifact :=
  { schemaVersion := 1,
    profileId := "nightstream/goldilocks/streaming-terminal-lifecycle/v1",
    sourceArtifactIdentity := "rust:nightstream/streaming-terminal-lifecycle/source-rows/v1",
    lifecycleScope := "recursive-terminal-arm-435",
    rowFamily := "terminal.streaming.profile_selection", rowStart := 2261, rowStop := 2264,
    columnCount := 28863843, selectorColumns := [28038856, 650, 28033367] }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalProfileSelection
