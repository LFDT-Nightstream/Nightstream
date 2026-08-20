import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalProfile

/-!
Contract: structural validation of the exact Rust terminal ownership profile.

This certificate checks only compact geometry and ownership metadata. Row
soundness remains in the corresponding handwritten leaf modules.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalProfile

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProfile.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalProfile

theorem rawArtifact_valid : rawArtifact.Valid := by
  refine {
    schemaVersion := rfl
    profileId := rfl
    lifecycleScope := rfl
    sourceArtifactIdentity := rfl
    finalArtifactIdentity := rfl
    acceptedWorkItems := rfl
    terminalArm := by decide
    sourcePublicInside := by decide
    finalPublicInside := by decide
    selectorsInside := by decide
    sourceStageRows := by decide
    finalStageRuns := finalStageRunsWithin
    xOutLength := rfl
    nebulaLaneLength := rfl
    localStateLength := by decide
    localStateRows := by decide
    delayedPayloadRows := by decide
    sourceStageBindings := sourceStageBindingsWithin }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalProfile
