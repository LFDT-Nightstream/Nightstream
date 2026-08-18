import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestLinkSchema

/-! Generated link projection for the exact Rust terminal Nebula-state-digest family.

Rust checks the four final links and their ownership in the 19,353-row source family.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact

def artifactSha256 : String := "1c6446af63170105d41caac67f91681a1421416881ccbda7bddf8213229a876e"

def rawArtifact : RawArtifact :=
  { schemaVersion := 1, profileId := "nightstream/goldilocks/streaming-terminal-nebula-state-digest-link/v1",
    sourceIdentity := "rust:streaming-terminal-nebula-state-digest-link/v1",
    sourceRowsSha256 := "af2ed2c8ace07a629d6ae35e1af742abb35d180178d58b2147e4b70a463b0a11", rowCount := 19353, columnCount := 23087,
    sourceRowStart := 3660, finalRowStart := 3660,
    hashOutputColumns := [23083, 23084, 23085, 23086], xOutStateColumns := [29, 30, 31, 32],
    baselineDigestValue := 6284679863123074783, equalityRowStart := 19349, selectedSourceRow := 23009 }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink
