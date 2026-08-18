import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutContextSchema

/-! Generated compact geometry for the exact Rust terminal XOut context family.

Rust compares all 24 source rows with the structural Lean recipe.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutContext

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContext.Artifact

def artifactSha256 : String := "3218d4392aba73591be85a545a019bdee250513ab9be3a446a279bee465047d6"

def rawArtifact : RawArtifact :=
  { schemaVersion := 1, profileId := "nightstream/goldilocks/streaming-terminal-x-out-context/v1",
    sourceIdentity := "rust:streaming-terminal-x-out-context/v1",
    sourceRowsSha256 := "af2ed2c8ace07a629d6ae35e1af742abb35d180178d58b2147e4b70a463b0a11", rowCount := 24, columnCount := 23087,
    domainTag := 1313210370, acceptedWorkItems := 436, nebulaMarker := 1312967745,
    baselineChangedValue := 11, mutatedChangedValue := 12,
    xOutColumns := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    vkFsSourceColumns := [91, 92, 93, 94], piCcsHeaderSourceColumns := [95, 96, 97, 98],
    boundarySourceColumns := [99, 100, 101, 102], accumulatorSourceColumns := [103, 104, 105, 106] }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutContext
