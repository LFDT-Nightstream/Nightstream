import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutContextSchema

/-! Generated compact geometry for the exact Rust terminal XOut context family.

Rust compares all 24 source rows with the structural Lean recipe.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutContext

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContext.Artifact

def artifactSha256 : String := "3be45e3412c97afd17ea616f2ed44389fa0980131108fa759d3c7fd6655f8fa0"

def rawArtifact : RawArtifact :=
  { schemaVersion := 1, profileId := "nightstream/goldilocks/streaming-terminal-x-out-context/v1",
    sourceIdentity := "rust:streaming-terminal-x-out-context/v1",
    sourceRowsSha256 := "89aae9a5eb9aa1f455cb97d60b648c7fdd03d729935d6d6cc87fe5419773173d", rowCount := 24, columnCount := 352017,
    domainTag := 1313210370, acceptedWorkItems := 436, nebulaMarker := 1312967745,
    baselineChangedValue := 11, mutatedChangedValue := 12,
    xOutColumns := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    vkFsSourceColumns := [2256, 2257, 2258, 2259], piCcsHeaderSourceColumns := [2260, 2261, 2262, 2263],
    boundarySourceColumns := [2264, 2265, 2266, 2267], accumulatorSourceColumns := [2268, 2269, 2270, 2271] }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutContext
