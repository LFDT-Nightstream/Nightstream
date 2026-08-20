import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutContextSchema

/-! Generated exact full-layout Rust terminal XOut context geometry.

The empty SHA field is legacy diagnostic structure and is not authority.
Emits constraints: no. Rust emits the checked rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullXOutContext

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContext.Artifact

def lifecycleScope : String := "recursive-terminal-arm-435"

def rowStart : Nat := 2264

def rowStop : Nat := 2288

def rawArtifact : RawArtifact :=
  { schemaVersion := 1,
    profileId := "nightstream/goldilocks/streaming-terminal-full-x-out-context/v1",
    sourceIdentity := "rust:nightstream/streaming-terminal-lifecycle/source-rows/v1",
    sourceRowsSha256 := "", rowCount := 24, columnCount := 28863843,
    domainTag := 1313210370, acceptedWorkItems := 436, nebulaMarker := 1312967745,
    baselineChangedValue := 0, mutatedChangedValue := 1,
    xOutColumns := [28041899, 28041900, 28041901, 28041902, 28041903, 28041904, 28041905, 28041906, 28041907, 28041908, 28041909, 28041910, 28041911, 28041912, 28041913, 28041914, 28041915, 28041916, 28041917, 28041918, 28041919, 28041920, 28041921, 28041922, 28041923, 28041924, 28041925, 28041926, 28041927, 28041928, 28041929, 28041930],
    vkFsSourceColumns := [28041883, 28041884, 28041885, 28041886], piCcsHeaderSourceColumns := [28041887, 28041888, 28041889, 28041890],
    boundarySourceColumns := [28041891, 28041892, 28041893, 28041894], accumulatorSourceColumns := [28041895, 28041896, 28041897, 28041898] }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullXOutContext
