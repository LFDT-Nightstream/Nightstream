import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticSchema

/-! Generated compact geometry for the exact Rust terminal XOut phase-semantic family.

Rust compares all 3,636 source rows with the structural Lean recipe.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact

def artifactSha256 : String := "0fb48b10c0d65b7f831c05719cc1c2fdf513c10c005b3f5643502d0bc638b851"

def phaseConstantValues : List Nat := [57, 30521782141150574, 31069335676202596, 27422324158721583, 30796712690673199, 27414614995316581, 29396737889036653, 30792317818729313, 33266151269363297, 49, 4]

def rawArtifact : RawArtifact :=
  { schemaVersion := 1, profileId := "nightstream/goldilocks/streaming-terminal-phase-semantic/v1",
    sourceIdentity := "rust:streaming-terminal-phase-semantic/v1",
    sourceRowsSha256 := "af2ed2c8ace07a629d6ae35e1af742abb35d180178d58b2147e4b70a463b0a11", rowCount := 3636, columnCount := 23087,
    sourceRowStart := 24, finalRowStart := 24,
    constantValues := phaseConstantValues, constantStartColumn := 107,
    localColumns := [33, 34, 35, 36], payloadColumns := [37, 38, 39, 40],
    hashOutputColumns := [3731, 3732, 3733, 3734], xOutSemanticColumns := [20, 21, 22, 23],
    baselineDigestValue := 509956021210391786, equalityRowStart := 3632 }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic
