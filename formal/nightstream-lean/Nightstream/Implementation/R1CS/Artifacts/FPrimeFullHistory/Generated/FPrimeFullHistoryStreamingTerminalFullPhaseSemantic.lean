import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticSchema

/-! Generated exact full-layout Rust terminal phase-semantic recipe.

Rust compares all rows with the authoritative audit recipe under the emitted relocation.
The empty SHA field is legacy diagnostic structure and is not authority.
Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullPhaseSemantic

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact

def lifecycleScope : String := "recursive-terminal-arm-435"

def phaseConstantValues : List Nat := [57, 30521782141150574, 31069335676202596, 27422324158721583, 30796712690673199, 27414614995316581, 29396737889036653, 30792317818729313, 33266151269363297, 49, 2169]

def rawArtifact : RawArtifact :=
  { schemaVersion := 2,
    profileId := "nightstream/goldilocks/streaming-terminal-full-phase-semantic/v1",
    sourceIdentity := "rust:nightstream/streaming-terminal-lifecycle/source-rows/v1",
    sourceRowsSha256 := "", rowCount := 330401, columnCount := 28863843,
    sourceRowStart := 2288, finalRowStart := 2288,
    constantValues := phaseConstantValues, constantStartColumn := 28044154,
    localColumns := [28041981, 28041982, 28041983, 28041984], payloadColumns := List.range' 28041985 2169,
    hashOutputColumns := [28374543, 28374544, 28374545, 28374546], xOutSemanticColumns := [28041918, 28041919, 28041920, 28041921],
    baselineDigestValue := 16993964594624123621, equalityRowStart := 330397 }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullPhaseSemantic
