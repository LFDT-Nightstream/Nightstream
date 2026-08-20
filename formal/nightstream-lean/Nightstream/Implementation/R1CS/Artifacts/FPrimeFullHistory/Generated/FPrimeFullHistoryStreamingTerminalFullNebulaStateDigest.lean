import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestLinkSchema

/-! Generated exact full-layout Rust terminal Nebula-state-digest recipe.

Rust compares all rows with the authoritative audit recipe under the emitted relocation.
The empty SHA field is legacy diagnostic structure and is not authority.
Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullNebulaStateDigest

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact

def lifecycleScope : String := "recursive-terminal-arm-435"

def absentConstantValues : List Nat := [36, 30521782141150574, 31069335676202596, 27422324158721583, 28252386919279663, 33266224450594665, 52, 0, 0, 0, 0, 0, 4]

def presentConstantValues : List Nat := [36, 30521782141150574, 31069335676202596, 27422324158721583, 28252386919279663, 33266224450594665, 52, 1, 2, 4]

def rawArtifact : RawArtifact :=
  { schemaVersion := 2,
    profileId := "nightstream/goldilocks/streaming-terminal-full-nebula-state-digest/v1",
    sourceIdentity := "rust:nightstream/streaming-terminal-lifecycle/source-rows/v1",
    sourceRowsSha256 := "", rowCount := 19353, columnCount := 28863843,
    sourceRowStart := 332689, finalRowStart := 332689, openColumn := 28041935,
    absentConstantValues := absentConstantValues, absentConstantStartColumn := 28374551,
    absentInputColumns := [28374551, 28374552, 28374553, 28374554, 28374555, 28374556, 28374557, 28041931, 28041932, 28041933, 28041934, 28041936, 28041937, 28041938, 28041951, 28041952, 28374558, 28374559, 28374560, 28374561, 28374562, 28374563, 28041943, 28041944, 28041945, 28041946, 28041947, 28041948, 28041949, 28041950, 28041953, 28041954, 28041955, 28041956, 28041957, 28041958, 28041959, 28041960, 28041961, 28041962, 28041963, 28041964, 28041965, 28041966, 28041967, 28041968, 28041969, 28041970, 28041971, 28041972, 28041973, 28041974, 28041975, 28041976, 28041977, 28041978, 28041979, 28041980], absentOutputColumns := [28384216, 28384217, 28384218, 28384219],
    presentConstantValues := presentConstantValues, presentConstantStartColumn := 28384224,
    presentInputColumns := [28384224, 28384225, 28384226, 28384227, 28384228, 28384229, 28384230, 28041931, 28041932, 28041933, 28041934, 28041936, 28041937, 28041938, 28041951, 28041952, 28384231, 28384232, 28041939, 28041940, 28041941, 28041942, 28384233, 28041943, 28041944, 28041945, 28041946, 28041947, 28041948, 28041949, 28041950, 28041953, 28041954, 28041955, 28041956, 28041957, 28041958, 28041959, 28041960, 28041961, 28041962, 28041963, 28041964, 28041965, 28041966, 28041967, 28041968, 28041969, 28041970, 28041971, 28041972, 28041973, 28041974, 28041975, 28041976, 28041977, 28041978, 28041979, 28041980], presentOutputColumns := [28393887, 28393888, 28393889, 28393890],
    hashOutputColumns := [28393895, 28393896, 28393897, 28393898], xOutStateColumns := [28041927, 28041928, 28041929, 28041930],
    baselineDigestValue := 14091033322851699371, absentRowStart := 1, presentRowStart := 9674,
    muxRowStart := 19345, equalityRowStart := 19349, selectedSourceRow := 352038 }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullNebulaStateDigest
