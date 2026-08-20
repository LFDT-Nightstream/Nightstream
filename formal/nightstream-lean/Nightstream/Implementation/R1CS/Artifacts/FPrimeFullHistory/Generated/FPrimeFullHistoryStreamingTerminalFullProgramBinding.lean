import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalProgramBindingSchema

/-! Generated exact full-layout Rust terminal Nebula program-binding recipe.

Rust compares every row with a reference built by the production function.
The empty SHA field is legacy diagnostic structure and is not authority.
Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullProgramBinding

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProgramBinding.Artifact

def lifecycleScope : String := "recursive-terminal-arm-435"

def constantValues : List Nat := [12872764427556359090, 15292966035223957217, 5743294475876934593, 16709509729038889535, 2612558357572312719, 13271880277905665356, 1514749146798453643, 858875686232489973, 16762699050894319676, 17667643812818287206, 2571937014624677718, 1670829655690360737, 40, 30521782141150574, 31069335676202596, 27422324158721583, 27428861317902383, 29665297931853677, 212436215662]

def rawArtifact : RawArtifact :=
  { schemaVersion := 1,
    profileId := "nightstream/goldilocks/streaming-terminal-full-program-binding/v1",
    sourceIdentity := "rust:nightstream/streaming-terminal-lifecycle/source-rows/v1",
    sourceRowsSha256 := "", rowCount := 3644, columnCount := 28863843,
    sourceRowStart := 352042, finalRowStart := 352042,
    constantValues := constantValues, constantStartColumn := 28393899,
    inputColumns := [28393911, 28393912, 28393913, 28393914, 28393915, 28393916, 28393917, 28393899, 28393900, 28393901, 28393902, 28393903, 28393904, 28393905, 28393906, 28393907, 28393908, 28393909, 28393910], hashOutputColumns := [28397531, 28397532, 28397533, 28397534],
    carriedBindingColumns := [28041931, 28041932, 28041933, 28041934], equalityRowStart := 3640 }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullProgramBinding
