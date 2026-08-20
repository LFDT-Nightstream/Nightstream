import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticSchema

/-! Generated compact geometry for the exact Rust terminal XOut phase-semantic family.

Rust compares all 330401 source rows with the structural Lean recipe.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact

def artifactSha256 : String := "c5639e0e53cca6f9aa5c17e5b496708a086ea966b8319912bf64c2a77d9c2333"

def phaseConstantValues : List Nat := [57, 30521782141150574, 31069335676202596, 27422324158721583, 30796712690673199, 27414614995316581, 29396737889036653, 30792317818729313, 33266151269363297, 49, 2169]

def rawArtifact : RawArtifact :=
  { schemaVersion := 2, profileId := "nightstream/goldilocks/streaming-terminal-phase-semantic/v2",
    sourceIdentity := "rust:streaming-terminal-phase-semantic/v2",
    sourceRowsSha256 := "89aae9a5eb9aa1f455cb97d60b648c7fdd03d729935d6d6cc87fe5419773173d", rowCount := 330401, columnCount := 352017,
    sourceRowStart := 24, finalRowStart := 24,
    constantValues := phaseConstantValues, constantStartColumn := 2272,
    localColumns := [33, 34, 35, 36], payloadColumns := List.range' 37 2169,
    hashOutputColumns := [332661, 332662, 332663, 332664], xOutSemanticColumns := [20, 21, 22, 23],
    baselineDigestValue := 18263547049594940461, equalityRowStart := 330397 }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic
