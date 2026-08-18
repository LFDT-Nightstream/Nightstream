import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSemanticLinkSchema

/-! Generated compact geometry for the Rust lifecycle semantic-link family.

The Rust generator compares every represented source row with the compact recipe.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSemanticLink

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact

def artifactSha256 : String := "4cb14e8de528c1b6bd399176bb484cabc539cc1d53b47f73f280ccbfacbffaee"

def phaseConstantValues : List Nat := [57, 30521782141150574, 31069335676202596, 27422324158721583, 30796712690673199, 27414614995316581, 29396737889036653, 30792317818729313, 33266151269363297, 49, 2169]

def rawArtifact : RawArtifact :=
  { schemaVersion := 1, profileId := "nightstream/goldilocks/streaming-lifecycle-semantic-link/v1",
    sourceIdentity := "rust:streaming-lifecycle-semantic-link/v1",
    sourceRowsSha256 := "7871fed6057ca32416bfc5396c2186fa92a2475a10ec56ce9c72d6dfd07d3548", rowCount := 665140, columnCount := 665149,
    constantValues := phaseConstantValues,
    beforeSemanticColumns := [1, 2, 3, 4], afterSemanticColumns := [5, 6, 7, 8],
    beforeLocalColumns := [9, 10, 11, 12], afterLocalColumns := [13, 14, 15, 16],
    beforePayloadStartColumn := 17, afterPayloadStartColumn := 2186,
    beforeHashConstantStartColumn := 4355, afterHashConstantStartColumn := 334752,
    beforeHashOutputColumns := [334744, 334745, 334746, 334747], afterHashOutputColumns := [665141, 665142, 665143, 665144],
    equalityRowStart := 665132 }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSemanticLink
