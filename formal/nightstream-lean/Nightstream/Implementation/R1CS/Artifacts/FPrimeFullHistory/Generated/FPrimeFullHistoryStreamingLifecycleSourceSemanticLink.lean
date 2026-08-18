import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSourceSemanticLinkSchema

/-! Generated compact geometry for the exact base and recursive lifecycle semantic-link source stages.

The Rust generator compares every represented source row with its scope-specific compact recipe.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSourceSemanticLink

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSourceSemanticLink.Artifact

def artifactSha256 : String := "8c86eab94e8f4d049dc7c75db6bee97374972b6b6686a6d2a92a6352ab09337f"

def phaseConstantValues : List Nat := [57, 30521782141150574, 31069335676202596, 27422324158721583, 30796712690673199, 27414614995316581, 29396737889036653, 30792317818729313, 33266151269363297, 49, 2169]

def baseArtifact : SourceArtifact :=
  { scope := .base, schemaVersion := 1,
    profileId := "nightstream/goldilocks/streaming-lifecycle-source-semantic-link/base/v1", sourceIdentity := "rust:streaming-lifecycle-source-semantic-link/base/v1",
    sourceRowsSha256 := "b647c041e8e632e49c3863a8e27f2ee496dc9347d7bb705a2342b753cf1ad9ba", rowCount := 665140, columnCount := 665149,
    constantValues := phaseConstantValues,
    beforeSemanticColumns := [1, 2, 3, 4], afterSemanticColumns := [5, 6, 7, 8],
    beforeLocalColumns := [9, 10, 11, 12], afterLocalColumns := [2182, 2183, 2184, 2185],
    beforePayloadStartColumn := 13, afterPayloadStartColumn := 2186,
    beforeHashConstantStartColumn := 4355, afterHashConstantStartColumn := 334752,
    beforeHashOutputColumns := [334744, 334745, 334746, 334747], afterHashOutputColumns := [665141, 665142, 665143, 665144],
    beforePayloadRowStart := 0, beforeHashConstantRowStart := 2169,
    afterPayloadRowStart := 332566, afterHashConstantRowStart := 334735,
    equalityRowStart := 665132 }

def recursiveArtifact : SourceArtifact :=
  { scope := .recursive, schemaVersion := 1,
    profileId := "nightstream/goldilocks/streaming-lifecycle-source-semantic-link/recursive/v1", sourceIdentity := "rust:streaming-lifecycle-source-semantic-link/recursive/v1",
    sourceRowsSha256 := "14bb0c9f0ae92dd3134de5b07a35ea8e7dbff36ab546c4172de53a736ba967fd", rowCount := 662971, columnCount := 665149,
    constantValues := phaseConstantValues,
    beforeSemanticColumns := [1, 2, 3, 4], afterSemanticColumns := [5, 6, 7, 8],
    beforeLocalColumns := [9, 10, 11, 12], afterLocalColumns := [2182, 2183, 2184, 2185],
    beforePayloadStartColumn := 13, afterPayloadStartColumn := 2186,
    beforeHashConstantStartColumn := 4355, afterHashConstantStartColumn := 334752,
    beforeHashOutputColumns := [334744, 334745, 334746, 334747], afterHashOutputColumns := [665141, 665142, 665143, 665144],
    beforePayloadRowStart := 0, beforeHashConstantRowStart := 0,
    afterPayloadRowStart := 330397, afterHashConstantRowStart := 332566,
    equalityRowStart := 662963 }

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleSourceSemanticLink
