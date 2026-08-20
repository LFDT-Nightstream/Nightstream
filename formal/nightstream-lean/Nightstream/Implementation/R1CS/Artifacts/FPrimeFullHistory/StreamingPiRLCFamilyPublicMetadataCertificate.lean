import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic

/-!
Contract: bounded metadata certificate for the Rust-emitted PiRLC
public-family artifact.

Owns only the fixed profile scalars in `RawArtifact.MetadataValid`. It owns no
arm geometry, row ownership, or protocol semantics.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicMetadataCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact

theorem rawArtifact_metadata_valid : rawArtifact.MetadataValid := by
  norm_num [RawArtifact.MetadataValid, RawPublicDecoder.Valid, rawArtifact,
    evenArm, oddArm]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicMetadataCertificate
