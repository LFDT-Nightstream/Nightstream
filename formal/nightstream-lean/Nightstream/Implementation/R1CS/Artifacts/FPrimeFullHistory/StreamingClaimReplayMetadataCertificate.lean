import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayCoordinateCallCertificates

/-!
Contract: structural metadata certificate for the Rust-emitted streaming
claim-replay artifact.

Assurance tier: Rust-to-Lean artifact metadata certificate.

Owns the exact profile scalars and arm roles in `RawArtifact.MetadataValid`.
It reuses the separate exact coordinate-call identity certificates.

Does not own leaf geometry, row ownership, or claim semantics.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayMetadataCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateCallCertificates

theorem rawArtifact_metadata_valid : rawArtifact.MetadataValid := by
  unfold RawArtifact.MetadataValid
  simp only [rawArtifact]
  rw [fullArm_coordinateCalls_exact, finalArm_coordinateCalls_exact]
  norm_num [fullArm, finalArm, fullStatementFreshCall,
    fullRunningMetadataCall, finalStatementFreshCall]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayMetadataCertificate
