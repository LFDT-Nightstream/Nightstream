import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay

/-!
Contract: exact schedule identity for the two Rust running-instance
coordinate maps.

Assurance tier: structural Rust-to-Lean artifact certificate.

Owns the equality between each literal Rust schedule and the corresponding
verifier-owned schedule derived from its fixed seed profile.

Does not own sampler liveness, physical placement, or lifecycle semantics.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayRunningScheduleCertificate

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay

theorem commitments_schedule_exact :
    runningCommitmentsSchedule =
      MapKind.runningCommitments.expectedSchedule := by
  rfl

theorem public_schedule_exact :
    runningPublicSchedule = MapKind.runningPublic.expectedSchedule := by
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayRunningScheduleCertificate
