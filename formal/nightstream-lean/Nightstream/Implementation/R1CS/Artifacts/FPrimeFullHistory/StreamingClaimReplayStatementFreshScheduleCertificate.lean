import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay

/-!
Contract: exact schedule identity for the Rust statement-and-fresh coordinate
map.

Assurance tier: structural Rust-to-Lean artifact certificate.

Owns the equality between the literal Rust schedule and the verifier-owned
schedule derived from the fixed statement-and-fresh seed profile.

Does not own sampler liveness, physical placement, or lifecycle semantics.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStatementFreshScheduleCertificate

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay

theorem schedule_exact :
    statementFreshSchedule = MapKind.statementFresh.expectedSchedule := by
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStatementFreshScheduleCertificate
