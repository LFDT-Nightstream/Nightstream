import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay

/-!
Contract: structural canonical-u64 call certificates for both Rust-emitted
streaming claim-replay arms.

Assurance tier: Rust-to-Lean leaf-geometry certificate.

Owns exact ten-call coverage and the scalar geometry of each compact
canonical-u64 call.

Does not own canonical-u64 semantic soundness, other leaf families, or row
ownership.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCanonicalCallCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact

theorem fullArm_canonicalCalls_valid : fullArm.CanonicalCallsValid := by
  norm_num [RawArm.CanonicalCallsValid, CanonicalCall.Valid, fullArm]

theorem finalArm_canonicalCalls_valid : finalArm.CanonicalCallsValid := by
  norm_num [RawArm.CanonicalCallsValid, CanonicalCall.Valid, finalArm]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCanonicalCallCertificate
