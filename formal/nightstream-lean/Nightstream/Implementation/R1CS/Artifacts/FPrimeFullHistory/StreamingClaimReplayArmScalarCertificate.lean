import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay

/-!
Contract: structural scalar certificates for the two Rust-emitted streaming
claim-replay arms.

Assurance tier: Rust-to-Lean artifact geometry certificate.

Owns row and column positivity, public width, replay and state-digest call
counts, and the exact Poseidon2 list lengths.

Does not own individual calls, row ownership, or claim semantics.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArmScalarCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact

theorem fullArm_scalar_valid : fullArm.ScalarValid := by
  norm_num [RawArm.ScalarValid, fullArm]

theorem finalArm_scalar_valid : finalArm.ScalarValid := by
  norm_num [RawArm.ScalarValid, finalArm]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArmScalarCertificate
