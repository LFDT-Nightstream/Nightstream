import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay

/-!
Contract: structural public-word and digest-pin layout certificates for both
Rust-emitted streaming claim-replay arms.

Assurance tier: Rust-to-Lean artifact layout certificate.

Owns the two fixed ten-word call permutations and the four fixed thirteen-pin
lists, including uniqueness and column bounds.

Does not own public-input semantics, digest authority, or leaf rows.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPublicAndPinLayoutCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact

theorem fullArm_publicWordLayout_valid : fullArm.PublicWordLayoutValid := by
  norm_num [RawArm.PublicWordLayoutValid, fullArm]

theorem finalArm_publicWordLayout_valid : finalArm.PublicWordLayoutValid := by
  norm_num [RawArm.PublicWordLayoutValid, finalArm]

theorem fullArm_digestPinLayout_valid : fullArm.DigestPinLayoutValid := by
  norm_num [RawArm.DigestPinLayoutValid, fullArm]

theorem finalArm_digestPinLayout_valid : finalArm.DigestPinLayoutValid := by
  norm_num [RawArm.DigestPinLayoutValid, finalArm]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPublicAndPinLayoutCertificate
