import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic

/-!
Contract: bounded canonical-u64 call certificates for both Rust-emitted
PiRLC public-family arms.

Owns the exact 11-call lists and their scalar geometry. It owns no
canonical-u64 semantics or row ownership.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCanonicalCallCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact

theorem evenArm_canonicalCalls_valid : evenArm.CanonicalCallsValid := by
  norm_num [RawArm.CanonicalCallsValid, CanonicalCall.Valid, evenArm]

theorem oddArm_canonicalCalls_valid : oddArm.CanonicalCallsValid := by
  norm_num [RawArm.CanonicalCallsValid, CanonicalCall.Valid, oddArm]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCanonicalCallCertificate
