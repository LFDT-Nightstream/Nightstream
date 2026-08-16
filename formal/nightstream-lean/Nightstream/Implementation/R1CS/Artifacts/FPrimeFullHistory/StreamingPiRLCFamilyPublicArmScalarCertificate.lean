import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicEvenPoseidon2CallCertificate
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicOddPoseidon2CallCertificate

/-!
Contract: structural scalar certificates for both Rust-emitted PiRLC
public-family arms.

Owns fixed row, column, public-width, phase-envelope, and call-count scalars.
It reuses the separate structural 490-call length certificates.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArmScalarCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact

theorem evenArm_scalar_valid : evenArm.ScalarValid := by
  unfold RawArm.ScalarValid
  rw [Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenPoseidon2CallCertificate.evenArm_poseidon2Calls_length]
  norm_num [evenArm]

theorem oddArm_scalar_valid : oddArm.ScalarValid := by
  unfold RawArm.ScalarValid
  rw [Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddPoseidon2CallCertificate.oddArm_poseidon2Calls_length]
  norm_num [oddArm]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArmScalarCertificate
