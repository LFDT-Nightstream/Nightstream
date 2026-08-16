import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic

/-!
Contract: bounded XOut Poseidon2 hash-layout certificate for the Rust-emitted
odd PiRLC public-family arm.

Owns two fixed nine-round hash traces and their links to exact compact
Poseidon2 calls and glue rows. It owns no hash collision-resistance claim.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddHashLayoutCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact

theorem oddArm_afterXOutHash_rounds_valid :
    (List.range 9).all (fun index =>
      RawHash.roundValid oddArm oddArm.afterXOutHash index) = true := by
  rfl

theorem oddArm_afterXOutHash_valid :
    RawHash.Valid oddArm oddArm.afterXOutPreimageColumns
      oddArm.afterXOutDigestColumns 472 oddArm.afterXOutHash := by
  unfold RawHash.Valid
  rw [oddArm_afterXOutHash_rounds_valid]
  norm_num [columnsValid, hasGlueIndex, oddArm]

theorem oddArm_beforeXOutHash_rounds_valid :
    (List.range 9).all (fun index =>
      RawHash.roundValid oddArm oddArm.beforeXOutHash index) = true := by
  rfl

theorem oddArm_beforeXOutHash_valid :
    RawHash.Valid oddArm oddArm.beforeXOutPreimageColumns
      oddArm.beforeXOutDigestColumns 481 oddArm.beforeXOutHash := by
  unfold RawHash.Valid
  rw [oddArm_beforeXOutHash_rounds_valid]
  norm_num [columnsValid, hasGlueIndex, oddArm]

theorem oddArm_hashLayout_valid : oddArm.HashLayoutValid :=
  ⟨oddArm_afterXOutHash_valid, oddArm_beforeXOutHash_valid⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddHashLayoutCertificate
