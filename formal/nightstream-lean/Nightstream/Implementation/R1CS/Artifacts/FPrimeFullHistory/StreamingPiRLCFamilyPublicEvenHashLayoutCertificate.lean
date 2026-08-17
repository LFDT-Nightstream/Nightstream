import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic

/-!
Contract: bounded XOut Poseidon2 hash-layout certificate for the Rust-emitted
even PiRLC public-family arm.

Owns two fixed nine-round hash traces and their links to exact compact
Poseidon2 calls and glue rows. It owns no hash collision-resistance claim.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenHashLayoutCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact

theorem evenArm_afterXOutHash_rounds_valid :
    (List.range 9).all (fun index =>
      RawHash.roundValid evenArm evenArm.afterXOutHash index) = true := by
  rfl

theorem evenArm_afterXOutHash_valid :
    RawHash.Valid evenArm evenArm.afterXOutPreimageColumns
      evenArm.afterXOutDigestColumns 526 evenArm.afterXOutHash := by
  unfold RawHash.Valid
  rw [evenArm_afterXOutHash_rounds_valid]
  norm_num [columnsValid, hasGlueIndex, evenArm]

theorem evenArm_beforeXOutHash_rounds_valid :
    (List.range 9).all (fun index =>
      RawHash.roundValid evenArm evenArm.beforeXOutHash index) = true := by
  rfl

theorem evenArm_beforeXOutHash_valid :
    RawHash.Valid evenArm evenArm.beforeXOutPreimageColumns
      evenArm.beforeXOutDigestColumns 535 evenArm.beforeXOutHash := by
  unfold RawHash.Valid
  rw [evenArm_beforeXOutHash_rounds_valid]
  norm_num [columnsValid, hasGlueIndex, evenArm]

theorem evenArm_hashLayout_valid : evenArm.HashLayoutValid :=
  ⟨evenArm_afterXOutHash_valid, evenArm_beforeXOutHash_valid⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenHashLayoutCertificate
