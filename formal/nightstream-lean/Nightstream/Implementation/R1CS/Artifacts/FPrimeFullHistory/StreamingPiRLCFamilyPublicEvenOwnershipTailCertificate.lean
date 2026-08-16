import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicEvenOwnershipCertificate

/-!
Contract: second bounded owner-schedule certificate for the Rust-emitted even
PiRLC public-family arm.

Owns four 64-owner leaves, one 47-owner leaf, and their structural composition
with the five earlier leaves.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenOwnershipTailCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOwnershipCertificateSupport
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenOwnershipCertificate

theorem evenOwnerChunk5_checked :
    evenOwnerChunk5.length = 64 ∧
      runOwnerPrefix evenArm evenOwnerCursor5 evenOwnerChunk5 =
        some evenOwnerCursor6 := by
  constructor <;> rfl

theorem evenOwnerChunk6_checked :
    evenOwnerChunk6.length = 64 ∧
      runOwnerPrefix evenArm evenOwnerCursor6 evenOwnerChunk6 =
        some evenOwnerCursor7 := by
  constructor <;> rfl

theorem evenOwnerChunk7_checked :
    evenOwnerChunk7.length = 64 ∧
      runOwnerPrefix evenArm evenOwnerCursor7 evenOwnerChunk7 =
        some evenOwnerCursor8 := by
  constructor <;> rfl

theorem evenOwnerChunk8_checked :
    evenOwnerChunk8.length = 64 ∧
      runOwnerPrefix evenArm evenOwnerCursor8 evenOwnerChunk8 =
        some evenOwnerCursor9 := by
  constructor <;> rfl

theorem evenOwnerTail9_checked :
    evenOwnerTail9.length = 47 ∧
      runOwnerPrefix evenArm evenOwnerCursor9 evenOwnerTail9 =
        some evenOwnerCursor10 := by
  constructor <;> rfl

theorem evenArm_ownership_valid : evenArm.OwnershipValid := by
  have checked8 := runOwnerPrefix_of_take_drop
    evenOwnerChunk8_checked.2 evenOwnerTail9_checked.2
  have checked7 := runOwnerPrefix_of_take_drop
    evenOwnerChunk7_checked.2 checked8
  have checked6 := runOwnerPrefix_of_take_drop
    evenOwnerChunk6_checked.2 checked7
  have checked5 := runOwnerPrefix_of_take_drop
    evenOwnerChunk5_checked.2 checked6
  have checked4 := runOwnerPrefix_of_take_drop
    evenOwnerChunk4_checked.2 checked5
  have checked3 := runOwnerPrefix_of_take_drop
    evenOwnerChunk3_checked.2 checked4
  have checked2 := runOwnerPrefix_of_take_drop
    evenOwnerChunk2_checked.2 checked3
  have checked1 := runOwnerPrefix_of_take_drop
    evenOwnerChunk1_checked.2 checked2
  have checked0 := runOwnerPrefix_of_take_drop
    evenOwnerChunk0_checked.2 checked1
  exact ownershipValid_of_run checked0

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenOwnershipTailCertificate
