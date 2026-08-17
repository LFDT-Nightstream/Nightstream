import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicOddOwnershipCertificate

/-!
Contract: second bounded owner-schedule certificate for the Rust-emitted odd
PiRLC public-family arm.

Owns five 64-owner leaves, one 37-owner leaf, and their structural composition
with the five earlier leaves.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddOwnershipTailCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOwnershipCertificateSupport
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddOwnershipCertificate

theorem oddOwnerChunk5_checked :
    oddOwnerChunk5.length = 64 ∧
      runOwnerPrefix oddArm oddOwnerCursor5 oddOwnerChunk5 =
        some oddOwnerCursor6 := by
  constructor <;> rfl

theorem oddOwnerChunk6_checked :
    oddOwnerChunk6.length = 64 ∧
      runOwnerPrefix oddArm oddOwnerCursor6 oddOwnerChunk6 =
        some oddOwnerCursor7 := by
  constructor <;> rfl

theorem oddOwnerChunk7_checked :
    oddOwnerChunk7.length = 64 ∧
      runOwnerPrefix oddArm oddOwnerCursor7 oddOwnerChunk7 =
        some oddOwnerCursor8 := by
  constructor <;> rfl

theorem oddOwnerChunk8_checked :
    oddOwnerChunk8.length = 64 ∧
      runOwnerPrefix oddArm oddOwnerCursor8 oddOwnerChunk8 =
        some oddOwnerCursor9 := by
  constructor <;> rfl

theorem oddOwnerChunk9_checked :
    oddOwnerChunk9.length = 64 ∧
      runOwnerPrefix oddArm oddOwnerCursor9 oddOwnerChunk9 =
        some oddOwnerCursor10 := by
  constructor <;> rfl

theorem oddOwnerTail10_checked :
    oddOwnerTail10.length = 37 ∧
      runOwnerPrefix oddArm oddOwnerCursor10 oddOwnerTail10 =
        some oddOwnerCursor11 := by
  constructor <;> rfl

theorem oddArm_ownership_valid : oddArm.OwnershipValid := by
  have checked9 := runOwnerPrefix_of_take_drop
    oddOwnerChunk9_checked.2 oddOwnerTail10_checked.2
  have checked8 := runOwnerPrefix_of_take_drop
    oddOwnerChunk8_checked.2 checked9
  have checked7 := runOwnerPrefix_of_take_drop
    oddOwnerChunk7_checked.2 checked8
  have checked6 := runOwnerPrefix_of_take_drop
    oddOwnerChunk6_checked.2 checked7
  have checked5 := runOwnerPrefix_of_take_drop
    oddOwnerChunk5_checked.2 checked6
  have checked4 := runOwnerPrefix_of_take_drop
    oddOwnerChunk4_checked.2 checked5
  have checked3 := runOwnerPrefix_of_take_drop
    oddOwnerChunk3_checked.2 checked4
  have checked2 := runOwnerPrefix_of_take_drop
    oddOwnerChunk2_checked.2 checked3
  have checked1 := runOwnerPrefix_of_take_drop
    oddOwnerChunk1_checked.2 checked2
  have checked0 := runOwnerPrefix_of_take_drop
    oddOwnerChunk0_checked.2 checked1
  exact ownershipValid_of_run checked0

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddOwnershipTailCertificate
