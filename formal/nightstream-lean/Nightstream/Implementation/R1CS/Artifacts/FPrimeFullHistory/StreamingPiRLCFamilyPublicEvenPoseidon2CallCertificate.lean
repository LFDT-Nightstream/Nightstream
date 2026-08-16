import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicCertificateSupport

/-!
Contract: bounded Poseidon2 call-geometry certificate for the Rust-emitted
even PiRLC public-family arm.

Owns all 490 compact call records as seven 64-call leaves and one 42-call
leaf. It owns no Poseidon2 semantics or row ownership.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenPoseidon2CallCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCertificateSupport

def evenTail0 := evenArm.poseidon2Calls
def evenChunk0 := evenTail0.take 64
def evenTail1 := evenTail0.drop 64
def evenChunk1 := evenTail1.take 64
def evenTail2 := evenTail1.drop 64
def evenChunk2 := evenTail2.take 64
def evenTail3 := evenTail2.drop 64
def evenChunk3 := evenTail3.take 64
def evenTail4 := evenTail3.drop 64
def evenChunk4 := evenTail4.take 64
def evenTail5 := evenTail4.drop 64
def evenChunk5 := evenTail5.take 64
def evenTail6 := evenTail5.drop 64
def evenChunk6 := evenTail6.take 64
def evenTail7 := evenTail6.drop 64

theorem evenChunk0_length : evenChunk0.length = 64 := by rfl
theorem evenChunk1_length : evenChunk1.length = 64 := by rfl
theorem evenChunk2_length : evenChunk2.length = 64 := by rfl
theorem evenChunk3_length : evenChunk3.length = 64 := by rfl
theorem evenChunk4_length : evenChunk4.length = 64 := by rfl
theorem evenChunk5_length : evenChunk5.length = 64 := by rfl
theorem evenChunk6_length : evenChunk6.length = 64 := by rfl
theorem evenTail7_length : evenTail7.length = 42 := by rfl

theorem evenArm_poseidon2Calls_length :
    evenArm.poseidon2Calls.length = 490 := by
  have tail6 : evenTail6.length = 106 :=
    length_of_take_drop evenChunk6_length evenTail7_length
  have tail5 : evenTail5.length = 170 :=
    length_of_take_drop evenChunk5_length tail6
  have tail4 : evenTail4.length = 234 :=
    length_of_take_drop evenChunk4_length tail5
  have tail3 : evenTail3.length = 298 :=
    length_of_take_drop evenChunk3_length tail4
  have tail2 : evenTail2.length = 362 :=
    length_of_take_drop evenChunk2_length tail3
  have tail1 : evenTail1.length = 426 :=
    length_of_take_drop evenChunk1_length tail2
  exact length_of_take_drop evenChunk0_length tail1

theorem evenChunk0_valid :
    ∀ call ∈ evenChunk0,
      PoseidonCallValid 1233086 call ∧ 275006 ≤ call.rowStart := by
  norm_num [evenChunk0, evenTail0, PoseidonCallValid, evenArm]

theorem evenChunk1_valid :
    ∀ call ∈ evenChunk1,
      PoseidonCallValid 1233086 call ∧ 275006 ≤ call.rowStart := by
  norm_num [evenChunk1, evenTail1, evenTail0, PoseidonCallValid, evenArm]

theorem evenChunk2_valid :
    ∀ call ∈ evenChunk2,
      PoseidonCallValid 1233086 call ∧ 275006 ≤ call.rowStart := by
  norm_num [evenChunk2, evenTail2, evenTail1, evenTail0,
    PoseidonCallValid, evenArm]

theorem evenChunk3_valid :
    ∀ call ∈ evenChunk3,
      PoseidonCallValid 1233086 call ∧ 275006 ≤ call.rowStart := by
  norm_num [evenChunk3, evenTail3, evenTail2, evenTail1, evenTail0,
    PoseidonCallValid, evenArm]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenPoseidon2CallCertificate
