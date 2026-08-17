import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicCertificateSupport

/-!
Contract: bounded Poseidon2 call-geometry certificate for the Rust-emitted
odd PiRLC public-family arm.

Owns all 544 compact call records as eight 64-call leaves and one 32-call
leaf. It owns no Poseidon2 semantics or row ownership.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddPoseidon2CallCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCertificateSupport

def oddTail0 := oddArm.poseidon2Calls
def oddChunk0 := oddTail0.take 64
def oddTail1 := oddTail0.drop 64
def oddChunk1 := oddTail1.take 64
def oddTail2 := oddTail1.drop 64
def oddChunk2 := oddTail2.take 64
def oddTail3 := oddTail2.drop 64
def oddChunk3 := oddTail3.take 64
def oddTail4 := oddTail3.drop 64
def oddChunk4 := oddTail4.take 64
def oddTail5 := oddTail4.drop 64
def oddChunk5 := oddTail5.take 64
def oddTail6 := oddTail5.drop 64
def oddChunk6 := oddTail6.take 64
def oddTail7 := oddTail6.drop 64
def oddChunk7 := oddTail7.take 64
def oddTail8 := oddTail7.drop 64

theorem oddChunk0_length : oddChunk0.length = 64 := by rfl
theorem oddChunk1_length : oddChunk1.length = 64 := by rfl
theorem oddChunk2_length : oddChunk2.length = 64 := by rfl
theorem oddChunk3_length : oddChunk3.length = 64 := by rfl
theorem oddChunk4_length : oddChunk4.length = 64 := by rfl
theorem oddChunk5_length : oddChunk5.length = 64 := by rfl
theorem oddChunk6_length : oddChunk6.length = 64 := by rfl
theorem oddChunk7_length : oddChunk7.length = 64 := by rfl
theorem oddTail8_length : oddTail8.length = 32 := by rfl

theorem oddArm_poseidon2Calls_length :
    oddArm.poseidon2Calls.length = 544 := by
  have tail7 : oddTail7.length = 96 :=
    length_of_take_drop oddChunk7_length oddTail8_length
  have tail6 : oddTail6.length = 160 :=
    length_of_take_drop oddChunk6_length tail7
  have tail5 : oddTail5.length = 224 :=
    length_of_take_drop oddChunk5_length tail6
  have tail4 : oddTail4.length = 288 :=
    length_of_take_drop oddChunk4_length tail5
  have tail3 : oddTail3.length = 352 :=
    length_of_take_drop oddChunk3_length tail4
  have tail2 : oddTail2.length = 416 :=
    length_of_take_drop oddChunk2_length tail3
  have tail1 : oddTail1.length = 480 :=
    length_of_take_drop oddChunk1_length tail2
  exact length_of_take_drop oddChunk0_length tail1

theorem oddChunk0_valid :
    ∀ call ∈ oddChunk0,
      PoseidonCallValid 1302326 call ∧ 311846 ≤ call.rowStart := by
  norm_num [oddChunk0, oddTail0, PoseidonCallValid, oddArm]

theorem oddChunk1_valid :
    ∀ call ∈ oddChunk1,
      PoseidonCallValid 1302326 call ∧ 311846 ≤ call.rowStart := by
  norm_num [oddChunk1, oddTail1, oddTail0, PoseidonCallValid, oddArm]

theorem oddChunk2_valid :
    ∀ call ∈ oddChunk2,
      PoseidonCallValid 1302326 call ∧ 311846 ≤ call.rowStart := by
  norm_num [oddChunk2, oddTail2, oddTail1, oddTail0,
    PoseidonCallValid, oddArm]

theorem oddChunk3_valid :
    ∀ call ∈ oddChunk3,
      PoseidonCallValid 1302326 call ∧ 311846 ≤ call.rowStart := by
  norm_num [oddChunk3, oddTail3, oddTail2, oddTail1, oddTail0,
    PoseidonCallValid, oddArm]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddPoseidon2CallCertificate
