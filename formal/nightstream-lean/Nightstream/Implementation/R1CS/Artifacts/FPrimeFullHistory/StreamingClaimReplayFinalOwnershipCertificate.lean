import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayOwnershipCertificateSupport

/-!
Contract: exact owner-schedule certificate for the Rust-emitted terminal
streaming claim-replay arm.

Assurance tier: artifact-checked ownership certificate.

Owns all 701 owner records as eleven bounded prefix certificates. Each leaf
checks row continuity, source-object identity, and the next per-family index.

Does not own row semantics or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFinalOwnershipCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayOwnershipCertificateSupport

def finalOwnerTail0 := finalArm.owners
def finalOwnerChunk0 := finalOwnerTail0.take 64
def finalOwnerTail1 := finalOwnerTail0.drop 64
def finalOwnerChunk1 := finalOwnerTail1.take 64
def finalOwnerTail2 := finalOwnerTail1.drop 64
def finalOwnerChunk2 := finalOwnerTail2.take 64
def finalOwnerTail3 := finalOwnerTail2.drop 64
def finalOwnerChunk3 := finalOwnerTail3.take 64
def finalOwnerTail4 := finalOwnerTail3.drop 64
def finalOwnerChunk4 := finalOwnerTail4.take 64
def finalOwnerTail5 := finalOwnerTail4.drop 64
def finalOwnerChunk5 := finalOwnerTail5.take 64
def finalOwnerTail6 := finalOwnerTail5.drop 64
def finalOwnerChunk6 := finalOwnerTail6.take 64
def finalOwnerTail7 := finalOwnerTail6.drop 64
def finalOwnerChunk7 := finalOwnerTail7.take 64
def finalOwnerTail8 := finalOwnerTail7.drop 64
def finalOwnerChunk8 := finalOwnerTail8.take 64
def finalOwnerTail9 := finalOwnerTail8.drop 64
def finalOwnerChunk9 := finalOwnerTail9.take 64
def finalOwnerTail10 := finalOwnerTail9.drop 64

def finalOwnerCursor0 : OwnerCursor := OwnerCursor.zero
def finalOwnerCursor1 : OwnerCursor :=
  { row := 1997, canonical := 2, poseidon2 := 3, coordinate := 0, glue := 59 }
def finalOwnerCursor2 : OwnerCursor :=
  { row := 40397, canonical := 2, poseidon2 := 67, coordinate := 0, glue := 59 }
def finalOwnerCursor3 : OwnerCursor :=
  { row := 78797, canonical := 2, poseidon2 := 131, coordinate := 0, glue := 59 }
def finalOwnerCursor4 : OwnerCursor :=
  { row := 117197, canonical := 2, poseidon2 := 195, coordinate := 0, glue := 59 }
def finalOwnerCursor5 : OwnerCursor :=
  { row := 147211, canonical := 2, poseidon2 := 245, coordinate := 0, glue := 73 }
def finalOwnerCursor6 : OwnerCursor :=
  { row := 269317, canonical := 2, poseidon2 := 245, coordinate := 1, glue := 136 }
def finalOwnerCursor7 : OwnerCursor :=
  { row := 269381, canonical := 2, poseidon2 := 245, coordinate := 1, glue := 200 }
def finalOwnerCursor8 : OwnerCursor :=
  { row := 269445, canonical := 2, poseidon2 := 245, coordinate := 1, glue := 264 }
def finalOwnerCursor9 : OwnerCursor :=
  { row := 283286, canonical := 2, poseidon2 := 268, coordinate := 1, glue := 305 }
def finalOwnerCursor10 : OwnerCursor :=
  { row := 312701, canonical := 2, poseidon2 := 317, coordinate := 1, glue := 320 }
def finalOwnerCursor11 : OwnerCursor := OwnerCursor.finalFor finalArm

theorem finalOwnerChunk0_checked :
    finalOwnerChunk0.length = 64 ∧
      runOwnerPrefix finalArm finalOwnerCursor0 finalOwnerChunk0 =
        some finalOwnerCursor1 := by
  constructor <;> rfl

theorem finalOwnerChunk1_checked :
    finalOwnerChunk1.length = 64 ∧
      runOwnerPrefix finalArm finalOwnerCursor1 finalOwnerChunk1 =
        some finalOwnerCursor2 := by
  constructor <;> rfl

theorem finalOwnerChunk2_checked :
    finalOwnerChunk2.length = 64 ∧
      runOwnerPrefix finalArm finalOwnerCursor2 finalOwnerChunk2 =
        some finalOwnerCursor3 := by
  constructor <;> rfl

theorem finalOwnerChunk3_checked :
    finalOwnerChunk3.length = 64 ∧
      runOwnerPrefix finalArm finalOwnerCursor3 finalOwnerChunk3 =
        some finalOwnerCursor4 := by
  constructor <;> rfl

theorem finalOwnerChunk4_checked :
    finalOwnerChunk4.length = 64 ∧
      runOwnerPrefix finalArm finalOwnerCursor4 finalOwnerChunk4 =
        some finalOwnerCursor5 := by
  constructor <;> rfl

theorem finalOwnerChunk5_checked :
    finalOwnerChunk5.length = 64 ∧
      runOwnerPrefix finalArm finalOwnerCursor5 finalOwnerChunk5 =
        some finalOwnerCursor6 := by
  constructor <;> rfl

theorem finalOwnerChunk6_checked :
    finalOwnerChunk6.length = 64 ∧
      runOwnerPrefix finalArm finalOwnerCursor6 finalOwnerChunk6 =
        some finalOwnerCursor7 := by
  constructor <;> rfl

theorem finalOwnerChunk7_checked :
    finalOwnerChunk7.length = 64 ∧
      runOwnerPrefix finalArm finalOwnerCursor7 finalOwnerChunk7 =
        some finalOwnerCursor8 := by
  constructor <;> rfl

theorem finalOwnerChunk8_checked :
    finalOwnerChunk8.length = 64 ∧
      runOwnerPrefix finalArm finalOwnerCursor8 finalOwnerChunk8 =
        some finalOwnerCursor9 := by
  constructor <;> rfl

theorem finalOwnerChunk9_checked :
    finalOwnerChunk9.length = 64 ∧
      runOwnerPrefix finalArm finalOwnerCursor9 finalOwnerChunk9 =
        some finalOwnerCursor10 := by
  constructor <;> rfl

theorem finalOwnerTail10_checked :
    finalOwnerTail10.length = 61 ∧
      runOwnerPrefix finalArm finalOwnerCursor10 finalOwnerTail10 =
        some finalOwnerCursor11 := by
  constructor <;> rfl

theorem finalArm_ownership_valid : finalArm.OwnershipValid := by
  have checked10 := finalOwnerTail10_checked.2
  have checked9 := runOwnerPrefix_of_take_drop
    finalOwnerChunk9_checked.2 checked10
  have checked8 := runOwnerPrefix_of_take_drop
    finalOwnerChunk8_checked.2 checked9
  have checked7 := runOwnerPrefix_of_take_drop
    finalOwnerChunk7_checked.2 checked8
  have checked6 := runOwnerPrefix_of_take_drop
    finalOwnerChunk6_checked.2 checked7
  have checked5 := runOwnerPrefix_of_take_drop
    finalOwnerChunk5_checked.2 checked6
  have checked4 := runOwnerPrefix_of_take_drop
    finalOwnerChunk4_checked.2 checked5
  have checked3 := runOwnerPrefix_of_take_drop
    finalOwnerChunk3_checked.2 checked4
  have checked2 := runOwnerPrefix_of_take_drop
    finalOwnerChunk2_checked.2 checked3
  have checked1 := runOwnerPrefix_of_take_drop
    finalOwnerChunk1_checked.2 checked2
  have checked0 := runOwnerPrefix_of_take_drop
    finalOwnerChunk0_checked.2 checked1
  exact ownershipValid_of_run checked0

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFinalOwnershipCertificate
