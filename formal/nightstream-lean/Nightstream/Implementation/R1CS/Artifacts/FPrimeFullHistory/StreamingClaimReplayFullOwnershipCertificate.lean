import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayOwnershipCertificateSupport

/-!
Contract: exact owner-schedule certificate for the Rust-emitted full streaming
claim-replay arm.

Assurance tier: artifact-checked ownership certificate.

Owns all 876 owner records as fourteen bounded prefix certificates. Each leaf
checks row continuity, source-object identity, and the next per-family index.

Does not own row semantics or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFullOwnershipCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayOwnershipCertificateSupport

def fullOwnerTail0 := fullArm.owners
def fullOwnerChunk0 := fullOwnerTail0.take 64
def fullOwnerTail1 := fullOwnerTail0.drop 64
def fullOwnerChunk1 := fullOwnerTail1.take 64
def fullOwnerTail2 := fullOwnerTail1.drop 64
def fullOwnerChunk2 := fullOwnerTail2.take 64
def fullOwnerTail3 := fullOwnerTail2.drop 64
def fullOwnerChunk3 := fullOwnerTail3.take 64
def fullOwnerTail4 := fullOwnerTail3.drop 64
def fullOwnerChunk4 := fullOwnerTail4.take 64
def fullOwnerTail5 := fullOwnerTail4.drop 64
def fullOwnerChunk5 := fullOwnerTail5.take 64
def fullOwnerTail6 := fullOwnerTail5.drop 64
def fullOwnerChunk6 := fullOwnerTail6.take 64
def fullOwnerTail7 := fullOwnerTail6.drop 64
def fullOwnerChunk7 := fullOwnerTail7.take 64
def fullOwnerTail8 := fullOwnerTail7.drop 64
def fullOwnerChunk8 := fullOwnerTail8.take 64
def fullOwnerTail9 := fullOwnerTail8.drop 64
def fullOwnerChunk9 := fullOwnerTail9.take 64
def fullOwnerTail10 := fullOwnerTail9.drop 64
def fullOwnerChunk10 := fullOwnerTail10.take 64
def fullOwnerTail11 := fullOwnerTail10.drop 64
def fullOwnerChunk11 := fullOwnerTail11.take 64
def fullOwnerTail12 := fullOwnerTail11.drop 64
def fullOwnerChunk12 := fullOwnerTail12.take 64
def fullOwnerTail13 := fullOwnerTail12.drop 64

def fullOwnerCursor0 : OwnerCursor := OwnerCursor.zero
def fullOwnerCursor1 : OwnerCursor :=
  { row := 200, canonical := 2, poseidon2 := 0, coordinate := 0, glue := 62 }
def fullOwnerCursor2 : OwnerCursor :=
  { row := 264, canonical := 2, poseidon2 := 0, coordinate := 0, glue := 126 }
def fullOwnerCursor3 : OwnerCursor :=
  { row := 328, canonical := 2, poseidon2 := 0, coordinate := 0, glue := 190 }
def fullOwnerCursor4 : OwnerCursor :=
  { row := 13570, canonical := 2, poseidon2 := 22, coordinate := 0, glue := 232 }
def fullOwnerCursor5 : OwnerCursor :=
  { row := 51970, canonical := 2, poseidon2 := 86, coordinate := 0, glue := 232 }
def fullOwnerCursor6 : OwnerCursor :=
  { row := 90370, canonical := 2, poseidon2 := 150, coordinate := 0, glue := 232 }
def fullOwnerCursor7 : OwnerCursor :=
  { row := 128770, canonical := 2, poseidon2 := 214, coordinate := 0, glue := 232 }
def fullOwnerCursor8 : OwnerCursor :=
  { row := 160590, canonical := 2, poseidon2 := 256, coordinate := 1, glue := 253 }
def fullOwnerCursor9 : OwnerCursor :=
  { row := 160654, canonical := 2, poseidon2 := 256, coordinate := 1, glue := 317 }
def fullOwnerCursor10 : OwnerCursor :=
  { row := 233904, canonical := 2, poseidon2 := 256, coordinate := 2, glue := 380 }
def fullOwnerCursor11 : OwnerCursor :=
  { row := 233968, canonical := 2, poseidon2 := 256, coordinate := 2, glue := 444 }
def fullOwnerCursor12 : OwnerCursor :=
  { row := 257992, canonical := 2, poseidon2 := 296, coordinate := 2, glue := 468 }
def fullOwnerCursor13 : OwnerCursor :=
  { row := 287407, canonical := 2, poseidon2 := 345, coordinate := 2, glue := 483 }
def fullOwnerCursor14 : OwnerCursor := OwnerCursor.finalFor fullArm

theorem fullOwnerChunk0_checked :
    fullOwnerChunk0.length = 64 ∧
      runOwnerPrefix fullArm fullOwnerCursor0 fullOwnerChunk0 =
        some fullOwnerCursor1 := by
  constructor <;> rfl

theorem fullOwnerChunk1_checked :
    fullOwnerChunk1.length = 64 ∧
      runOwnerPrefix fullArm fullOwnerCursor1 fullOwnerChunk1 =
        some fullOwnerCursor2 := by
  constructor <;> rfl

theorem fullOwnerChunk2_checked :
    fullOwnerChunk2.length = 64 ∧
      runOwnerPrefix fullArm fullOwnerCursor2 fullOwnerChunk2 =
        some fullOwnerCursor3 := by
  constructor <;> rfl

theorem fullOwnerChunk3_checked :
    fullOwnerChunk3.length = 64 ∧
      runOwnerPrefix fullArm fullOwnerCursor3 fullOwnerChunk3 =
        some fullOwnerCursor4 := by
  constructor <;> rfl

theorem fullOwnerChunk4_checked :
    fullOwnerChunk4.length = 64 ∧
      runOwnerPrefix fullArm fullOwnerCursor4 fullOwnerChunk4 =
        some fullOwnerCursor5 := by
  constructor <;> rfl

theorem fullOwnerChunk5_checked :
    fullOwnerChunk5.length = 64 ∧
      runOwnerPrefix fullArm fullOwnerCursor5 fullOwnerChunk5 =
        some fullOwnerCursor6 := by
  constructor <;> rfl

theorem fullOwnerChunk6_checked :
    fullOwnerChunk6.length = 64 ∧
      runOwnerPrefix fullArm fullOwnerCursor6 fullOwnerChunk6 =
        some fullOwnerCursor7 := by
  constructor <;> rfl

theorem fullOwnerChunk7_checked :
    fullOwnerChunk7.length = 64 ∧
      runOwnerPrefix fullArm fullOwnerCursor7 fullOwnerChunk7 =
        some fullOwnerCursor8 := by
  constructor <;> rfl

theorem fullOwnerChunk8_checked :
    fullOwnerChunk8.length = 64 ∧
      runOwnerPrefix fullArm fullOwnerCursor8 fullOwnerChunk8 =
        some fullOwnerCursor9 := by
  constructor <;> rfl

theorem fullOwnerChunk9_checked :
    fullOwnerChunk9.length = 64 ∧
      runOwnerPrefix fullArm fullOwnerCursor9 fullOwnerChunk9 =
        some fullOwnerCursor10 := by
  constructor <;> rfl

theorem fullOwnerChunk10_checked :
    fullOwnerChunk10.length = 64 ∧
      runOwnerPrefix fullArm fullOwnerCursor10 fullOwnerChunk10 =
        some fullOwnerCursor11 := by
  constructor <;> rfl

theorem fullOwnerChunk11_checked :
    fullOwnerChunk11.length = 64 ∧
      runOwnerPrefix fullArm fullOwnerCursor11 fullOwnerChunk11 =
        some fullOwnerCursor12 := by
  constructor <;> rfl

theorem fullOwnerChunk12_checked :
    fullOwnerChunk12.length = 64 ∧
      runOwnerPrefix fullArm fullOwnerCursor12 fullOwnerChunk12 =
        some fullOwnerCursor13 := by
  constructor <;> rfl

theorem fullOwnerTail13_checked :
    fullOwnerTail13.length = 44 ∧
      runOwnerPrefix fullArm fullOwnerCursor13 fullOwnerTail13 =
        some fullOwnerCursor14 := by
  constructor <;> rfl

theorem fullArm_ownership_valid : fullArm.OwnershipValid := by
  have checked13 := fullOwnerTail13_checked.2
  have checked12 := runOwnerPrefix_of_take_drop
    fullOwnerChunk12_checked.2 checked13
  have checked11 := runOwnerPrefix_of_take_drop
    fullOwnerChunk11_checked.2 checked12
  have checked10 := runOwnerPrefix_of_take_drop
    fullOwnerChunk10_checked.2 checked11
  have checked9 := runOwnerPrefix_of_take_drop
    fullOwnerChunk9_checked.2 checked10
  have checked8 := runOwnerPrefix_of_take_drop
    fullOwnerChunk8_checked.2 checked9
  have checked7 := runOwnerPrefix_of_take_drop
    fullOwnerChunk7_checked.2 checked8
  have checked6 := runOwnerPrefix_of_take_drop
    fullOwnerChunk6_checked.2 checked7
  have checked5 := runOwnerPrefix_of_take_drop
    fullOwnerChunk5_checked.2 checked6
  have checked4 := runOwnerPrefix_of_take_drop
    fullOwnerChunk4_checked.2 checked5
  have checked3 := runOwnerPrefix_of_take_drop
    fullOwnerChunk3_checked.2 checked4
  have checked2 := runOwnerPrefix_of_take_drop
    fullOwnerChunk2_checked.2 checked3
  have checked1 := runOwnerPrefix_of_take_drop
    fullOwnerChunk1_checked.2 checked2
  have checked0 := runOwnerPrefix_of_take_drop
    fullOwnerChunk0_checked.2 checked1
  exact ownershipValid_of_run checked0

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFullOwnershipCertificate
