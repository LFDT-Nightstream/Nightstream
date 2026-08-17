import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicOwnershipCertificateSupport

/-!
Contract: first bounded owner-schedule certificate for the Rust-emitted odd
PiRLC public-family arm.

Owns five 64-owner prefix leaves. Each leaf checks row continuity,
source-object identity, and the next per-family index.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddOwnershipCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOwnershipCertificateSupport

def oddOwnerTail0 := oddArm.owners
def oddOwnerChunk0 := oddOwnerTail0.take 64
def oddOwnerTail1 := oddOwnerTail0.drop 64
def oddOwnerChunk1 := oddOwnerTail1.take 64
def oddOwnerTail2 := oddOwnerTail1.drop 64
def oddOwnerChunk2 := oddOwnerTail2.take 64
def oddOwnerTail3 := oddOwnerTail2.drop 64
def oddOwnerChunk3 := oddOwnerTail3.take 64
def oddOwnerTail4 := oddOwnerTail3.drop 64
def oddOwnerChunk4 := oddOwnerTail4.take 64
def oddOwnerTail5 := oddOwnerTail4.drop 64
def oddOwnerChunk5 := oddOwnerTail5.take 64
def oddOwnerTail6 := oddOwnerTail5.drop 64
def oddOwnerChunk6 := oddOwnerTail6.take 64
def oddOwnerTail7 := oddOwnerTail6.drop 64
def oddOwnerChunk7 := oddOwnerTail7.take 64
def oddOwnerTail8 := oddOwnerTail7.drop 64
def oddOwnerChunk8 := oddOwnerTail8.take 64
def oddOwnerTail9 := oddOwnerTail8.drop 64
def oddOwnerChunk9 := oddOwnerTail9.take 64
def oddOwnerTail10 := oddOwnerTail9.drop 64

def oddOwnerCursor0 : OwnerCursor := OwnerCursor.startFor oddArm
def oddOwnerCursor1 : OwnerCursor :=
  { row := 338402, canonical := 2, poseidon2 := 44, glue := 18,
    phaseEnvelope := 0 }
def oddOwnerCursor2 : OwnerCursor :=
  { row := 376802, canonical := 2, poseidon2 := 108, glue := 18,
    phaseEnvelope := 0 }
def oddOwnerCursor3 : OwnerCursor :=
  { row := 415202, canonical := 2, poseidon2 := 172, glue := 18,
    phaseEnvelope := 0 }
def oddOwnerCursor4 : OwnerCursor :=
  { row := 453602, canonical := 2, poseidon2 := 236, glue := 18,
    phaseEnvelope := 0 }
def oddOwnerCursor5 : OwnerCursor :=
  { row := 483017, canonical := 2, poseidon2 := 285, glue := 33,
    phaseEnvelope := 0 }
def oddOwnerCursor6 : OwnerCursor :=
  { row := 521417, canonical := 2, poseidon2 := 349, glue := 33,
    phaseEnvelope := 0 }
def oddOwnerCursor7 : OwnerCursor :=
  { row := 559817, canonical := 2, poseidon2 := 413, glue := 33,
    phaseEnvelope := 0 }
def oddOwnerCursor8 : OwnerCursor :=
  { row := 598217, canonical := 2, poseidon2 := 477, glue := 33,
    phaseEnvelope := 0 }
def oddOwnerCursor9 : OwnerCursor :=
  { row := 1290670, canonical := 3, poseidon2 := 526, glue := 46,
    phaseEnvelope := 1 }
def oddOwnerCursor10 : OwnerCursor :=
  { row := 1296996, canonical := 7, poseidon2 := 536, glue := 96,
    phaseEnvelope := 1 }
def oddOwnerCursor11 : OwnerCursor := OwnerCursor.finalFor oddArm

theorem oddOwnerChunk0_checked :
    oddOwnerChunk0.length = 64 ∧
      runOwnerPrefix oddArm oddOwnerCursor0 oddOwnerChunk0 =
        some oddOwnerCursor1 := by
  constructor <;> rfl

theorem oddOwnerChunk1_checked :
    oddOwnerChunk1.length = 64 ∧
      runOwnerPrefix oddArm oddOwnerCursor1 oddOwnerChunk1 =
        some oddOwnerCursor2 := by
  constructor <;> rfl

theorem oddOwnerChunk2_checked :
    oddOwnerChunk2.length = 64 ∧
      runOwnerPrefix oddArm oddOwnerCursor2 oddOwnerChunk2 =
        some oddOwnerCursor3 := by
  constructor <;> rfl

theorem oddOwnerChunk3_checked :
    oddOwnerChunk3.length = 64 ∧
      runOwnerPrefix oddArm oddOwnerCursor3 oddOwnerChunk3 =
        some oddOwnerCursor4 := by
  constructor <;> rfl

theorem oddOwnerChunk4_checked :
    oddOwnerChunk4.length = 64 ∧
      runOwnerPrefix oddArm oddOwnerCursor4 oddOwnerChunk4 =
        some oddOwnerCursor5 := by
  constructor <;> rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddOwnershipCertificate
