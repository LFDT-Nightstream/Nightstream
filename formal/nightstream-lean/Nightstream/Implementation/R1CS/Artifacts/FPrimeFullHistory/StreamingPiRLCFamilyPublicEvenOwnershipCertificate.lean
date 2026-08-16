import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicOwnershipCertificateSupport

/-!
Contract: first bounded owner-schedule certificate for the Rust-emitted even
PiRLC public-family arm.

Owns five 64-owner prefix leaves. Each leaf checks row continuity,
source-object identity, and the next per-family index.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenOwnershipCertificate

open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOwnershipCertificateSupport

def evenOwnerTail0 := evenArm.owners
def evenOwnerChunk0 := evenOwnerTail0.take 64
def evenOwnerTail1 := evenOwnerTail0.drop 64
def evenOwnerChunk1 := evenOwnerTail1.take 64
def evenOwnerTail2 := evenOwnerTail1.drop 64
def evenOwnerChunk2 := evenOwnerTail2.take 64
def evenOwnerTail3 := evenOwnerTail2.drop 64
def evenOwnerChunk3 := evenOwnerTail3.take 64
def evenOwnerTail4 := evenOwnerTail3.drop 64
def evenOwnerChunk4 := evenOwnerTail4.take 64
def evenOwnerTail5 := evenOwnerTail4.drop 64
def evenOwnerChunk5 := evenOwnerTail5.take 64
def evenOwnerTail6 := evenOwnerTail5.drop 64
def evenOwnerChunk6 := evenOwnerTail6.take 64
def evenOwnerTail7 := evenOwnerTail6.drop 64
def evenOwnerChunk7 := evenOwnerTail7.take 64
def evenOwnerTail8 := evenOwnerTail7.drop 64
def evenOwnerChunk8 := evenOwnerTail8.take 64
def evenOwnerTail9 := evenOwnerTail8.drop 64

def evenOwnerCursor0 : OwnerCursor := OwnerCursor.startFor evenArm
def evenOwnerCursor1 : OwnerCursor :=
  { row := 301562, canonical := 2, poseidon2 := 44, glue := 18,
    phaseEnvelope := 0 }
def evenOwnerCursor2 : OwnerCursor :=
  { row := 339962, canonical := 2, poseidon2 := 108, glue := 18,
    phaseEnvelope := 0 }
def evenOwnerCursor3 : OwnerCursor :=
  { row := 378362, canonical := 2, poseidon2 := 172, glue := 18,
    phaseEnvelope := 0 }
def evenOwnerCursor4 : OwnerCursor :=
  { row := 416163, canonical := 2, poseidon2 := 235, glue := 19,
    phaseEnvelope := 0 }
def evenOwnerCursor5 : OwnerCursor :=
  { row := 446177, canonical := 2, poseidon2 := 285, glue := 33,
    phaseEnvelope := 0 }
def evenOwnerCursor6 : OwnerCursor :=
  { row := 484577, canonical := 2, poseidon2 := 349, glue := 33,
    phaseEnvelope := 0 }
def evenOwnerCursor7 : OwnerCursor :=
  { row := 522977, canonical := 2, poseidon2 := 413, glue := 33,
    phaseEnvelope := 0 }
def evenOwnerCursor8 : OwnerCursor :=
  { row := 1221352, canonical := 2, poseidon2 := 472, glue := 37,
    phaseEnvelope := 1 }
def evenOwnerCursor9 : OwnerCursor :=
  { row := 1227147, canonical := 7, poseidon2 := 481, glue := 87,
    phaseEnvelope := 1 }
def evenOwnerCursor10 : OwnerCursor := OwnerCursor.finalFor evenArm

theorem evenOwnerChunk0_checked :
    evenOwnerChunk0.length = 64 ∧
      runOwnerPrefix evenArm evenOwnerCursor0 evenOwnerChunk0 =
        some evenOwnerCursor1 := by
  constructor <;> rfl

theorem evenOwnerChunk1_checked :
    evenOwnerChunk1.length = 64 ∧
      runOwnerPrefix evenArm evenOwnerCursor1 evenOwnerChunk1 =
        some evenOwnerCursor2 := by
  constructor <;> rfl

theorem evenOwnerChunk2_checked :
    evenOwnerChunk2.length = 64 ∧
      runOwnerPrefix evenArm evenOwnerCursor2 evenOwnerChunk2 =
        some evenOwnerCursor3 := by
  constructor <;> rfl

theorem evenOwnerChunk3_checked :
    evenOwnerChunk3.length = 64 ∧
      runOwnerPrefix evenArm evenOwnerCursor3 evenOwnerChunk3 =
        some evenOwnerCursor4 := by
  constructor <;> rfl

theorem evenOwnerChunk4_checked :
    evenOwnerChunk4.length = 64 ∧
      runOwnerPrefix evenArm evenOwnerCursor4 evenOwnerChunk4 =
        some evenOwnerCursor5 := by
  constructor <;> rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenOwnershipCertificate
