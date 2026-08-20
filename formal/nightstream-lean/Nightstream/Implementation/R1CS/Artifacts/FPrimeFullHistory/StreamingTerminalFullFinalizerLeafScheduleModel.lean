import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer
import Nightstream.Implementation.R1CS.Core.SeededAjtai

/-! Verifier-owned schedules and exact Rust schedule identities for the terminal Nebula leaves. -/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFullFinalizerLeafScheduleCertificate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer

def primaryMasterSeed : List Nat := List.replicate 32 0xC5
def compressionMasterSeed : List Nat := List.replicate 32 0xC6

def expectedPrimarySchedule : SeededPhi81.SeedSchedule :=
  SeededAjtai.schedule primaryMasterSeed 2 745 16

def expectedCompressionSchedule : SeededPhi81.SeedSchedule :=
  SeededAjtai.schedule compressionMasterSeed 1 82 16

theorem primary_schedule_exact :
    leafPrimarySchedule = expectedPrimarySchedule := by
  rfl

theorem compression_schedule_exact :
    leafCompressionSchedule = expectedCompressionSchedule := by
  rfl

def primarySeed0 : List Nat :=
  [168, 53, 101, 80, 39, 69, 99, 82, 125, 247, 4, 243, 125, 167,
    242, 8, 176, 109, 23, 92, 152, 84, 227, 30, 214, 25, 102, 175,
    218, 226, 86, 119]

def primarySeed1 : List Nat :=
  [54, 240, 30, 75, 79, 78, 8, 196, 159, 204, 247, 160, 177, 173,
    62, 112, 38, 220, 238, 219, 79, 66, 7, 194, 189, 222, 96, 105,
    242, 116, 35, 162]

def compressionSeed : List Nat :=
  [222, 97, 160, 75, 169, 146, 205, 28, 66, 7, 37, 46, 38, 226,
    240, 160, 130, 181, 109, 118, 6, 248, 19, 168, 202, 255, 83, 20,
    122, 228, 97, 38]

theorem primary_seeds_exact :
    leafPrimarySchedule.seedsByOutput = [[primarySeed0], [primarySeed1]] := by
  rfl

theorem compression_seed_exact :
    leafCompressionSchedule.seedsByOutput = [[compressionSeed]] := by
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFullFinalizerLeafScheduleCertificate
