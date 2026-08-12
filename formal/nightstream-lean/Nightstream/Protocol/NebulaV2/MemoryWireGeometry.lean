import Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry
import Nightstream.Protocol.NebulaV2.Digest

/-!
Contract: exact logical wire widths for V2 memory state and fresh claims.

Assurance tier: model-level.

Owns the fixed counter, challenge, product, root, public-suffix, ring-padding,
carry, and mandatory commitment-bundle widths.

Does not own the generated absolute offsets, application-public fields, NIFS
proof width, running-claim width, compiler capacity, or recursive-size
closure.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.MemoryWireGeometry

open Nightstream.Protocol.NebulaV2.LaneLayout

def baseFieldBitCount : Nat := 64
def extensionLimbCount : Nat := 2
def digestLimbCount : Nat := 4
def repetitionCount : Nat := 2
def challengeElementsPerRepetition : Nat := 2
def productsPerRepetition : Nat := 4

def segmentIndexBits : Nat := 7
def stepIndexBits : Nat := 11
def timestampBits : Nat := 23
def stepActiveAccessCountBits : Nat := 6
def segmentActiveAccessCountBits : Nat := 17
def phaseBits : Nat := 1

def challengeBaseFieldLimbs : Nat :=
  repetitionCount * challengeElementsPerRepetition * extensionLimbCount

def productStateBaseFieldLimbs : Nat :=
  repetitionCount * productsPerRepetition * extensionLimbCount

def rootsBaseFieldLimbs : Nat := 3 * digestLimbCount

/-- Segment index, step index, four timestamps, and one per-step active
access count. The four timestamps are input, output, segment start, and
declared segment end. -/
def stepCounterBits : Nat :=
  segmentIndexBits + stepIndexBits + 4 * timestampBits +
    stepActiveAccessCountBits

def stepChallengeBits : Nat := challengeBaseFieldLimbs * baseFieldBitCount
def stepProductBits : Nat := 2 * productStateBaseFieldLimbs * baseFieldBitCount

/-- `D_pre`, `D_seen_before`, and `D_seen_after`; each value has three
digests. -/
def stepRootBits : Nat := 3 * rootsBaseFieldLimbs * baseFieldBitCount

/-- Exact public memory block in each fresh claim. -/
def stepPublicBits : Nat :=
  stepCounterBits + stepChallengeBits + stepProductBits + stepRootBits

def stepPublicWithCcsOne : Nat := 1 + stepPublicBits
def stepPublicRingWidth : Nat :=
  ((stepPublicWithCcsOne + ringDegree - 1) / ringDegree) * ringDegree
def stepPublicRingPadding : Nat := stepPublicRingWidth - stepPublicWithCcsOne

/-- Phase, segment and step counters, global/start/end timestamps, and the
segment-wide active-access count. -/
def carryCounterBits : Nat :=
  phaseBits + segmentIndexBits + stepIndexBits + 3 * timestampBits +
    segmentActiveAccessCountBits

def carryChallengeBits : Nat := challengeBaseFieldLimbs * baseFieldBitCount
def carryProductBits : Nat := productStateBaseFieldLimbs * baseFieldBitCount

/-- `D_pre`, `D_seen`, and `D_memory`: seven digests in total. -/
def carryRootBits : Nat := 7 * digestLimbCount * baseFieldBitCount

def carryBits : Nat :=
  carryCounterBits + carryChallengeBits + carryProductBits + carryRootBits

def commitmentRank : Nat := 18
def commitmentFieldCount : Nat := commitmentRank * ringDegree
def bundleComponentCount : Nat := 4

/-- Full, operations, initial-snapshot, and final-snapshot commitments, with
every canonical base-field coordinate represented by 64 bits. -/
def mandatoryBundleBits : Nat :=
  bundleComponentCount * commitmentFieldCount * baseFieldBitCount

theorem challengeBaseFieldLimbs_exact : challengeBaseFieldLimbs = 8 := by
  decide

theorem productStateBaseFieldLimbs_exact :
    productStateBaseFieldLimbs = 16 := by
  decide

theorem stepCounterBits_exact : stepCounterBits = 116 := by
  decide

theorem stepChallengeBits_exact : stepChallengeBits = 512 := by
  decide

theorem stepProductBits_exact : stepProductBits = 2048 := by
  decide

theorem stepRootBits_exact : stepRootBits = 2304 := by
  decide

theorem stepPublicBits_exact : stepPublicBits = 4980 := by
  decide

theorem stepPublicRingWidth_exact : stepPublicRingWidth = 5022 := by
  decide

theorem stepPublicRingPadding_exact : stepPublicRingPadding = 41 := by
  decide

theorem stepPublicRingColumns_exact : stepPublicRingWidth / ringDegree = 93 := by
  decide

theorem stepPublicRingWidth_aligned : Aligned stepPublicRingWidth := by
  norm_num [Aligned, stepPublicRingWidth_exact, ringDegree]

theorem carryCounterBits_exact : carryCounterBits = 105 := by
  decide

theorem carryChallengeBits_exact : carryChallengeBits = 512 := by
  decide

theorem carryProductBits_exact : carryProductBits = 1024 := by
  decide

theorem carryRootBits_exact : carryRootBits = 1792 := by
  decide

theorem carryBits_exact : carryBits = 3433 := by
  decide

theorem commitmentFieldCount_exact : commitmentFieldCount = 972 := by
  decide

theorem mandatoryBundleBits_exact : mandatoryBundleBits = 248832 := by
  decide

end Nightstream.Protocol.NebulaV2.MemoryWireGeometry
