import Nightstream.Protocol.NebulaV2.LaneLayout

/-!
Contract: exact relative lane geometry for `PaddedRowIdentityMemoryV2`.

Assurance tier: model-level.

Owns the V2 record widths, whole-ring padding, relative lane order, and the
construction of a generic `LaneLayout.Layout` at an aligned location in a
complete assignment.

Does not own the generated assignment width or the final absolute placement.
Those values belong to the generated relation manifest.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry

open Nightstream.Protocol.NebulaV2.LaneLayout

def operationSlots : Nat := 63
def scanSlots : Nat := 64

def padBits : Nat := 1
def isWriteBits : Nat := 1
def isRamBits : Nat := 1
def addressBits : Nat := 16
def valueBits : Nat := 32
def timestampBits : Nat := 23

/-- `pad, is_write, is_ram, address, read_value, write_value,
read_timestamp`. -/
def operationSlotPayloadWidth : Nat :=
  padBits + isWriteBits + isRamBits + addressBits +
    valueBits + valueBits + timestampBits

def operationPayloadWidth : Nat :=
  operationSlots * operationSlotPayloadWidth

/-- The smallest whole-ring width that contains the operation payload. -/
def operationsLaneWidth : Nat :=
  ((operationPayloadWidth + ringDegree - 1) / ringDegree) * ringDegree

def snapshotSlotPayloadWidth : Nat := valueBits + timestampBits
def snapshotPayloadWidth : Nat := scanSlots * snapshotSlotPayloadWidth

/-- The smallest whole-ring width that contains one snapshot chunk. -/
def snapshotLaneWidth : Nat :=
  ((snapshotPayloadWidth + ringDegree - 1) / ringDegree) * ringDegree

def operationAlignmentPadding : Nat :=
  operationsLaneWidth - operationPayloadWidth

def snapshotAlignmentPadding : Nat :=
  snapshotLaneWidth - snapshotPayloadWidth

/-- Relative order is operations, initial snapshot, final snapshot. -/
def operationsRelativeStart : Nat := 0
def initialSnapshotRelativeStart : Nat := operationsLaneWidth
def finalSnapshotRelativeStart : Nat := operationsLaneWidth + snapshotLaneWidth
def blockWidth : Nat := operationsLaneWidth + 2 * snapshotLaneWidth

theorem operationSlotPayloadWidth_exact : operationSlotPayloadWidth = 106 := by
  norm_num [operationSlotPayloadWidth, padBits, isWriteBits, isRamBits,
    addressBits, valueBits, timestampBits]

theorem operationPayloadWidth_exact : operationPayloadWidth = 6678 := by
  norm_num [operationPayloadWidth, operationSlots,
    operationSlotPayloadWidth_exact]

theorem operationsLaneWidth_exact : operationsLaneWidth = 6696 := by
  norm_num [operationsLaneWidth, operationPayloadWidth_exact, ringDegree]

theorem operationAlignmentPadding_exact :
    operationAlignmentPadding = 18 := by
  norm_num [operationAlignmentPadding, operationsLaneWidth_exact,
    operationPayloadWidth_exact]

theorem operationRingColumns_exact : operationsLaneWidth / ringDegree = 124 := by
  norm_num [operationsLaneWidth_exact, ringDegree]

theorem snapshotSlotPayloadWidth_exact : snapshotSlotPayloadWidth = 55 := by
  norm_num [snapshotSlotPayloadWidth, valueBits, timestampBits]

theorem snapshotPayloadWidth_exact : snapshotPayloadWidth = 3520 := by
  norm_num [snapshotPayloadWidth, scanSlots, snapshotSlotPayloadWidth_exact]

theorem snapshotLaneWidth_exact : snapshotLaneWidth = 3564 := by
  norm_num [snapshotLaneWidth, snapshotPayloadWidth_exact, ringDegree]

theorem snapshotAlignmentPadding_exact : snapshotAlignmentPadding = 44 := by
  norm_num [snapshotAlignmentPadding, snapshotLaneWidth_exact,
    snapshotPayloadWidth_exact]

theorem snapshotRingColumns_exact : snapshotLaneWidth / ringDegree = 66 := by
  norm_num [snapshotLaneWidth_exact, ringDegree]

theorem blockWidth_exact : blockWidth = 13824 := by
  norm_num [blockWidth, operationsLaneWidth_exact, snapshotLaneWidth_exact]

theorem blockRingColumns_exact : blockWidth / ringDegree = 256 := by
  norm_num [blockWidth_exact, ringDegree]

theorem operationsLaneWidth_aligned : Aligned operationsLaneWidth := by
  norm_num [Aligned, operationsLaneWidth_exact, ringDegree]

theorem snapshotLaneWidth_aligned : Aligned snapshotLaneWidth := by
  norm_num [Aligned, snapshotLaneWidth_exact, ringDegree]

theorem blockWidth_aligned : Aligned blockWidth := by
  norm_num [Aligned, blockWidth_exact, ringDegree]

theorem aligned_add {left right : Nat}
    (leftAligned : Aligned left) (rightAligned : Aligned right) :
    Aligned (left + right) := by
  unfold Aligned at leftAligned rightAligned ⊢
  rw [Nat.add_mod, leftAligned, rightAligned]
  simp

/-- The generated relation selects the absolute block start and proves that
the complete block is inside its verifier-key-bound assignment. -/
structure Placement (assignmentWidth : Nat) where
  base : Nat
  assignmentAligned : Aligned assignmentWidth
  baseAligned : Aligned base
  blockWithin : base + blockWidth ≤ assignmentWidth

namespace Placement

variable {assignmentWidth : Nat}

def layout (placement : Placement assignmentWidth) :
    Layout assignmentWidth operationsLaneWidth snapshotLaneWidth where
  operationsStart := placement.base
  initialSnapshotStart := placement.base + initialSnapshotRelativeStart
  finalSnapshotStart := placement.base + finalSnapshotRelativeStart
  assignmentAligned := placement.assignmentAligned
  operationsStartAligned := placement.baseAligned
  operationsWidthAligned := operationsLaneWidth_aligned
  initialSnapshotStartAligned := by
    exact aligned_add placement.baseAligned operationsLaneWidth_aligned
  snapshotWidthAligned := snapshotLaneWidth_aligned
  finalSnapshotStartAligned := by
    simpa [finalSnapshotRelativeStart, Nat.add_assoc] using
      aligned_add
        (aligned_add placement.baseAligned operationsLaneWidth_aligned)
        snapshotLaneWidth_aligned
  operationsWithin := by
    have within := placement.blockWithin
    simp only [blockWidth] at within
    omega
  initialSnapshotWithin := by
    have within := placement.blockWithin
    simp only [blockWidth, initialSnapshotRelativeStart] at within ⊢
    omega
  finalSnapshotWithin := by
    have within := placement.blockWithin
    simp only [blockWidth, finalSnapshotRelativeStart] at within ⊢
    omega
  operationsInitialDisjoint := by
    left
    simp [initialSnapshotRelativeStart]
  operationsFinalDisjoint := by
    left
    simp [finalSnapshotRelativeStart]
  snapshotsDisjoint := by
    left
    simp [initialSnapshotRelativeStart, finalSnapshotRelativeStart,
      Nat.add_assoc]

theorem layout_operationsStart (placement : Placement assignmentWidth) :
    placement.layout.operationsStart = placement.base := rfl

theorem layout_initialSnapshotStart (placement : Placement assignmentWidth) :
    placement.layout.initialSnapshotStart = placement.base + operationsLaneWidth :=
  rfl

theorem layout_finalSnapshotStart (placement : Placement assignmentWidth) :
    placement.layout.finalSnapshotStart =
      placement.base + operationsLaneWidth + snapshotLaneWidth :=
  by simp [Placement.layout, finalSnapshotRelativeStart, Nat.add_assoc]

end Placement

end Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry
