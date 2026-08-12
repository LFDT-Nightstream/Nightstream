import Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry

set_option autoImplicit false

namespace tests.NebulaV2ConcreteLaneGeometry

open Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry
open Nightstream.Protocol.NebulaV2.LaneLayout

def exactBlockPlacement : Placement blockWidth where
  base := 0
  assignmentAligned := blockWidth_aligned
  baseAligned := by norm_num [Aligned, ringDegree]
  blockWithin := by simp

theorem exact_relative_starts :
    exactBlockPlacement.layout.operationsStart = 0 ∧
    exactBlockPlacement.layout.initialSnapshotStart = 6696 ∧
    exactBlockPlacement.layout.finalSnapshotStart = 10260 := by
  norm_num [exactBlockPlacement, Placement.layout,
    initialSnapshotRelativeStart, finalSnapshotRelativeStart,
    operationsLaneWidth_exact, snapshotLaneWidth_exact]

theorem exact_padding_is_small_and_canonicalizable :
    operationAlignmentPadding = 18 ∧ snapshotAlignmentPadding = 44 := by
  exact ⟨operationAlignmentPadding_exact, snapshotAlignmentPadding_exact⟩

/-- A generated assignment can place the block after one complete ring
column without changing its relative geometry. -/
def translatedPlacement : Placement (ringDegree + blockWidth) where
  base := ringDegree
  assignmentAligned := aligned_add (by rfl) blockWidth_aligned
  baseAligned := by rfl
  blockWithin := by omega

theorem translated_starts_remain_aligned :
    Aligned translatedPlacement.layout.operationsStart ∧
    Aligned translatedPlacement.layout.initialSnapshotStart ∧
    Aligned translatedPlacement.layout.finalSnapshotStart := by
  exact ⟨translatedPlacement.layout.operationsStartAligned,
    translatedPlacement.layout.initialSnapshotStartAligned,
    translatedPlacement.layout.finalSnapshotStartAligned⟩

end tests.NebulaV2ConcreteLaneGeometry
