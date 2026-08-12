import Nightstream.Protocol.NebulaV2.LaneLayout

set_option autoImplicit false

namespace tests.NebulaV2LaneLayout

open Nightstream.Protocol.NebulaV2.LaneLayout

def layout : Layout 216 54 54 where
  operationsStart := 54
  initialSnapshotStart := 108
  finalSnapshotStart := 162
  assignmentAligned := by norm_num [Aligned, ringDegree]
  operationsStartAligned := by norm_num [Aligned, ringDegree]
  operationsWidthAligned := by norm_num [Aligned, ringDegree]
  initialSnapshotStartAligned := by norm_num [Aligned, ringDegree]
  snapshotWidthAligned := by norm_num [Aligned, ringDegree]
  finalSnapshotStartAligned := by norm_num [Aligned, ringDegree]
  operationsWithin := by decide
  initialSnapshotWithin := by decide
  finalSnapshotWithin := by decide
  operationsInitialDisjoint := by norm_num [DisjointRanges]
  operationsFinalDisjoint := by norm_num [DisjointRanges]
  snapshotsDisjoint := by norm_num [DisjointRanges]

theorem operations_lane_reads_the_exact_full_assignment_coordinate
    (assignment : Fin 216 → Nat) (index : Fin 54) :
    layout.operationsProjection assignment index =
      assignment ⟨54 + index.val, by have := index.isLt; omega⟩ :=
  rfl

/-- A one-coordinate offset cannot satisfy the whole-ring alignment rule. -/
theorem offset_one_is_rejected : ¬ Aligned 1 := by
  norm_num [Aligned, ringDegree]

/-- Overlapping snapshot ranges cannot form a V2 layout. -/
theorem overlapping_snapshots_are_rejected :
    ¬ DisjointRanges 108 54 108 54 := by
  norm_num [DisjointRanges]

end tests.NebulaV2LaneLayout
