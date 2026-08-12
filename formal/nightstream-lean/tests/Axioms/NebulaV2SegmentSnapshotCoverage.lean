import Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage.snapshotList_coe_eq_chunkSnapshot' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms snapshotList_coe_eq_chunkSnapshot

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage.snapshotChunkSum_eq_flattenedLists' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms snapshotChunkSum_eq_flattenedLists

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage.CheckedRun.invocationAt_stepIndex' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedRun.invocationAt_stepIndex

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage.CheckedRun.invocationAt_segmentBounds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedRun.invocationAt_segmentBounds

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage.CheckedRun.invocationAt_boundaryValue' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedRun.invocationAt_boundaryValue

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage.CheckedRun.snapshotRecords_structural' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedRun.snapshotRecords_structural

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage.CheckedRun.snapshotValidAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedRun.snapshotValidAt

/-- info: 'Nightstream.Implementation.NebulaV2.SegmentSnapshotCoverage.CheckedRun.snapshotChunksCover' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedRun.snapshotChunksCover
