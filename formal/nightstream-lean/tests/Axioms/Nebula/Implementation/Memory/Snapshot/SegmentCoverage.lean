import Nightstream.Implementation.Nebula.Memory.Snapshot.SegmentCoverage
import tests.Axioms.Support

open Nightstream.Implementation.Nebula.SegmentSnapshotCoverage

/-- info: 'Nightstream.Implementation.Nebula.SegmentSnapshotCoverage.snapshotList_coe_eq_chunkSnapshot' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms snapshotList_coe_eq_chunkSnapshot

/-- info: 'Nightstream.Implementation.Nebula.SegmentSnapshotCoverage.snapshotChunkSum_eq_flattenedLists' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms snapshotChunkSum_eq_flattenedLists

/-- info: 'Nightstream.Implementation.Nebula.SegmentSnapshotCoverage.CheckedRun.invocationAt_stepIndex' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedRun.invocationAt_stepIndex

/-- info: 'Nightstream.Implementation.Nebula.SegmentSnapshotCoverage.CheckedRun.invocationAt_segmentBounds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedRun.invocationAt_segmentBounds

/-- info: 'Nightstream.Implementation.Nebula.SegmentSnapshotCoverage.CheckedRun.invocationAt_boundaryValue' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedRun.invocationAt_boundaryValue

/-- info: 'Nightstream.Implementation.Nebula.SegmentSnapshotCoverage.CheckedRun.snapshotRecords_structural' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedRun.snapshotRecords_structural

/-- info: 'Nightstream.Implementation.Nebula.SegmentSnapshotCoverage.CheckedRun.snapshotValidAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedRun.snapshotValidAt

/-- info: 'Nightstream.Implementation.Nebula.SegmentSnapshotCoverage.CheckedRun.snapshotChunksCover' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedRun.snapshotChunksCover
