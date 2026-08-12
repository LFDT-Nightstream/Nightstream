import Nightstream.Implementation.NebulaV2.ProductionMemorySnapshotCoverage
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionMemorySnapshotCoverage

open Nightstream.Implementation.NebulaV2.ProductionMemorySnapshotCoverage

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemorySnapshotCoverage.snapshotChunkSum_eq_flattenedLists' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms snapshotChunkSum_eq_flattenedLists

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemorySnapshotCoverage.SegmentRun.snapshotRecords_structural' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.snapshotRecords_structural

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemorySnapshotCoverage.SegmentRun.snapshotValidAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.snapshotValidAt

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemorySnapshotCoverage.SegmentRun.snapshotChunksCover' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.snapshotChunksCover

end tests.Axioms.NebulaV2ProductionMemorySnapshotCoverage
