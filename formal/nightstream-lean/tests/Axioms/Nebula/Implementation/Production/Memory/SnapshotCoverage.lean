import Nightstream.Implementation.Nebula.Production.Memory.SnapshotCoverage
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionMemorySnapshotCoverage

open Nightstream.Implementation.Nebula.ProductionMemorySnapshotCoverage

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemorySnapshotCoverage.snapshotChunkSum_eq_flattenedLists' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms snapshotChunkSum_eq_flattenedLists

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemorySnapshotCoverage.SegmentRun.snapshotRecords_structural' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.snapshotRecords_structural

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemorySnapshotCoverage.SegmentRun.snapshotValidAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.snapshotValidAt

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemorySnapshotCoverage.SegmentRun.snapshotChunksCover' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.snapshotChunksCover

end tests.Axioms.NebulaProductionMemorySnapshotCoverage
