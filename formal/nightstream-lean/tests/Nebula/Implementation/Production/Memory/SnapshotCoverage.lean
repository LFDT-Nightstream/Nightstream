import Nightstream.Implementation.Nebula.Production.Memory.SnapshotCoverage

/-! Regression surface for row-derived complete snapshot reconstruction. -/

set_option autoImplicit false

namespace tests.NebulaProductionMemorySnapshotCoverage

open Nightstream.Implementation.Nebula.ProductionMemorySnapshotCoverage

#check snapshotChunkSum_eq_flattenedLists
#check SegmentRun.stepAt_stepIndex
#check SegmentRun.stepAt_segmentBounds
#check SegmentRun.snapshotRecords_structural
#check SegmentRun.snapshotValidAt
#check SegmentRun.snapshotChunksCover

end tests.NebulaProductionMemorySnapshotCoverage
