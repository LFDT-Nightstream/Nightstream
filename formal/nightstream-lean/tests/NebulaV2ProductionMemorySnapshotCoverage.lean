import Nightstream.Implementation.NebulaV2.ProductionMemorySnapshotCoverage

/-! Regression surface for row-derived complete snapshot reconstruction. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionMemorySnapshotCoverage

open Nightstream.Implementation.NebulaV2.ProductionMemorySnapshotCoverage

#check snapshotChunkSum_eq_flattenedLists
#check SegmentRun.stepAt_stepIndex
#check SegmentRun.stepAt_segmentBounds
#check SegmentRun.snapshotRecords_structural
#check SegmentRun.snapshotValidAt
#check SegmentRun.snapshotChunksCover

end tests.NebulaV2ProductionMemorySnapshotCoverage
