import Nightstream.Implementation.Nebula.Production.Memory.RowSegments

/-! Regression surface for exact row-derived delayed F-prime segments. -/

set_option autoImplicit false

namespace tests.NebulaProductionMemoryRowSegments

open Nightstream.Implementation.Nebula.ProductionMemoryRowSegments

#check BatchRun.toStepRun
#check BatchRun.toVerifiedRun
#check BatchRun.claimsExact
#check SegmentRun.toProtocol
#check SegmentRun.exactBatchCount
#check SegmentRun.exactStepCount
#check SegmentRun.stepIndexAt
#check SegmentRun.segmentBoundsAt
#check Chain.exactBatchCount
#check delayedRun_to_rowSegmentChain

end tests.NebulaProductionMemoryRowSegments
