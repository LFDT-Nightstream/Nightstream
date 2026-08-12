import Nightstream.Implementation.NebulaV2.Production.Memory.SegmentSoundness

/-! Regression surface for non-circular row-derived memory soundness. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionMemorySegmentSoundness

open Nightstream.Implementation.NebulaV2.ProductionMemorySegmentSoundness

#check SegmentRun.orderedActiveToClosed
#check SegmentRun.covers
#check SegmentRun.openingProductsConcrete
#check SegmentRun.fingerprintAccepted
#check SegmentRun.balanceOrEvaluationFailure
#check SegmentRun.executesOrEvaluationFailure

end tests.NebulaV2ProductionMemorySegmentSoundness
