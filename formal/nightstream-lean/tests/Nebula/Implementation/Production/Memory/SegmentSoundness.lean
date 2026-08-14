import Nightstream.Implementation.Nebula.Production.Memory.SegmentSoundness

/-! Regression surface for non-circular row-derived memory soundness. -/

set_option autoImplicit false

namespace tests.NebulaProductionMemorySegmentSoundness

open Nightstream.Implementation.Nebula.ProductionMemorySegmentSoundness

#check SegmentRun.orderedActiveToClosed
#check SegmentRun.covers
#check SegmentRun.openingProductsConcrete
#check SegmentRun.fingerprintAccepted
#check SegmentRun.balanceOrEvaluationFailure
#check SegmentRun.executesOrEvaluationFailure

end tests.NebulaProductionMemorySegmentSoundness
