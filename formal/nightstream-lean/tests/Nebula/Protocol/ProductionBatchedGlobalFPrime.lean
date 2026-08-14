import Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrime

/-! Regression surface for the candidate-specific global delayed chain. -/

set_option autoImplicit false

namespace tests.NebulaProductionBatchedGlobalFPrime

open Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrime

#check SegmentRun.exactClaimCount
#check SegmentRun.exactSuffixCount
#check Chain.exactClaimCount
#check Chain.exactSuffixCount
#check Chain.completeDelayedSchedule

end tests.NebulaProductionBatchedGlobalFPrime
