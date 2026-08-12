import Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime

/-! Regression surface for the candidate-specific global delayed chain. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionBatchedGlobalFPrime

open Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime

#check SegmentRun.exactClaimCount
#check SegmentRun.exactSuffixCount
#check Chain.exactClaimCount
#check Chain.exactSuffixCount
#check Chain.completeDelayedSchedule

end tests.NebulaV2ProductionBatchedGlobalFPrime
