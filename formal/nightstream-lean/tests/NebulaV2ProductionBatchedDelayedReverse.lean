import Nightstream.Protocol.NebulaV2.ProductionBatchedDelayedReverse

/-! Regression surface for reverse batch-aware F-prime extraction. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionBatchedDelayedReverse

open Nightstream.Protocol.NebulaV2.ProductionBatchedDelayedReverse

#check VerifiedRun.append
#check delayedRun_to_segmentChain
#check segmentChain_iff_delayedRun

end tests.NebulaV2ProductionBatchedDelayedReverse
