import Nightstream.Protocol.Nebula.ProductionBatchedDelayedReverse

/-! Regression surface for reverse batch-aware F-prime extraction. -/

set_option autoImplicit false

namespace tests.NebulaProductionBatchedDelayedReverse

open Nightstream.Protocol.Nebula.ProductionBatchedDelayedReverse

#check VerifiedRun.append
#check delayedRun_to_segmentChain
#check segmentChain_iff_delayedRun

end tests.NebulaProductionBatchedDelayedReverse
