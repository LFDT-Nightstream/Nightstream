import Nightstream.Protocol.Nebula.ProductionBatchedScanSchedule

/-! Regression surface for flattened production-batch scan ordering. -/

set_option autoImplicit false

namespace tests.NebulaProductionBatchedScanSchedule

open Nightstream.Protocol.Nebula.ProductionBatchedScanSchedule

#check ConsumesList.claim_step_at
#check ConsumesList.claim_segment_bounds_at
#check SegmentRun.suffixes_length_exact
#check SegmentRun.suffix_step_at
#check SegmentRun.suffix_segment_bounds_at

end tests.NebulaProductionBatchedScanSchedule
