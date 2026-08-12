import Nightstream.Protocol.NebulaV2.ProductionBatchedScanSchedule

/-! Regression surface for flattened production-batch scan ordering. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionBatchedScanSchedule

open Nightstream.Protocol.NebulaV2.ProductionBatchedScanSchedule

#check ConsumesList.claim_step_at
#check ConsumesList.claim_segment_bounds_at
#check SegmentRun.suffixes_length_exact
#check SegmentRun.suffix_step_at
#check SegmentRun.suffix_segment_bounds_at

end tests.NebulaV2ProductionBatchedScanSchedule
