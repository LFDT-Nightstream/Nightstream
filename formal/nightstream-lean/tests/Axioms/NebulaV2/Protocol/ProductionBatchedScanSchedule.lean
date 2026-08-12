import Nightstream.Protocol.NebulaV2.ProductionBatchedScanSchedule
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaV2ProductionBatchedScanSchedule

open Nightstream.Protocol.NebulaV2.ProductionBatchedScanSchedule

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedScanSchedule.ConsumesList.claim_step_at' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConsumesList.claim_step_at

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedScanSchedule.ConsumesList.claim_segment_bounds_at' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConsumesList.claim_segment_bounds_at

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedScanSchedule.SegmentRun.suffix_step_at' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.suffix_step_at

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedScanSchedule.SegmentRun.suffix_segment_bounds_at' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.suffix_segment_bounds_at

end tests.Axioms.NebulaV2ProductionBatchedScanSchedule
