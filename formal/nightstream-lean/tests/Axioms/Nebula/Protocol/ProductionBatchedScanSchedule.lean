import Nightstream.Protocol.Nebula.ProductionBatchedScanSchedule
import tests.Axioms.Support

set_option autoImplicit false

namespace tests.Axioms.NebulaProductionBatchedScanSchedule

open Nightstream.Protocol.Nebula.ProductionBatchedScanSchedule

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedScanSchedule.ConsumesList.claim_step_at' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConsumesList.claim_step_at

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedScanSchedule.ConsumesList.claim_segment_bounds_at' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConsumesList.claim_segment_bounds_at

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedScanSchedule.SegmentRun.suffix_step_at' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.suffix_step_at

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedScanSchedule.SegmentRun.suffix_segment_bounds_at' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentRun.suffix_segment_bounds_at

end tests.Axioms.NebulaProductionBatchedScanSchedule
