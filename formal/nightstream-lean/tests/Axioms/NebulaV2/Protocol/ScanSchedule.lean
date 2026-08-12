import Nightstream.Protocol.NebulaV2.ScanSchedule
import tests.Axioms.Support

open Nightstream.Protocol.NebulaV2.ScanSchedule

/-- info: 'Nightstream.Protocol.NebulaV2.ScanSchedule.globalIndex_bijective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms globalIndex_bijective

/-- info: 'Nightstream.Protocol.NebulaV2.ScanSchedule.verifiedRun_claim_step_at' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms verifiedRun_claim_step_at

/-- info: 'Nightstream.Protocol.NebulaV2.ScanSchedule.verifiedRun_claim_segment_bounds_at' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms verifiedRun_claim_segment_bounds_at

/-- info: 'Nightstream.Protocol.NebulaV2.ScanSchedule.segment_claim_step_at' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms segment_claim_step_at

/-- info: 'Nightstream.Protocol.NebulaV2.ScanSchedule.segment_snapshot_global_index_at' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms segment_snapshot_global_index_at

/-- info: 'Nightstream.Protocol.NebulaV2.ScanSchedule.repeatedStepIndexes_is_not_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms repeatedStepIndexes_is_not_canonical
