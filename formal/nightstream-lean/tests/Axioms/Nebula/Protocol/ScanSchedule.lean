import Nightstream.Protocol.Nebula.ScanSchedule
import tests.Axioms.Support

open Nightstream.Protocol.Nebula.ScanSchedule

/-- info: 'Nightstream.Protocol.Nebula.ScanSchedule.globalIndex_bijective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms globalIndex_bijective

/-- info: 'Nightstream.Protocol.Nebula.ScanSchedule.verifiedRun_claim_step_at' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms verifiedRun_claim_step_at

/-- info: 'Nightstream.Protocol.Nebula.ScanSchedule.verifiedRun_claim_segment_bounds_at' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms verifiedRun_claim_segment_bounds_at

/-- info: 'Nightstream.Protocol.Nebula.ScanSchedule.segment_claim_step_at' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms segment_claim_step_at

/-- info: 'Nightstream.Protocol.Nebula.ScanSchedule.segment_snapshot_global_index_at' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms segment_snapshot_global_index_at

/-- info: 'Nightstream.Protocol.Nebula.ScanSchedule.repeatedStepIndexes_is_not_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms repeatedStepIndexes_is_not_canonical
