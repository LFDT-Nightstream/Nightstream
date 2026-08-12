import Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime
import tests.Axioms.Support

/-! Dependency audit for candidate-specific batched delayed consumption. -/

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime.ConsumesList.remaining_eq_length_add' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime.ConsumesList.remaining_eq_length_add

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime.ConsumesList.after_unique' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime.ConsumesList.after_unique

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime.Transition.decreases_by_exact_factor' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime.Transition.decreases_by_exact_factor

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime.Transition.before_active' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime.Transition.before_active

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime.VerifiedRun.full_segment_has_exact_batch_count' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime.VerifiedRun.full_segment_has_exact_batch_count
