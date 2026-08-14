import Nightstream.Protocol.Nebula.ProductionBatchedFPrime
import tests.Axioms.Support

/-! Dependency audit for candidate-specific batched delayed consumption. -/

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedFPrime.ConsumesList.remaining_eq_length_add' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionBatchedFPrime.ConsumesList.remaining_eq_length_add

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedFPrime.ConsumesList.after_unique' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionBatchedFPrime.ConsumesList.after_unique

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedFPrime.Transition.decreases_by_exact_factor' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionBatchedFPrime.Transition.decreases_by_exact_factor

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedFPrime.Transition.before_active' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionBatchedFPrime.Transition.before_active

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedFPrime.VerifiedRun.full_segment_has_exact_batch_count' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionBatchedFPrime.VerifiedRun.full_segment_has_exact_batch_count
