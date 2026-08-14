import Nightstream.Protocol.Nebula.CheckedStepBatch
import tests.Axioms.Support

/-! Dependency audit for sequential checked-step batching. -/

/-- info: 'Nightstream.Protocol.Nebula.CheckedStepBatch.Batch.rowList_flatMap_accesses' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.CheckedStepBatch.Batch.rowList_flatMap_accesses

/-- info: 'Nightstream.Protocol.Nebula.CheckedStepBatch.Witness.step_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.CheckedStepBatch.Witness.step_exact
