import Nightstream.Implementation.Nebula.Production.Memory.BatchPoseidonBinding
import tests.Axioms.Support

/-! Dependency audit for the complete successor memory-batch digest. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryBatchPoseidonBinding.frame_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryBatchPoseidonBinding.frame_injective

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryBatchPoseidonBinding.schedule_table' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryBatchPoseidonBinding.schedule_table

/-- info: 'Nightstream.Implementation.Nebula.ProductionMemoryBatchPoseidonBinding.batch_eq_or_poseidon_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionMemoryBatchPoseidonBinding.batch_eq_or_poseidon_collision
