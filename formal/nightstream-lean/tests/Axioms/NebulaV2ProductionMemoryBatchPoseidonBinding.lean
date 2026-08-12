import Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonBinding
import tests.Axioms.Support

/-! Dependency audit for the complete successor memory-batch digest. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonBinding.frame_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonBinding.frame_injective

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonBinding.schedule_table' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonBinding.schedule_table

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonBinding.batch_eq_or_poseidon_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonBinding.batch_eq_or_poseidon_collision
