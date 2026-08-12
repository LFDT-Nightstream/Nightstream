import Nightstream.Implementation.NebulaV2.Production.Memory.BatchPoseidonBinding

/-! Regression surface for the complete successor memory-batch digest. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionMemoryBatchPoseidonBinding

open Nightstream.Implementation.NebulaV2.ProductionMemoryBatchPoseidonBinding

#check frame_injective
#check batch_eq_or_poseidon_collision
#check schedule_table

example : fixedPrefix .e4 ≠ fixedPrefix .e8 := by decide

end tests.NebulaV2ProductionMemoryBatchPoseidonBinding
