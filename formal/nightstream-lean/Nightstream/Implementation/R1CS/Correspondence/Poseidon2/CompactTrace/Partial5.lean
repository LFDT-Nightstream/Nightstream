import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Schema

set_option autoImplicit false
set_option maxRecDepth 65536
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement

theorem compact_partial_schedule_exact_5 :
    ∀ offset : Fin 2, PartialScheduleExactAt (partialShardIndex5 offset) := by
  unfold PartialScheduleExactAt
  intro offset
  fin_cases offset <;> decide

end Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
