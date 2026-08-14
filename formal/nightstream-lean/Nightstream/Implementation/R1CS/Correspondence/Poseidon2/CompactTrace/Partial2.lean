import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Schema

set_option autoImplicit false
set_option maxRecDepth 65536
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement

theorem compact_partial_schedule_exact_2 :
    ∀ offset : Fin 4, PartialScheduleExactAt (partialShardIndex2 offset) := by
  unfold PartialScheduleExactAt
  native_decide

end Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
