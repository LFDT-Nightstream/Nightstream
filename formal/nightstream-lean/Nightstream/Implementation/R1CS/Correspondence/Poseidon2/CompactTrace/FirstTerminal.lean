import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Schema

set_option autoImplicit false
set_option maxRecDepth 65536
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement

open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core

theorem compact_first_terminal_schedule_exact :
    ∀ lane : Fin width, FirstTerminalScheduleExactAt lane := by
  unfold FirstTerminalScheduleExactAt
  native_decide

end Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
