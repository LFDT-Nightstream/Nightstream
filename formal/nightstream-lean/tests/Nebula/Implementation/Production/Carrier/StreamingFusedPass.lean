import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFusedPass

/-! Regression surface for one fused state-binding and algebra pass. -/

set_option autoImplicit false

namespace tests.NebulaProductionStreamingFusedPass

open Nightstream.Implementation.Nebula.ProductionStreamingFusedPass

#check run_transcript
#check run_cursor
#check run_accumulator
#check run_schedule_exact
#check accepted_run_recovers_fold_or_collision_at
#check accepted_run_recovers_fold_or_collision
#check persistentFields_length

end tests.NebulaProductionStreamingFusedPass
