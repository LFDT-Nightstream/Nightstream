import Nightstream.Implementation.Nebula.Production.Carrier.StreamingSuccessorStateBinding

/-! Regression surface for bounded-chunk successor-state binding. -/

set_option autoImplicit false

namespace tests.NebulaProductionSuccessorStateStreaming

open Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming

#check prefixFrame_lengthFor
#check prefixFrame_length_r26
#check stateFrame_length_r26
#check replayChunks_transcript
#check replayChunks_cursor
#check persistentFields_length
#check preCarryState_replays_prefixFrame
#check scheduled_preCarryState_eq
#check scheduled_outputState_eq
#check production_prefix_chunk_count_exact
#check production_state_chunk_count_exact
#check StateReplayCollision

end tests.NebulaProductionSuccessorStateStreaming
