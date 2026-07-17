import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Replay

/-! Focused checks for complete typed later-round replay. -/

open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc

#check LaterRound.Replay.messages
#check LaterRound.Replay.stateBefore
#check LaterRound.Replay.stateBefore_next
#check LaterRound.Replay.incomingBound_all
#check LaterRound.Replay.messages_eq_prefix_append_final
#check LaterRound.Replay.run_eq_finalRun
#check LaterRound.Replay.run_eq_finalCallOutput
