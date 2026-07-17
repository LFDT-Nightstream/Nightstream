import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Execution

/-! Focused checks for indexed later-round semantic execution. -/

open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc

#check LaterRound.Execution.MessageBound
#check LaterRound.Execution.IncomingBound
#check LaterRound.Execution.PermutationBound
#check LaterRound.Execution.firstBoundary_eq_callInput
#check LaterRound.Execution.secondBoundary_eq_callInput
#check LaterRound.Execution.squeezeBoundary_eq_callInput
#check LaterRound.Execution.permutationBound_of_runRound
#check LaterRound.Execution.successor_eq_callOutputState
